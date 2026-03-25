import argparse
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ppga.models.actor_critic import Actor


# ── Dataset / DataLoader ──────────────────────────────────────────────────────

class EliteCheckpointDataset(Dataset):
    """PyTorch Dataset over a directory of elite_checkpoint_*.pt files.

    Each item is the full checkpoint dict:
        {
            'model_state_dict':     OrderedDict,
            'obs_normalizer_state': OrderedDict | None,
            'objective':            float tensor,
            'measures':             float tensor  [num_dims],
            'traj_length':          float tensor,
            'elite_index':          int tensor,
        }
    """

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = checkpoint_dir
        paths = [
            os.path.join(checkpoint_dir, f)
            for f in sorted(os.listdir(checkpoint_dir))
            if f.startswith('elite_checkpoint_') and f.endswith('.pt')
        ]
        assert len(paths) > 0, f'No elite_checkpoint_*.pt files in {checkpoint_dir}'
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        return torch.load(self.paths[idx], map_location='cpu')


def elite_collate_fn(batch):
    """Custom collate: keeps state_dicts as a list (can't stack OrderedDicts),
    stacks scalar/tensor fields normally."""
    return {
        'model_state_dict':     [item['model_state_dict']     for item in batch],
        'obs_normalizer_state': [item.get('obs_normalizer_state') for item in batch],
        'objective':    torch.tensor([item['objective']    for item in batch]),
        'measures':     torch.stack([torch.tensor(item['measures'])  for item in batch]),
        'traj_length':  torch.tensor([item['traj_length']  for item in batch]),
        'elite_index':  torch.tensor([item['elite_index']  for item in batch]),
    }


def get_elite_dataloader(checkpoint_dir: str,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
    """Returns a DataLoader over all elite checkpoints in checkpoint_dir.

    Args:
        checkpoint_dir: directory containing elite_checkpoint_*.pt files
        batch_size:     number of elites per batch
        shuffle:        whether to shuffle the order of elites
        num_workers:    DataLoader worker processes (0 = main process only)

    Returns:
        DataLoader where each batch is a dict with keys:
            model_state_dict     — list[OrderedDict] of length batch_size
            obs_normalizer_state — list[OrderedDict | None] of length batch_size
            objective            — float tensor [batch_size]
            measures             — float tensor [batch_size, num_dims]
            traj_length          — float tensor [batch_size]
            elite_index          — int   tensor [batch_size]

    Example usage:
        loader = get_elite_dataloader('data/checkpoints', batch_size=8)
        for batch in loader:
            for sd, obj in zip(batch['model_state_dict'], batch['objective']):
                model.load_state_dict(sd, strict=False)
                print(f'elite obj={obj:.2f}')
    """
    dataset = EliteCheckpointDataset(checkpoint_dir)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      collate_fn=elite_collate_fn)


# ── Conversion script ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--archive_path', type=str, required=True)
    parser.add_argument('--output_dir',   type=str, required=True)
    args = parser.parse_args()

    with open(args.archive_path, "rb") as f:
        archive = pickle.load(f)

    df = archive
    print(f'Archive loaded with {len(df)} elites')
    os.makedirs(args.output_dir, exist_ok=True)

    obs_shape         = (244,)
    action_shape      = (17,)
    normalize_obs     = True
    normalize_returns = False
    solution_cols     = [c for c in df.columns if c.startswith("solution_")]
    measures_cols     = [c for c in df.columns if c.startswith("measures_")]

    for i, (_, elite_row) in enumerate(df.iterrows()):
        metadata     = elite_row["metadata"]
        flat_weights = elite_row[solution_cols].values.astype(np.float32)

        model = Actor(obs_shape, action_shape, normalize_obs, normalize_returns)
        model.deserialize(flat_weights)

        # Override actor_logstd with zeros to match ppo.evaluate() behavior.
        # train_ppga.py lines 570-571 overwrote each branched agent's logstd with
        # mean_agent.actor_logstd before evaluation — the CMA-ES sampled value in
        # solution_cols was never used during the evaluation that produced `objective`.
        model.actor_logstd.data = torch.zeros_like(model.actor_logstd.data)

        # Restore obs_normalizer from metadata
        if normalize_obs and metadata is not None:
            model.obs_normalizer.load_state_dict(metadata["obs_normalizer"])
            mean_sum = model.obs_normalizer.obs_rms.mean.abs().sum().item()
            if mean_sum < 1e-6:
                print(f"  WARNING elite {i}: obs_rms.mean is all zeros after loading!")

        state_dict = model.state_dict()

        if normalize_obs:
            norm_sd = model.obs_normalizer.state_dict()
            for k, v in norm_sd.items():
                full_key = f"obs_normalizer.{k}"
                if full_key not in state_dict:
                    state_dict[full_key] = v
                    print(f"  [elite {i}] Injected {full_key} into state_dict (was missing)")

        state_dict["actor_logstd"] = state_dict["actor_logstd"]

        objective   = float(elite_row["objective"])
        measures    = elite_row[measures_cols].values.astype(np.float32)
        traj_length = float(metadata.get("traj_length", -1)) if metadata is not None else -1
        obs_normalizer_state = model.obs_normalizer.state_dict() if normalize_obs else None

        output_path = os.path.join(args.output_dir, f'elite_checkpoint_{i}.pt')
        torch.save({
            "model_state_dict":     state_dict,
            "obs_normalizer_state": obs_normalizer_state,
            "objective":            objective,
            "measures":             measures,
            "traj_length":          traj_length,
            "elite_index":          i,
        }, output_path)
        print(f'[{i+1}/{len(df)}] Saved {output_path} '
            f'(obj={objective:.1f}, traj_len={traj_length:.0f}, '
            f'obs_mean[:3]={model.obs_normalizer.obs_rms.mean[:3].tolist()})')