import argparse
import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ppga.models.actor_critic import Actor


# ── Dataset / DataLoader ──────────────────────────────────────────────────────

OBS_SHAPE         = (244,)
ACTION_SHAPE      = (17,)
NORMALIZE_OBS     = True
NORMALIZE_RETURNS = False


class EliteCheckpointDataset(Dataset):
    """PyTorch Dataset that reads elites directly from a pyribs archive pickle.

    Deserializes flat weight vectors into Actor state_dicts on-the-fly,
    exactly as the conversion script does, with no intermediate .pt files.

    Each item is a dict matching the format produced by convert_state_dict.py:
        {
            'model_state_dict':     OrderedDict,
            'obs_normalizer_state': OrderedDict | None,
            'objective':            float,
            'measures':             np.ndarray  [num_dims],
            'traj_length':          float,
            'elite_index':          int,
        }

    Args:
        archive_path: path to the pickled pyribs archive DataFrame (.pkl)
    """

    def __init__(self, archive_path: str):
        with open(archive_path, "rb") as f:
            df = pickle.load(f)

        self.solution_cols = [c for c in df.columns if c.startswith("solution_")]
        self.measures_cols = [c for c in df.columns if c.startswith("measures_")]

        # Store rows as a list for indexed access
        self.rows = [row for _, row in df.iterrows()]
        print(f"[EliteCheckpointDataset] Loaded {len(self.rows)} elites from {archive_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row         = self.rows[idx]
        metadata    = row["metadata"]
        flat_weights = row[self.solution_cols].values.astype(np.float32)

        model = Actor(OBS_SHAPE, ACTION_SHAPE, NORMALIZE_OBS, NORMALIZE_RETURNS)
        model.deserialize(flat_weights)

        # Zero logstd to match ppo.evaluate() — the CMA-ES sampled value was
        # never used during the evaluation that produced `objective`.
        model.actor_logstd.data = torch.zeros_like(model.actor_logstd.data)

        # Restore obs normalizer from metadata
        if NORMALIZE_OBS and metadata is not None:
            model.obs_normalizer.load_state_dict(metadata["obs_normalizer"])

        state_dict = model.state_dict()

        # Inject obs_normalizer keys if missing (matches conversion script logic)
        if NORMALIZE_OBS:
            norm_sd = model.obs_normalizer.state_dict()
            for k, v in norm_sd.items():
                full_key = f"obs_normalizer.{k}"
                if full_key not in state_dict:
                    state_dict[full_key] = v

        objective   = float(row["objective"])
        measures    = row[self.measures_cols].values.astype(np.float32)
        traj_length = float(metadata.get("traj_length", -1)) if metadata is not None else -1
        obs_normalizer_state = model.obs_normalizer.state_dict() if NORMALIZE_OBS else None

        return {
            "model_state_dict":     state_dict,
            "obs_normalizer_state": obs_normalizer_state,
            "objective":            objective,
            "measures":             measures,
            "traj_length":          traj_length,
            "elite_index":          idx,
        }


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


def get_elite_dataloader(archive_path: str,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0) -> DataLoader:
    """Returns a DataLoader over all elites in a pyribs archive pickle.

    Reads the archive directly — no intermediate .pt checkpoint files needed.

    Args:
        archive_path: path to the pickled pyribs archive DataFrame (.pkl)
        batch_size:   number of elites per batch
        shuffle:      whether to shuffle the order of elites
        num_workers:  DataLoader worker processes (0 = main process only,
                      recommended since __getitem__ constructs an Actor)

    Returns:
        DataLoader where each batch is a dict with keys:
            model_state_dict     — list[OrderedDict] of length batch_size
            obs_normalizer_state — list[OrderedDict | None] of length batch_size
            objective            — float tensor [batch_size]
            measures             — float tensor [batch_size, num_dims]
            traj_length          — float tensor [batch_size]
            elite_index          — int   tensor [batch_size]

    Example:
        loader = get_elite_dataloader('data/archive_df.pkl', batch_size=8)
        for batch in loader:
            for sd, obj in zip(batch['model_state_dict'], batch['objective']):
                model.load_state_dict(sd, strict=False)
                print(f'elite obj={obj:.2f}')
    """
    dataset = EliteCheckpointDataset(archive_path)
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