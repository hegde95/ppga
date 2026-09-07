"""Shared helpers for inspecting and rendering MJLab policy archives."""

from __future__ import annotations

import numpy as np

from ppga.models.actor_critic import Actor


def success_gated_objectives(objectives, metadata, min_success_rate,
                             threshold_min):
    """Make policies below a success threshold ineligible for insertion."""
    objectives = np.asarray(objectives)
    if min_success_rate <= 0:
        return objectives
    success_rates = np.asarray([
        data.get('episode_success_rate', np.nan)
        if isinstance(data, dict) else np.nan for data in metadata
    ], dtype=np.float64)
    if not np.isfinite(success_rates).all():
        raise ValueError(
            'archive_min_success_rate requires episode_success_rate metadata')
    gated = objectives.copy()
    objective_dtype = (objectives.dtype
                       if np.issubdtype(objectives.dtype, np.floating)
                       else np.dtype(np.float64))
    rejection_value = (
        np.nextafter(np.asarray(threshold_min, dtype=objective_dtype),
                     np.asarray(-np.inf, dtype=objective_dtype)).item()
        if np.isfinite(threshold_min) else -1e30)
    gated[success_rates < min_success_rate] = rejection_value
    return gated


def archive_solution_columns(archive):
    columns = [
        column for column in archive.columns if column.startswith('solution_')
    ]
    if not columns:
        raise ValueError('Archive has no solution columns')
    return columns


def restore_archive_actor(row, cfg, solution_columns=None):
    """Restore policy parameters and per-elite normalization state."""
    if solution_columns is None:
        solution_columns = [
            column for column in row.index if column.startswith('solution_')
        ]
    actor = Actor(
        cfg.obs_shape, cfg.action_shape, cfg.normalize_obs,
        cfg.normalize_returns, cfg.action_transform,
        getattr(cfg, 'action_std_parameterization', 'log'),
        hidden_dims=getattr(cfg, 'actor_hidden_dims',
                            (400, 200, 100))).deserialize(
                                row[solution_columns].to_numpy(
                                    dtype=np.float32))
    metadata = row.get('metadata')
    if isinstance(metadata, dict):
        if cfg.normalize_obs and 'obs_normalizer' in metadata:
            actor.obs_normalizer.load_state_dict(metadata['obs_normalizer'])
        if cfg.normalize_returns and 'return_normalizer' in metadata:
            actor.return_normalizer.load_state_dict(
                metadata['return_normalizer'])
    return actor


def metadata_success_rate(row):
    metadata = row.get('metadata')
    if not isinstance(metadata, dict):
        return float('nan')
    return float(metadata.get('episode_success_rate',
                              metadata.get('success_rate', float('nan'))))


def select_representative_elites(archive, count=5, min_success_rate=0.5):
    """Select the best elite, then descriptor-space farthest points."""
    if count < 1:
        raise ValueError('count must be positive')
    candidates = archive[
        archive.apply(
            lambda row: metadata_success_rate(row) >= min_success_rate,
            axis=1)
    ]
    if candidates.empty:
        raise ValueError(
            f'Archive has no elites with success rate >= {min_success_rate}')

    count = min(int(count), len(candidates))
    best_index = candidates['objective'].astype(float).idxmax()
    selected = [best_index]
    remaining = set(candidates.index) - {best_index}
    measure_columns = sorted(
        [column for column in candidates.columns
         if column.startswith('measures_')],
        key=lambda name: int(name.rsplit('_', 1)[1]))
    points = candidates[measure_columns].to_numpy(dtype=np.float64)
    index_to_position = {
        index: position for position, index in enumerate(candidates.index)
    }

    while len(selected) < count:
        selected_points = points[
            [index_to_position[index] for index in selected]]
        best_candidate = None
        best_key = None
        for index in sorted(remaining):
            point = points[index_to_position[index]]
            min_distance = np.square(selected_points - point).sum(
                axis=1).min()
            key = (float(min_distance),
                   float(candidates.loc[index, 'objective']))
            if best_key is None or key > best_key:
                best_key = key
                best_candidate = index
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return [(index, candidates.loc[index],
             'best_objective' if position == 0 else f'diverse_{position:02d}')
            for position, index in enumerate(selected)]
