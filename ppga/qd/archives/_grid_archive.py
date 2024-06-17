"""Contains the GridArchive."""
import numpy as np
import ribs.archives


class GridArchive(ribs.archives.GridArchive):
    """Differs from pyribs GridArchive by the addition of a reward_offset."""

    def __init__(self, *args, reward_offset=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._reward_offset = reward_offset

    @property
    def offset_qd_score(self):
        """Computes the QD score accounting for the offset introduced by the
        reward."""
        if self._reward_offset is None:
            raise ValueError(
                "Cannot compute offset_qd_score for None reward_offset")

        objs = self.data("objective")
        traj_lengths = np.asarray(
            [m["traj_length"] for m in self.data("metadata")])
        offset_score = np.sum(self._reward_offset * traj_lengths)
        offset_qd_score = np.sum(objs) + offset_score

        return offset_qd_score
