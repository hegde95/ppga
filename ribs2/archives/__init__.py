"""Archives store solutions found by a QD algorithm.

.. autosummary::
    :toctree:

    ribs2.archives.GridArchive
    ribs2.archives.CVTArchive
    ribs2.archives.SlidingBoundariesArchive
    ribs2.archives.ArchiveBase
    ribs2.archives.AddStatus
    ribs2.archives.Elite
    ribs2.archives.EliteBatch
    ribs2.archives.ArchiveDataFrame
    ribs2.archives.ArchiveStats
"""
from ribs2.archives._add_status import AddStatus
from ribs2.archives._archive_base import ArchiveBase
from ribs2.archives._archive_data_frame import ArchiveDataFrame
from ribs2.archives._archive_stats import ArchiveStats
from ribs2.archives._cvt_archive import CVTArchive
from ribs2.archives._elite import Elite, EliteBatch
from ribs2.archives._grid_archive import GridArchive
from ribs2.archives._sliding_boundaries_archive import SlidingBoundariesArchive

__all__ = [
    "GridArchive",
    "CVTArchive",
    "SlidingBoundariesArchive",
    "ArchiveBase",
    "AddStatus",
    "Elite",
    "ArchiveDataFrame",
    "ArchiveStats",
]
