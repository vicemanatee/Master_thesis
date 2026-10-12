'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-09-29 14:05:45
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-12 14:42:49
FilePath: /Master_Thesis/src/data/__init__.py
'''
"""Data loading for the proteome benchmark and future phosphoproteome work."""

from .clinical import align_clinical_labels, load_clinical_labels
from .gene_set_loader import GeneSet, load_gmt
from .omics_loader import (
    LoadedMatrix,
    load_diann_matrix,
    load_matrix,
    load_phosphoproteome_matrix,
    load_proteome_matrix,
    load_proteome_rollup,
)
from .gene_set_loader import (
    load_hallmark,
    GeneSet,
    load_gmt
)

__all__ = [
    "GeneSet",
    "LoadedMatrix",
    "align_clinical_labels",
    "load_clinical_labels",
    "load_diann_matrix",
    "load_gmt",
    "load_hallmark",
    "load_matrix",
    "load_phosphoproteome_matrix",
    "load_proteome_matrix",
    "load_proteome_rollup",
]
