'''
Author: Li Yangqing liyangqingbuaa@gmail.com
Date: 2026-10-12 03:59:34
LastEditors: Li Yangqing liyangqingbuaa@gmail.com
LastEditTime: 2026-10-12 14:53:44
FilePath: /Master_Thesis/src/pathway/map_annotation.py
Description: mapping feature metadata to gene-set annotations

Need to add feat to store the mapping results. 
'''


ANNOTATION = {
  'unknown': 'Unknown',
  'none': 'None',
  'partial': 'Partial',
  'all': 'All',
}

MISSINGMARKERS = {'', 'NA', 'N/A', 'na', 'n/a', 'nan', 'NaN', 'null', 'Null', 'NULL'}

import pandas as pd

from data import GeneSet
    

def _feature_metadata_to_gene_sets(
        feature_metadata: pd.DataFrame,
):
    """
    Map feature metadata to gene-set annotations.
    """

    feature_gene_sets = {}

    for feature_id, raw_gene in feature_metadata['Genes'].items():
        if pd.isna(raw_gene):
            feature_gene_sets[feature_id] = set()
            continue
        else:
            parts = raw_gene.split(";")
            cleaned = [gene.strip() for gene in parts]
            feature_gene_sets[feature_id]= {gene for gene in cleaned if gene not in MISSINGMARKERS}
    return feature_gene_sets
    

def mapping_annotation(
    feature_metadata: pd.DataFrame,
    gene_sets: dict[str, GeneSet],):
    """
    Map feature metadata to gene-set annotations.

    feature_metadata: DataFrame containing feature metadata. from diann or rollup
    gene_sets: Dictionary of gene sets loaded from a GMT file.
    Returns a DataFrame with the mapping results.
    """

    feature_gene_sets = _feature_metadata_to_gene_sets(feature_metadata)

    annotation = pd.DataFrame(index=feature_metadata.index, columns = gene_sets.keys())

    for feature_id, feature_genes in feature_gene_sets.items():
        if not feature_genes:
            annotation.loc[feature_id] = {set_name: ANNOTATION['unknown'] for set_name in gene_sets.keys()}
            continue
        for set_name, gene_set in gene_sets.items():
            pathway_genes = gene_set['genes']

            matched_gene = feature_genes & pathway_genes
            unmatched_gene = feature_genes - pathway_genes

            if len(matched_gene) == 0:
                annotation.loc[feature_id, set_name] = ANNOTATION['none']
            elif (len(unmatched_gene) > 0) and (len(matched_gene) > 0):
                annotation.loc[feature_id, set_name] = ANNOTATION['partial']
            elif len(unmatched_gene) == 0:
                annotation.loc[feature_id, set_name] = ANNOTATION['all']

    return annotation

    

    