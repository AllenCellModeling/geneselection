import scanpy.api as sc
import os
from ..utils.data import split_anndata, write_splits
import pandas as pd
import numpy as np


def preprocess_cardio(
    adata_in,
    nz_thresh=0.05,
    transform=None,
    f_coding_genes="/allen/aics/modeling/rorydm/projects/geneselection/all_human_protein_genes_and_exons.txt",
    days="all",
):

    # load list of protein coding genes
    df = pd.read_csv(f_coding_genes, delimiter="\t")
    coding_genes = [str(g) + "_HUMAN" for g in df["Gene name"].unique()]

    # filter our data for only protein coding genes
    cols = np.array([c for c in adata_in.var.index if c in coding_genes])
    adata = adata_in[:, cols]

    # filter for the days we want
    if days is not "all":
        adata = adata[adata.obs["day"].isin(days)].copy()

    # filter for genes that apoear in at least x frac of cells
    gene_nz_freq = (adata.X > 0).mean(axis=0)
    adata = adata[:, cols[gene_nz_freq > nz_thresh]]

    if transform is not None:
        adata.X = transform(adata.X)

    return adata.copy()


def load(
    split="train",
    original_fpath="/allen/aics/modeling/data/scRNAseq_SeeligCollaboration/data_for_modeling/scrnaseq_cardio_20181129.h5ad",
    cache_dir="data_cache",
    cache=True,
    selected_genes_path=None,
    threshold=0,
    days="all",
):
    """
    Load requested split of cardio data, where the whole dataset originated at original_fpath.
    Looks for local cache of split, and if it can't find that, makes a split on the fly.
    If cache=True, caches the result in cache_dir for next time.
    Loads raw count values.
    """

    original_fname = os.path.basename(original_fpath)
    original_bname, original_ext = os.path.splitext(original_fname)
    target_fname = "{0}_{1}{2}".format(original_bname, split, original_ext)
    target_fpath = os.path.join(cache_dir, target_fname)

    if not os.path.exists(target_fpath):
        adata_in = sc.read_h5ad(original_fpath)
        adata_raw = sc.AnnData(
            X=adata_in.raw.X.todense(),
            obs=adata_in.obs,
            var=adata_in.var,
            uns=adata_in.uns,
        )
        split_inds, split_adata = split_anndata(adata_raw)
        if cache:
            write_splits(
                split_inds_dict=split_inds,
                split_adata_dict=split_adata,
                basename=original_bname,
                out_dir=cache_dir,
            )

    adata = sc.read_h5ad(target_fpath)

    adata = preprocess_cardio(
        adata,
        nz_thresh=threshold,
        transform=None,
        f_coding_genes=selected_genes_path,
        days=days,
    )

    return adata
