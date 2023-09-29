"""
Utility functions related to the processing and analysis of VCFs.
Specifically, initially designed for working with germline VCFs 
in the prostate cancer dataset.

Author: Gwen Miller <gwen_miller@g.harvard.edu>
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt


logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
# logger.setLevel(logging.INFO)

##############################
## working with VCFs
##############################



# Define a function to binarize the genotype values
def binarize(value):
    if value == "./.":
        return 0
    else:
        return 1

    
def binarize_burden_mat(value):
    if value != 0.:
        return 1
    else:
        return 0
    

def binarize_vcf(df):
    df = df.applymap(utils.binarize)
    return df


def get_sample_col_names_from_VCF_matrix(df):
    return [col for col in df.columns if col.endswith(".GT")]


def get_sample_col_names_from_VCF(df):
    # exclude any sample-related columns
    return [col for col in df.columns if col.endswith(('.AD', '.DP', '.GQ', '.GT', '.VAF', '.PL'))]

def get_non_sample_col_names_from_VCF(df):
    # exclude any sample-related columns
    return [col for col in df.columns if not col.endswith(('.AD', '.DP', '.GQ', '.GT', '.VAF', '.PL'))]


def get_sample_names_from_VCF(df):
    logging.info("Extracting the sample IDs from the column names")
    sample_ids = [col.split(".GT")[0] for col in df.columns if col.endswith(".GT")]
    logging.info("We found {} sample_ids".format(len(sample_ids)))
    return sample_ids


def get_sample_cols_from_VCF(df):
    logging.debug("restricting to just the genotype columns (the ones that end in .GT)...")
    df = df.filter(regex=".GT$")
    df.columns = df.columns.map(lambda x: x[:-3] if x.endswith('.GT') else x)
    logging.info(f"vcf shape: {df.shape}")
    return df

def load_sample_cols_from_VCF(annot_vcf_f):
    logging.info("loading the VCF")
    df = pd.read_csv(annot_vcf_f, sep="\t")
    df = get_sample_cols_from_VCF(df)
    return df


def load_vcf_and_format_as_binary_df(vcf_f, pathogenic_vars_only=False): # TODO: is this identical to calling load_sample_cols_from_VCF? No, we also binarize here
    logging.info(f"loading VCF at {vcf_f}")
    vcf = pd.read_csv(vcf_f, sep="\t", low_memory=False).set_index("Uploaded_variation")
    
    if pathogenic_vars_only:
        vcf = restrict_vcf_to_patho_only(vcf)

    logging.info("restricting to just the genotype columns (the ones that end in .GT) and binarizing...")
    vcf = vcf.filter(regex=".GT$").applymap(utils.binarize)
    vcf.columns = vcf.columns.map(lambda x: x[:-3] if x.endswith('.GT') else x)
    logging.debug(f"vcf shape: {vcf.shape}")
    return vcf
    

def load_VCF_annotation_cols(annot_vcf_f, columns_with_annotations=None):
    logging.info("reading in the columns with annotation info...")
    if columns_with_annotations is None:
        df = pd.read_csv(annot_vcf_f, sep="\t", low_memory=False)
        df = get_variant_metadata_from_VCF(df)
    else:
        df = pd.read_csv(annot_vcf_f, sep="\t",
                     usecols = columns_with_annotations)
    logging.info("finished reading in the file.")
    logging.debug(f"The shape of the df is {df.shape}. Num total variants = {len(df)}.")
    return df


def get_variant_metadata_from_VCF(df):
    logging.info("grabbing the columns with variant metadata and information...")
    variant_annotation_cols = get_non_sample_col_names_from_VCF(df)

    # use filter to keep the selected columns
    variant_metadata = df.loc[:, variant_annotation_cols]
    logging.debug(f"Head of the variant metadata DF:")
    logging.debug(variant_metadata.head())
    return variant_metadata


def restrict_vcf_to_patho_only(vcf):
    logging.info("making VCF with pathogenic variants only")
    genes = vcf.SYMBOL.unique().tolist()
    patho_vcf = subset_to_pathogenic_only(vcf, genes)
    patho_vcf = patho_vcf.set_index("Uploaded_variation")
    return patho_vcf


def load_full_VCF(annot_vcf_f, pathogenic_vars_only=False):
    logging.info(f"reading in the annotated VCF at {annot_vcf_f}...")
    df = pd.read_csv(annot_vcf_f, sep="\t", low_memory=False)
    
    if pathogenic_vars_only:
        df = restrict_vcf_to_patho_only(df)
    
    # logging.debug("setting SYMBOL as the index")
    # df = df.set_index("SYMBOL", drop=False) # TODO: consider uncommenting
    # logging.debug("sorting the index to help speed up future selection queries")
    # df = df.sort_index()
    logging.info("finished loading the file.")
    logging.debug(f"The shape of the df is {df.shape}. Num total variants = {len(df)}.")
    return df


def n_variants_per_sample_from_vcf(vcf, savefig_f = False,
                                   plot_title="# variants per sample", 
                                   plot_id=None,
                                   plot_xlabel="# variants/sample",
                                   ax=None):
    """
    Inputs:
    - Expect vcf is a pandas DF, variants x samples
    - savefig_f: directory to save the image to, if desired
    """
    logging.info(f"working with a VCF with {vcf.shape[1]} samples and {vcf.shape[0]} rows (variants, genes, etc)")
    n_variants_per_sample = vcf.sum(axis=0)
    
    logging.info("building plot")
    if ax is None:
        fig.ax = plt.subplots()
    ax.hist(n_variants_per_sample.tolist())

    if plot_id is not None:
        fig.suptitle(plot_id)
    ax.set_title(plot_title)
    ax.set_xlabel(plot_xlabel)
    
    if savefig_f:
        utils.savefig(savefig_f)
    return fig

def n_samples_per_variant_from_vcf(vcf, savefig_f = False, 
                                   plot_title="# samples per variant (dataset MAF=0.01 in red)", 
                                   plot_id=None,
                                   plot_xlabel="log2(# samples/variant)",
                                   ax=None):
    """
    Inputs:
    - Expect vcf is a pandas DF, variants x samples
    - savefig_f: directory to save the image to, if desired
    """
    logging.info(f"working with a VCF with {vcf.shape[1]} samples and {vcf.shape[0]} rows (variants, genes, etc)")
    n_samples_per_variant = vcf.sum(axis=1)

    logging.info("building plot")
    if ax is None:
        fig,ax = plt.subplots()
    
    # plot vertical line corresponding to MAF = 0.01
    maf_1percent = vcf.shape[1]*0.01
    ax.hist(np.log2(n_samples_per_variant.tolist()))
    ax.axvline(x=np.log2(maf_1percent), color="red")
    
    if plot_id is not None:
        plt.suptitle(plot_id)
    ax.set_title(plot_title)
    ax.set_xlabel(plot_xlabel)
    
    if savefig_f:
        utils.savefig(savefig_f)
    return fig


def filter_annotated_vcf_by_gene_list(annot_vcf, annot_vcf_f, gene_list,
                                     save_filtered_df_path = "annotated_vcf_filtered_to_genes_of_interest.txt"):
    """
    Filter down an annotated VCF file to keep (1) just the variants in the gene list and (2) only the genotypes information for each sample (the .GT column)
    This function does NOT rely on chunking, and may not be very efficient with large VCF files.
    """
    assert annot_vcf_f.endswith(".txt") or annot_vcf_f.endswith(".txt.gz"), "Require file that ends with .txt or .txt.gz"

    logging.info("\n1. determine cols of interest and load just these columns...")
    logging.info("Generating list of columns we care about...")
    df = pd.read_csv(annot_vcf_f, nrows=1, sep="\t") 
    
    columns_with_annotations  = get_non_sample_col_names_from_VCF(df)
    columns_with_sample_gts = get_sample_col_names_from_VCF_matrix(df) # grab the ".GT" columns
    logging.info(f"The number of samples we have data for is {len(columns_with_sample_gts)}.")
    columns_to_use = np.array(columns_with_annotations).reshape(-1).tolist() + np.array(columns_with_sample_gts).reshape(-1).tolist()
    assert len(columns_to_use) == len(columns_with_annotations) + len(columns_with_sample_gts)
    
    logging.info("using a pre-loaded VCF DF subsetted to the annotation columns...")
    df = annot_vcf
    
    logging.info("\n2. determine rows (aka variants) of interest...")
    logging.info("Determine which rows contain variants in genes of interest")
    assert all([i in df.SYMBOL.tolist() for i in gene_list]), "Some of the genes you want aren't in the DF's SYMBOL column"
    filtered_df = df[df["SYMBOL"].isin(gene_list)]
    logging.info(f"We have filtered down to {len(filtered_df)} rows.")
    rows_to_use = filtered_df.index
    
    logging.info("\n3. create DF with just the rows and columns of interest...")
    logging.info("Filter the original df by columns and rows to determine which patients")
    df = pd.read_csv(annot_vcf_f, sep="\t",
                     usecols = columns_to_use)
    df = df.iloc[rows_to_use]
    logging.info(f"Restricted to the genes of interest, we have shape {df.shape}")
    
    logging.info("\n4. save file")
    logging.info(f"Saving down the filtered DF here {save_filtered_df_path}")
    df.to_csv(save_filtered_df_path, index=False) 
    return df


def filter_VCF_chunk(df, gene_list):
    assert "SYMBOL" in df.columns.tolist(), f"SYMBOL isn't in the columns, which are \n{df.columns.tolist()}"
    logging.debug("Determine which rows contain variants in genes of interest")
    filtered_df = df[df["SYMBOL"].isin(gene_list)]
    if len(filtered_df)>0:
        logging.debug(f"df.SYMBOL.value_counts().index.isin(gene_list): {df.SYMBOL.value_counts().index.isin(gene_list)}")
        # assert sum(df.SYMBOL.value_counts().index.isin(gene_list)) < 2, "we were only getting filtered DFs from 1 gene, but at this stage we have more than 1 of our target genes" # TODO: uncomment
    return filtered_df


def filter_annotated_vcf_by_gene_list_chunking(annot_vcf_f, gene_list,
                                     save_filtered_df_path = "annotated_vcf_filtered_to_genes_of_interest.txt", chunksize=20000):
    """
    Filter down a (large) annotated VCF file to keep (1) just the variants in the gene list and (2) only the genotypes information for each sample (the .GT column).
    This function DOES rely on chunking, making it more efficient for large VCF files.
    """
    assert annot_vcf_f.endswith(".txt") or annot_vcf_f.endswith(".txt.gz"), "Require file that ends with .txt or .txt.gz"

    # try the chunked version: load whole DF (all cols, then loop over row chunks). Filter rows by SYMBOL col
    logging.debug(f"gene_list: {gene_list}")
    list_of_dfs = []
    with pd.read_csv(annot_vcf_f, chunksize=chunksize, sep = "\t", low_memory=False) as reader: # TODO: uncomment
    # with pd.read_csv(annot_vcf_f, chunksize=chunksize, low_memory=False) as reader: # TODO: used this for temporary check.
        for i,chunk in tqdm(enumerate(reader)):
            logging.debug(f"working on filtering chunk {i} with shape {chunk.shape}:")
            filtered_df = filter_VCF_chunk(chunk, gene_list)
            if len(filtered_df) >0:
                logging.debug(f"In this chunk, we have filtered down to {len(filtered_df)} rows.")
                logging.debug(f"filtered df:\n{filtered_df.head()}")
                list_of_dfs.append(filtered_df)
    if len(list_of_dfs)==0:
        return f"none of the genes were found in the DF: {gene_list}"
    
    logging.debug("shapes of the filtered chunks:")
    for i,l in enumerate(list_of_dfs):
        logging.debug(l.shape)
        
    df = pd.concat(list_of_dfs) 
    
    logging.debug(f"Restricted to the genes of interest, we have shape {df.shape}")
    
    logging.info(f"\n4. Saving the filtered DF to {save_filtered_df_path}")
    df.to_csv(save_filtered_df_path, index=False, sep="\t") 
    return df


def remove_variants_too_common_in_dataset(vcf_f, pathogenic_vars_only=True, remove_vars_above_threshold = 0.05):
    """
    TODO: currently non-functional
    """    
    subset_id = filename(vcf_f) # use the filename as the subset identifier
    logging.info(f"working on gene subset {subset_id}\nfile {vcf_f}")
    vcf = pd.read_csv(vcf_f, sep="\t", low_memory=False).set_index("Uploaded_variation")
    
    if pathogenic_vars_only:
        logging.info("restricting to pathogenic variants only")
        vcf = subset_to_pathogenic_only(vcf)
        vcf = vcf.set_index("Uploaded_variation")
    
    logging.debug("restricting to just the genotype columns (the ones that end in .GT) and binarizing...")
    vcf = vcf.filter(regex=".GT$").applymap(binarize)
    vcf.columns = vcf.columns.map(lambda x: x[:-3] if x.endswith('.GT') else x)
    logging.info(f"vcf shape: {vcf.shape}")

    N = vcf.shape[1]
    n_variants_per_sample = vcf.sum(axis=0)
    n_samples_per_variant = vcf.sum(axis=1)
    vcf['n_samples_with_variant'] = n_samples_per_variant
    vcf['percent_of_samples_with_variant'] = n_samples_per_variant/N
    vcf = relocate(vcf, ['n_samples_with_variant', 'percent_of_samples_with_variant'])

    remove_df = vcf[vcf['percent_of_samples_with_variant'] >= remove_vars_above_threshold].sort_values(by=["n_samples_with_variant"], ascending=False).copy()
    logging.info(f"here are the {len(remove_df)} variants that we filter out")
    print(remove_df)

    logging.debug("only keeping the variants under the threshold")
    vcf = vcf[vcf['percent_of_samples_with_variant'] < remove_vars_above_threshold].copy()
    
    logging.info("returning the filtered VCF and the DF of variants we removed")
    return vcf, remove_df

def subset_to_pathogenic_only(df, gene_list=None, save_f=None, remove_vars_above_threshold = 0.05):
    """
    Filter a VCF from all variants to just those deemed pathogenic by the filters defined in the `conflicting_filter_criteria`, which is a function in the `filter_to_pathogenic_variants` script (imported above).
    """
    if gene_list is None: # default to keeping all the genes
        gene_list = df.SYMBOL.unique().tolist()

    logging.debug("running variant selection workflow...")
    logging.debug(f"gene list: {gene_list}")
    pathogenic_variants_df = patho.variant_selection_workflow(
        vep_df=df.reset_index(),
        filter_criteria=patho.conflicting_filter_criteria,
        genes_to_subset=gene_list,
        clinsig_col = 'ClinVar_updated_2021Jun_CLNSIG',
        clin_conflict = 'ClinVar_updated_2021Jun_CLNSIGCONF')
    
    logging.debug(f"pathogenic_variants_df.shape: {pathogenic_variants_df.shape}")
    
    logging.debug(pathogenic_variants_df.head())
    logging.info(f"We filtered down to {len(pathogenic_variants_df)} 'pathogenic' variants from the original {len(df)}.")
    logging.debug(df.shape)
    logging.debug(pathogenic_variants_df.shape)
    
#     if remove_vars_above_threshold <= 1:
#         # logging.info(f"removing any variants that occur in gnomAD with high-frequency (above the specified threshold of {remove_vars_above_threshold}")
#         # pathogenic_variants_df = patho.subset_to_low_frequency(pathogenic_variants_df, freq=remove_vars_above_threshold,
#         #                                                       verbose=True)
    
#         logging.info(f"removing variants that occur in more than {remove_vars_above_threshold} of our dataset's samples - these are likely artifacts.")
#         # remove_variants_too_common_in_dataset(df, pathogenic_vars_only=True, remove_vars_above_threshold = 0.05)
        
        
    if save_f is not None:
        pathogenic_variants_df.to_csv(save_f, index=False, sep="\t")
    return pathogenic_variants_df
    
    
def make_binary_genotype_mat_from_VCF(df):
    # Apply the binarize function to the columns with ".GT" at the end of their names
    binary_genotype = df.filter(regex=".GT$").applymap(binarize)
    logging.debug("removing the .GT from the sample names")
    binary_genotype.columns = binary_genotype.columns.map(lambda x: x[:-3] if x.endswith('.GT') else x)
    logging.debug(f"Head of the binary genotype matrix:")
    logging.debug(binary_genotype.head())
    return binary_genotype


def convert_binary_var_mat_to_gene_level_mat(binary_genotypes, variant_metadata, binary_output = False):
    """
    Description of process to go from a variant-level genotypes matrix to a (binary) gene-level genotypes matrix.
    1. Use list of vars included in the variant level matrix to filter the variant metadata df.
    2. Group the filtered variant metadata df by gene. 
    - For each gene, get associated variants.
    - Filter the variant level matrix by these rows.
    - Sum across the rows to get a per-sample count of the number of variants in this gene.
    - Set this as a row in the `gene_burden_matrix`
    3. Create the final output matrix by concatenating each row of gene information (gene_burden_mat): n_genes by n_samples.
    4. If binary_output is True, then binarize the output.
    """
    logging.info("1. Use list of vars included in the variant level matrix to filter the variant metadata df.")
    vars_to_use = binary_genotypes.index.tolist()
    logging.debug(len(vars_to_use))
    variant_metadata = variant_metadata.loc[vars_to_use,:]
    logging.debug(f"filtered variant_metdata.shape: {variant_metadata.shape}")
    
    logging.info("2. Group the filtered variant metadata df by gene, get gene-level counts")
    logging.info("creating one row per gene with the variant counts for each sample")
    
    gene_burden_rows = []
    genes = []
    for gene in set(variant_metadata.SYMBOL.tolist()):
        curr_var_data = variant_metadata[variant_metadata.SYMBOL==gene]
        curr_vars = curr_var_data.index.tolist()
        logging.debug(f"for gene {gene} we have {len(curr_vars)} variants: {curr_vars}")
        logging.debug(f"binary_genotypes.loc[curr_vars,:]: {binary_genotypes.loc[curr_vars,:].shape}")
        curr_gene_level_info = binary_genotypes.loc[curr_vars,:].sum(axis=0)
        logging.debug(f"curr_gene_level_info: {curr_gene_level_info.shape}")
        gene_burden_rows.append(curr_gene_level_info)
        genes.append(gene)
    
    logging.info("3. Create the final output matrix by concatenating each row of gene information (gene_burden_mat): n_genes by n_samples.")
    gene_burden_mat = pd.DataFrame(gene_burden_rows, index=genes)
    
    if binary_output:
        logging.info("4. Binarizing the matrix (anything !=0. gets set to 1)")
        gene_burden_mat = gene_burden_mat.applymap(binarize_burden_mat)
        assert not (gene_burden_mat > 1).any().any(), "At least one value in the supposedly binarized gene_burden_mat is greater than 1"

    return gene_burden_mat