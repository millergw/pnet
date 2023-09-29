"""
Utility functions related to the processing and analysis of germline data.
Specifically, working with germline VCFs for the prostate cancer dataset.

Author: Gwen Miller <gwen_miller@g.harvard.edu>
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import logging
import matplotlib
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# from filter_to_pathogenic_variants import *
import filter_to_pathogenic_variants as patho

logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
# logger.setLevel(logging.INFO)


############################## 
# General data loading and munging
##############################

def remove_whitespace_from_df(df):
    """Remove all leading and trailing whitespace from DF column names and every cell"""
    logging.debug("Remove leading and trailing whitespace from every cell")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    logging.debug("Remove leading and trailing whitespace from column names")
    df.columns = df.columns.str.strip()
    return df


def find_mapping(list_a, list_b, reverse_dict=False):
    """
    To find the mapping between two lists where each element in list A is a strict substring of one of the elements in list B, you can use this function.
    # Example usage
    list_a = ['apple', 'banana', 'cat']
    list_b = ['apple pie', 'banana bread', 'black cat']

    result = find_mapping(list_a, list_b)
    print(result)
    
                 Mapped Value
    apple        apple pie
    banana       banana bread
    cat          black cat

    """
    mapping = {}
    for a in list_a:
        for b in list_b:
            if a in b:
                mapping[a] = b
                break
    logging.debug(f"Found matches for a total of {len(mapping)} out of {len(list_a)} items.")
    if reverse_dict is True: # TODO: need to check what happens if the values are not unique
        logging.info("Reversing the dict so the superstrings are the keys and the substrings are the values")
        logging.info(f"len(set(mapping.values())) == len(mapping.values()): {len(set(mapping.values())) == len(mapping.values())}")
        reversed_dict = {value: key for key, value in mapping.items()}
        mapping = reversed_dict
    return mapping


def make_path_if_needed(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        logging.debug(f"We're creating the non-existing directories in {directory}")
        os.makedirs(directory)
    return
                      

def filename(f):
    return os.path.splitext(os.path.basename(f))[0]


def get_files_with_suffix_from_dir(dir_path, suffix):
    all_files = os.listdir(dir_path)    
    suffix_files = list(filter(lambda f: f.endswith(suffix), all_files))
    return [os.path.join(dir_path, p) for p in suffix_files]


def savefig(save_path, png=True, svg=True):
    make_path_if_needed(save_path)
    logging.info(f"saving plot to {save_path}")
    if png:
        plt.savefig(save_path, bbox_inches='tight')
    if svg:
        plt.savefig(f"{save_path}.svg", format="svg", bbox_inches='tight')


def relocate(df, cols): 
    """cols is a list of column names you want to place in the front"""
    new_var_order = cols + df.columns.drop(cols).tolist() 
    df = df[new_var_order] 
    return(df)


def encode_cat_vars_in_df(df, numeric_col_names, categorical_col_names):
    """
    Example:
    ---
    # Assuming you have a data matrix 'data' with multiple numeric and categorical columns
    data = np.array([[1, 2, 'red', 'small'],
                     [4, 5, 'blue', 'medium'],
                     [7, 8, 'green', 'large'],
                     [10, 11, 'red', 'small']])

    # Specify the column names for numeric and categorical columns
    numeric_cols = ['num1', 'num2']  # Column names of numeric columns
    categorical_cols = ['color', 'size']  # Column names of categorical columns

    # Convert the data matrix to a pandas DataFrame
    df = pd.DataFrame(data, columns=numeric_cols + categorical_cols)

    encoded_df = encode_cat_vars_in_df(df, numeric_cols, categorical_cols) # will have columns like "color_red" and "size_medium"
    ---
    """
    logging.debug("Create a ColumnTransformer with OneHotEncoder for categorical columns")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_col_names),  # Pass through numeric columns (aka don't do anything)
            ('cat', OneHotEncoder(), categorical_col_names)  # OneHotEncoder for categorical columns
        ])

    logging.debug("Apply the preprocessing to the DataFrame and retrieve feature names")
    encoded_data = preprocessor.fit_transform(df)
    feature_names = list(numeric_col_names) + list(preprocessor.named_transformers_['cat'].get_feature_names_out())

    logging.debug("Create a new DataFrame with updated column names")
    encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

    logging.debug(f"Returning the one-hot encoded DataFrame with new columns: {list(preprocessor.named_transformers_['cat'].get_feature_names_out())}")
    return encoded_df



############################## 
# Loading prostate datasets
##############################

def read_gene_list_from_csv(list_f):
    return pd.read_csv(list_f, comment='#').columns.str.strip().tolist()


def load_germline_metadata(metadata_f = "../data/prostate/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv"):
    # the clinical metadata file can be indexed by ther germline ID ('sample_original')
    met_col = "Disease_status_saud_edited"
    id_col = "BamID_modified"
    terra_control_id = "sample_id"
    is_met_col = "is_met"
    logging.debug(f"metadata_f: {metadata_f}")
    metadata = pd.read_csv(metadata_f)
    # metadata = pd.read_csv(metadata_f, usecols = [id_col, met_col, terra_control_id])
    # create binary "is_met_col" column
    metadata[is_met_col] = metadata[met_col].map({'Metastatic': 1,  'Primary': 0})
    metadata = metadata.rename(columns={id_col: "germline_id", met_col: "disease_status"})
    logging.debug(f"Head of the metadata DF:")
    logging.debug(metadata.head())
    return metadata


def load_sample_metadata_with_all_germline_ids(sample_metadata_f="../data/prostate/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv", 
                                               germline_somatic_id_map_f='../data/prostate/germline_somatic_id_map_outer_join.csv', 
                                               sample_metadata_germline_id_col="germline_id", 
                                               germline_id_map_col="sample_metadata_germline_id"):
    """
    Args:
    - sample_metadata_f: filepath to the sample metadata file
    - germline_somatic_id_map_f: filepath to the germline ID mapping DF
    - sample_metadata_germline_id_col: name of the germline ID column in the sample metadata DF
    - germline_id_map_col: name of the column in the germline ID mapping DF that corresponds to (aka is the same as) the column data from the sample metadata DF
    """
    # Load in the DFs
    germline_somatic_id_map = pd.read_csv(germline_somatic_id_map_f)
    logging.debug(f"sample_metadata_f: {sample_metadata_f}")
    sample_metadata = load_germline_metadata(sample_metadata_f)

    # add more metadata columns to the sample_metadata DF
    sample_metadata = pd.merge(germline_somatic_id_map, sample_metadata, left_on=germline_id_map_col, right_on=sample_metadata_germline_id_col)
    # drop the now redundant column
    sample_metadata.drop(sample_metadata_germline_id_col, axis=1, inplace=True)
    sample_metadata.set_index(germline_id_map_col, inplace=True, drop=True)
    return sample_metadata


def load_somatic_mutation_data(somatic_mut_f="../data/pnet_database/prostate/processed/"):
    pass


def load_germline_mutation_data(germline_vars_f="../data/prostate/prostate_germline_vcf_subset_to_germline_tier_1and2_pathogenic_vars_only.txt"):
#     logging.info("# Loading the germline VCF")
#     df = pd.read_csv(germline_vars_f, low_memory=False, sep="\t")
#     logging.debug("## Setting the variant ID as the DF index")
#     df = df.set_index("Uploaded_variation")
    
#     logging.info("# Make the binary variant-level genotypes matrix")
#     binary_genotypes = make_binary_genotype_mat_from_VCF(df)
#     logging.info(binary_genotypes.shape)


#     logging.info("# Make the (binary) gene-level genotypes matrix")
#     gene_burden_mat =  convert_binary_var_mat_to_gene_level_mat(binary_genotypes, 
#                                                                 variant_metadata, 
#                                                                 binary_output = BINARY_GENE_BURDEN_MAT)

    pass


def keep_paired_samples(somatic_df, germline_df, germline_somatic_id_map_f="../data/prostate/germline_somatic_id_map_outer_join.csv"):
    logging.info("only keeping rows in metadata table that (1) have values for both 'vcf_germline_id' and 'Tumor_Sample_Barcode' and (2) exist in the DFs passed into this function")
    paired_sample_df = pd.read_csv(germline_somatic_id_map_f).dropna(subset=['vcf_germline_id', 'Tumor_Sample_Barcode'])
    logging.debug(f"shape after restricting by (1): {paired_sample_df.shape}")
    paired_sample_df = paired_sample_df[paired_sample_df.Tumor_Sample_Barcode.isin(somatic_df.index.tolist())]
    logging.debug(f"shape after restricting to samples that are in the somatic DF: {paired_sample_df.shape}")
    paired_sample_df = paired_sample_df[paired_sample_df.vcf_germline_id.isin(germline_df.index.tolist())]
    logging.debug(f"shape after restricting to samples that are in the germline DF: {paired_sample_df.shape}")

    logging.info("filtering the germline and somatic DFs to those we have paired germline-somatic data for")
    germline_samples = paired_sample_df.vcf_germline_id.tolist()
    somatic_samples = paired_sample_df.Tumor_Sample_Barcode.tolist()
    filt_somatic_df = somatic_df.loc[somatic_samples,:]
    filt_germline_df = germline_df.loc[germline_samples,:]
    logging.info(f"filtered somatic DF shape: {filt_somatic_df.shape}")
    logging.info(f"filtered germline DF shape: {filt_germline_df.shape}")
    
    logging.info("returning the filt_somatic_df, filt_germline_df, paired_sample_df (aka the paired samples' metadata)")
    return filt_somatic_df, filt_germline_df, paired_sample_df


def load_metadata_on_specified_samples(sample_list=None, col_name=None, germline_somatic_id_map_f="../data/prostate/germline_somatic_id_map_outer_join.csv"):
    if sample_list is None:
        pass
    else:
        pass
    pass


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


def filter_df_by_list_of_values_in_column(df, colname, list_of_vals_to_keep):
    filtered_df = df[df[colname].isin(list_of_vals_to_keep)]
    return filtered_df


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