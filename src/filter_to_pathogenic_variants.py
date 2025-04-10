"""
Additional filtering: germline-specific factors (e.g. MAF <0.01)

Here we use the function that the germline folks use to filter down to a reasonable subset of variants (aka those likely to be high impact and biologically relevant, not just popping out due to population structure).

Modified from this notebook: https://app.terra.bio/#workspaces/vanallen-firecloud-nih/Germline_pipeline_components/analysis/launch/Jan2023_PathogenicVariantFilteringNotebook.ipynb
"""

import pandas as pd
import logging
logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
logger.setLevel(logging.INFO)


######
# Defining inclusion-exclusion criteria for variants with conflicting interpretation
######
def conflicting_filter_criteria(row):
    """
    Defining inclusion-exclusion criteria for variants with conflicting interpretation
    """
    patho_consolidated = row["Pathogenic_consolidated"]
    benign_consolidated = row["Benign_consolidated"]
    uncertain_consolidated = row["Uncertain_significance"]
    if patho_consolidated < 2:
        return "exclude"
    elif patho_consolidated < benign_consolidated:
        return "exclude"
    elif patho_consolidated < uncertain_consolidated:
        return "exclude"
    else:
        return "include"

    
######
# Defining helper functions
######
# Subsets the vep df to variants in the predefined gene list
def subset_to_gene_list(vep_df,gene_list):
    df_subset = vep_df[vep_df["SYMBOL"].isin(gene_list)].copy()
    return df_subset

# Consolidate the consequence of each variant to a single consequence 
def add_consolidated_consequence(vep_df):
    vep_copy = vep_df.copy()
    vep_copy['Consequence_consolidated'] = vep_copy['Consequence'].str.split(',').str[0]
    return vep_copy

# Subset to varaints that has severe consequences as defined by containing one of the keywords
def subset_to_severe_consequence(vep_df,severe_consequences=None):
    if severe_consequences is None:
        severe_consequences = "splice_donor_variant|frameshift_variant|stop_gained|splice_acceptor_variant|transcript_ablation|stop_lost|start_lost|transcript_amplification"
    vep_copy = add_consolidated_consequence(vep_df)
    vep_copy = vep_copy[vep_copy['Consequence_consolidated'].str.contains(severe_consequences)]
    return vep_copy

# Remove variants with clinvar classification of one of the benigns
def subset_to_non_benign(vep_df,clinsig_col=None,clinvar_benign=None):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if clinvar_benign is None:
        clinvar_benign = "Benign|Likely_benign|Benign/Likely_benign"
    vep_df_nob = vep_df[~(vep_df[clinsig_col].str.contains(clinvar_benign))].copy()
    return vep_df_nob

# Remove variants with MAF above the predefined frequency 
def subset_to_low_frequency(vep_df,freq_col="gnomAD_AF",freq=0.01, verbose=False):
    # Convert missing to zero
    vep_df_copy = vep_df.copy()
    vep_df_copy.loc[vep_df['gnomAD_AF']=='-','gnomAD_AF']=0
    vep_df_copy.loc[vep_df['MAX_AF']=='-','MAX_AF']=0
    vep_df_copy['gnomAD_AF']=vep_df_copy['gnomAD_AF'].astype(float)
    
    vep_df_copy_lf = vep_df_copy[vep_df_copy[freq_col] < freq].copy()
    
    if verbose:
        removed_df = vep_df_copy[vep_df_copy[freq_col] >= freq]
        logging.info(f"we removed {len(removed_df)} variants")
        logging.info(removed_df)
    return vep_df_copy_lf

# Subset to varaints defined as pathogenic in clinvar
def subset_to_clinvar_pathogenic(vep_df,clinsig_col=None,clinvar_pathogenic=None,conflicting_col=None):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if clinvar_pathogenic is None:
        clinvar_pathogenic = "Pathogenic|Likely_pathogenic|_risk_factor|risk_factor|Pathogenic/Likely_pathogenic"
    vep_df_patho = vep_df[vep_df[clinsig_col].str.contains(clinvar_pathogenic)]
    vep_df_patho_no_conflict = subset_to_clinvar_conflicting(vep_df_patho, clinsig_col, conflicting_col, invert=True)
    
    return vep_df_patho_no_conflict

# Subset to variants with conflicitng clinvar classifications
def subset_to_clinvar_conflicting(vep_df,clinsig_col=None,conflicting_col=None,invert=False):
    if clinsig_col is None:
        clinsig_col = "ClinVar_updated_2022Aug_CLNSIG"
    if conflicting_col is None:
        conflicting_col = "Conflicting_interpretations_of_pathogenicity"
    if not invert:
        vep_conflicting = vep_df[vep_df[clinsig_col].str.contains(conflicting_col)]
    else:
        vep_conflicting = vep_df[~vep_df[clinsig_col].str.contains(conflicting_col)]
    return vep_conflicting
    
# For variants with conflicting clinvar evidences, create a table where each column 
# has the number of submissions relating to that classification
def make_confliciting_evidence_table(vep_df,id_col="Uploaded_variation",clinsig_col=None,conflicting_col=None,clin_conflict=None,conseq_col=None):
    if clinsig_col is None:
        clinsig_col="ClinVar_updated_2022Aug_CLNSIG"
    if clin_conflict is None:
        clin_conflict = "ClinVar_updated_2022Aug_CLNSIGCONF"
    if conflicting_col is None:
        conflicting_col = "Conflicting_interpretations_of_pathogenicity"
    if conseq_col is None:
        conseq_col = "Consequence_consolidated"
    
    vep_df_no_dup = vep_df.drop_duplicates(subset=[id_col])
    
    ## get conflicting ones
    conflicting = subset_to_clinvar_conflicting(vep_df_no_dup,clinsig_col=clinsig_col,conflicting_col=conflicting_col)
    conflicting = conflicting[[id_col, clin_conflict]]
    conflicting = conflicting.set_index(id_col)
    
    ## Parse the lines of evidence for pathogenicity
    ## This line may change between VEP version depending on delimiter style
    conflicting_expanded = conflicting[clin_conflict].str.split('|', expand = True)

    ## get long format
    conflicting_expanded = conflicting_expanded.stack().reset_index()
    conflicting_expanded = conflicting_expanded.drop(['level_1'], axis = 1)
    conflicting_expanded = conflicting_expanded.rename(columns = {0 : 'clinvar_annotation'})
    
    ## correct classifications names that start with a "_"
    conflicting_expanded["clinvar_annotation"] = conflicting_expanded["clinvar_annotation"].apply(lambda x: x[1:] if x[0]=="_" else x)

    conflicting_expanded['count'] = conflicting_expanded['clinvar_annotation'].str.split('(').str[1]
    conflicting_expanded['count'] = conflicting_expanded['count'].str.split(')').str[0]
    conflicting_expanded['clinvar_annotation'] = conflicting_expanded['clinvar_annotation'].str.split('(').str[0]

    ## pivot table to get clinvar terms as col names
    conflicting_expanded_transformed = conflicting_expanded.pivot(columns='clinvar_annotation', values='count')
    conflicting_expanded_transformed[id_col] = conflicting_expanded[id_col]
    conflicting_expanded_transformed = conflicting_expanded_transformed.set_index(id_col)
    conflicting_expanded_transformed = conflicting_expanded_transformed.rename_axis(None, axis=1).reset_index()
    
    ## combine rows and their values for each position
    clinvar_classifications = ["Benign","Likely_benign","Likely_pathogenic","Pathogenic","Uncertain_significance"]
    clinvar_count_operation = {var_type:"sum" for var_type in clinvar_classifications}
    
    for var_type in clinvar_classifications:
        if var_type in conflicting_expanded_transformed.columns:
            conflicting_expanded_transformed[var_type] = conflicting_expanded_transformed[var_type].fillna(0).astype(int)
        else:
            conflicting_expanded_transformed[var_type] = 0
            
    conflicting_expanded_transformed = conflicting_expanded_transformed.groupby([id_col],as_index=False).agg(clinvar_count_operation)
    
    ## get consolidated cols
    conflicting_expanded_transformed['Benign_consolidated'] = conflicting_expanded_transformed['Benign'] + conflicting_expanded_transformed['Likely_benign']
    conflicting_expanded_transformed['Pathogenic_consolidated'] = conflicting_expanded_transformed['Pathogenic'] + conflicting_expanded_transformed['Likely_pathogenic']
    
    ## Add variant consequences
    vep_df_no_dup = add_consolidated_consequence(vep_df_no_dup)
    conseq = vep_df_no_dup[[id_col, conseq_col]]
    conflicting_expanded_transformed_conseq = pd.merge(conflicting_expanded_transformed, conseq, on = id_col, how = 'inner')    
    return conflicting_expanded_transformed_conseq

# A function that decides on whether to include the conflicitng variant
# This function is very project dependent. Make your own variant filter criteria as it fits for your project
# The requirement for filter_criteria: takes in a dataframe row and returns "exclude" or "include"
def variant_filter_criteria(row):
    patho_consolidated = row["Pathogenic_consolidated"]
    benign_consolidated = row["Benign_consolidated"]
    uncertain_consolidated = row["Uncertain_significance"]
    
    if patho_consolidated == 0:
        return "exclude"
    elif patho_consolidated < benign_consolidated:
        return "exclude"
    elif patho_consolidated < uncertain_consolidated:
        return "exclude"
    else:
        return "include"

# Make a decision whether to include or exclude the conflicting variants based on the lines of evidences
# The filter_critera is a user-defined function that decides on whether the variant should be included or excluded
def make_exlcusion_decision(evidence_table,filter_criteria=variant_filter_criteria):
    evidence_table_copy = evidence_table.copy()
    evidence_table_copy['decision'] = evidence_table_copy.apply(filter_criteria,axis=1)
    return evidence_table_copy
                 
    
######
# Defining the variant filtering workflow
######

def variant_selection_workflow(
    vep_df,
    genes_to_subset,
    id_col="Uploaded_variation",
    filter_criteria=variant_filter_criteria,
    clinsig_col = 'ClinVar_updated_2021Jun_CLNSIG',
    clin_conflict = 'ClinVar_updated_2021Jun_CLNSIGCONF'
):
    # Subset to variants in predefined gene list
    vep_gene_subset = subset_to_gene_list(vep_df,gene_list=genes_to_subset)
    
    # Set 1 - Variants with severe consequence
    # subset to variants with severe consequence
    vep_truncating = subset_to_severe_consequence(vep_gene_subset)
    # Remove benign high-impact variants
    vep_truncating_nob = subset_to_non_benign(vep_truncating, clinsig_col)
    # Remove high-impact variants with high-frequency
    vep_truncating_nob_lof = subset_to_low_frequency(vep_truncating_nob)
    logging.debug(f"vep_truncating_nob_lof: {vep_truncating_nob_lof.shape}\n{vep_truncating_nob_lof.head()}")
    
    # Set 2
    # Subset to variants with pathogenic clinvar annotation
    vep_patho = subset_to_clinvar_pathogenic(vep_gene_subset, clinsig_col=clinsig_col, conflicting_col=clin_conflict)
    logging.debug(f"vep_patho: {vep_patho.shape}\n{vep_patho.head()}")
    
    # Set 3
    # Subset to variants with conflicting clinvar annotation
    vep_conflicting = subset_to_clinvar_conflicting(vep_gene_subset, clinsig_col=clinsig_col, conflicting_col=clin_conflict)
    logging.debug(f"vep_conflicting: {vep_conflicting}")
    if len(vep_conflicting) >0:
        # Create a table representing lines of evidence
        vep_conflicting_table = make_confliciting_evidence_table(vep_conflicting, id_col="Uploaded_variation", clinsig_col=clinsig_col, clin_conflict=clin_conflict)
        # Makes inclusion-exclusion decision based on lines of evidence
        vep_conflicting_table = make_exlcusion_decision(vep_conflicting_table,filter_criteria=filter_criteria)

        # Keep only conflicting variants meeting inclusion criteria
        vep_conflicting_table_include = vep_conflicting_table[vep_conflicting_table["decision"]=="include"]
        vep_conflicting_table_include_merged = vep_conflicting_table_include[[id_col]].merge(vep_gene_subset,on=id_col)

        # Combine all variants subsetted so far and remove duplicate entries
        vep_all_patho_variants = pd.concat([
            vep_truncating_nob_lof,
            vep_patho,
            vep_conflicting_table_include_merged
        ],ignore_index=True)
    
    else:
        # Combine all variants subsetted so far and remove duplicate entries
        vep_all_patho_variants = pd.concat([
            vep_truncating_nob_lof,
            vep_patho
        ],ignore_index=True)
    
    vep_all_patho_variants = vep_all_patho_variants.drop_duplicates(subset=[id_col])
    vep_all_patho_variants = add_consolidated_consequence(vep_all_patho_variants)
    
    return vep_all_patho_variants