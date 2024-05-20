# Script to prepare each data modality to be directly loaded into P-NET
# Author: Gwen Miller <gwen_miller@g.harvard.edu>
# TODO: Goal is to edit such that easy to loop over as many different genetic data modalities as we need (e.g. if we have 16 different germline combos, don't want to have to write everything out manually...)
import os

# Gwen's scripts
import data_manipulation
import prostate_data_loaders
import report_and_eval

import wandb
import argparse

import logging

logging.basicConfig(
    filename="run_pnet.log",
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script")
    parser.add_argument("--wandb_group", default="", help="Wandb group name")
    parser.add_argument("--use_only_paired", action="store_true", help="Use only paired")
    parser.add_argument("--convert_ids_to", default="somatic", help="Convert IDs to")
    parser.add_argument("--zero_impute_germline", action="store_true", help="Zero impute germline")
    parser.add_argument("--zero_impute_somatic", action="store_true", help="Zero impute somatic")
    parser.add_argument(
        "--somatic_datadir",
        default="../../pnet_germline/data/pnet_database/prostate/processed",
        help="Somatic data directory",
    )
    parser.add_argument("--germline_datadir", default="../../pnet_germline/data/", help="Germline data directory")
    parser.add_argument(
        "--save_dir", default="../../pnet_germline/processed/", help="Directory storing model-ready input"
    )

    return parser.parse_args()


def main():
    """
    Here, we prepare our inputs so they can be easily loaded into P-NET (and other models) without requiring any additional work beyond creating train/test/val splits.
    Specifically, we take care of issues including:
    1. Harmonizing the IDs
    2. Performing imputation as necessary (to keep non-overlapping genes)

    Load each of your data modalities of interest.
    Format should be samples x genes. Set the sample IDs as the index.

    Data modalities:
    1. somatic amp
    1. somatic del
    1. somatic mut
    1. germline mut (subset to a small number of genes).

    Our somatic data has information for many more genes compared to the germline data. We will need to either:
    1. impute zeros for the excluded germline genes, or
    2. subset the somatic datasets down to the ones that overlap with the germline data.


    We will be subsetting to the ~950 samples that we have matched somatic and germline data for.
    """

    logging.info("Parsing command-line arguments")
    args = parse_arguments()

    WANDB_GROUP = args.wandb_group
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project="prostate_met_status",
        group=WANDB_GROUP,
    )
    wandb_run_id = wandb.run.id

    # Access the values
    USE_ONLY_PAIRED = (
        args.use_only_paired
    )  # TODO: note that only have 943 somatic with IDs mapping to the metadata as done here... but should be able to get all 1011 like in PNET paper
    CONVERT_IDS_TO = args.convert_ids_to
    ZERO_IMPUTE_GERMLINE = args.zero_impute_germline
    ZERO_IMPUTE_SOMATIC = args.zero_impute_somatic
    SOMATIC_DATADIR = args.somatic_datadir
    GERMLINE_DATADIR = args.germline_datadir
    SAVE_DIR = args.save_dir

    logging.info("Constructing the save directory using the input parameters")
    if WANDB_GROUP != "":
        SAVE_DIR = os.path.join(
            SAVE_DIR,
            f"wandb-group-{WANDB_GROUP}/converted-IDs-to-{CONVERT_IDS_TO}_imputed-germline_{ZERO_IMPUTE_GERMLINE}_imputed-somatic_{ZERO_IMPUTE_SOMATIC}_paired-samples-{USE_ONLY_PAIRED}/wandb-run-id-{wandb_run_id}",
        )
        # SAVE_DIR = os.path.join(SAVE_DIR, f"wandb-group-{WANDB_GROUP}/converted-IDs-to-{CONVERT_IDS_TO}_imputed-germline_{ZERO_IMPUTE_GERMLINE}_imputed-somatic_{ZERO_IMPUTE_SOMATIC}/wandb-run-id-{wandb_run_id}")
    else:
        SAVE_DIR = os.path.join(
            SAVE_DIR,
            f"converted-IDs-to-{CONVERT_IDS_TO}_imputed-germline-{ZERO_IMPUTE_GERMLINE}_imputed-somatic-{ZERO_IMPUTE_SOMATIC}_paired-samples-{USE_ONLY_PAIRED}/wandb-run-id-{wandb_run_id}",
        )
    report_and_eval.make_dir_if_needed(SAVE_DIR)

    logging.debug("Defining file paths based on directories")
    somatic_mut_f = os.path.join(SOMATIC_DATADIR, "P1000_final_analysis_set_cross_important_only.csv")
    somatic_cnv_f = os.path.join(SOMATIC_DATADIR, "P1000_data_CNA_paper.csv")

    # TODO: update paths
    germline_rare_lof_f = os.path.join(
        GERMLINE_DATADIR,
        "prostate/prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_rare_high-impact.txt",
    )
    germline_rare_missense_f = os.path.join(
        GERMLINE_DATADIR,
        "prostate/prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_rare_moderate-impact.txt",
    )
    germline_common_lof_f = os.path.join(
        GERMLINE_DATADIR,
        "prostate/prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_common_high-impact.txt",
    )
    germline_common_missense_f = os.path.join(
        GERMLINE_DATADIR,
        "prostate/prostate_germline_vcf_subset_to_germline_tier_12_and_somatic_passed-universal-filters_common_moderate-impact.txt",
    )

    id_map_f = os.path.join(GERMLINE_DATADIR, "prostate/germline_somatic_id_map_outer_join.csv")
    sample_metadata_f = os.path.join(
        GERMLINE_DATADIR, "prostate/pathogenic_variants_with_clinical_annotation_1341_aug2021_correlation.csv"
    )
    logging.debug("Defining path(s) for the confounder/clincial/additional data")
    additional_f = sample_metadata_f

    logging.info("Adding hyperparameters and run metadata to Weights and Biases")
    hparams = {
        "zero_impute_germline": ZERO_IMPUTE_GERMLINE,
        "zero_impute_somatic": ZERO_IMPUTE_SOMATIC,
        "restricted_to_pairs": USE_ONLY_PAIRED,
        "somatic_mut_f": somatic_mut_f,
        "somatic_cnv_f": somatic_cnv_f,
        "germline_rare_lof_f": germline_rare_lof_f,
        "germline_rare_missense_f": germline_rare_missense_f,
        "germline_common_lof_f": germline_common_lof_f,
        "germline_common_missense_f": germline_common_missense_f,
        "id_map_f": id_map_f,
        "sample_metadata_f": sample_metadata_f,
        "additional_f": sample_metadata_f,
        "save_dir": SAVE_DIR,
    }
    wandb.config.update(hparams)

    logging.info("Loading data")
    somatic_mut = prostate_data_loaders.get_somatic_mutation(somatic_mut_f)
    somatic_amp, somatic_del = prostate_data_loaders.get_somatic_amp_and_del(somatic_cnv_f)
    germline_rare_lof = prostate_data_loaders.get_germline_mutation(germline_rare_lof_f)
    germline_rare_missense = prostate_data_loaders.get_germline_mutation(germline_rare_missense_f)
    germline_common_lof = prostate_data_loaders.get_germline_mutation(germline_common_lof_f)
    germline_common_missense = prostate_data_loaders.get_germline_mutation(germline_common_missense_f)

    # response / target variable
    y = prostate_data_loaders.get_target(
        id_map_f, sample_metadata_f, id_to_use="Tumor_Sample_Barcode", target_col="is_met"
    )
    # confounders
    additional = prostate_data_loaders.get_additional_data(
        additional_f,
        id_map_f,
        cols_to_include=["PCA1", "PCA2", "PCA3", "PCA4", "PCA5", "PCA6", "PCA7", "PCA8", "PCA9", "PCA10"],
    )

    if (
        USE_ONLY_PAIRED or CONVERT_IDS_TO
    ):  # TODO: if only USE_ONLY_PAIRED, then we need to specify how to convert... or we just make the CONVERT_IDS_TO a required parameter TODO: Need to test the function going both ways, to somatic and to germline
        logging.info("Harmonizing IDs (switching to {} IDs)".format(CONVERT_IDS_TO))
        prostate_data_loaders.harmonize_prostate_ids(
            datasets_w_germline_ids=[
                germline_rare_lof,
                germline_rare_missense,
                germline_common_lof,
                germline_common_missense,
            ],
            datasets_w_somatic_ids=[somatic_mut, somatic_amp, somatic_del] + [additional, y],
            convert_ids_to=CONVERT_IDS_TO,
        )  # want to run stuff if change_to_somatic_ids or paired
    if (
        USE_ONLY_PAIRED
    ):  # want to run stuff if paired; note that this will only work correctly if the IDs are correctly harmonized
        logging.info("Restricting to overlapping samples (the rows/indices)")
        (
            somatic_mut,
            somatic_amp,
            somatic_del,
            germline_rare_lof,
            germline_rare_missense,
            germline_common_lof,
            germline_common_missense,
            additional,
            y,
        ) = data_manipulation.restrict_to_overlapping_indices(
            somatic_mut,
            somatic_amp,
            somatic_del,
            germline_rare_lof,
            germline_rare_missense,
            germline_common_lof,
            germline_common_missense,
            additional,
            y,
        )

    # zero impute dataset columns (genes) as necessary (maybe have a reference dataset vs all others? Unsure of best way to parameterize this function)
    logging.info(
        "Zero-imputing columns (genes) as defined by user (impute germline: {}, impute somatic: {})".format(
            ZERO_IMPUTE_GERMLINE, ZERO_IMPUTE_SOMATIC
        )
    )
    germline_rare_lof, germline_rare_missense, germline_common_lof, germline_common_missense = (
        prostate_data_loaders.zero_impute_germline_datasets(
            germline_datasets=[
                germline_rare_lof,
                germline_rare_missense,
                germline_common_lof,
                germline_common_missense,
            ],
            somatic_datasets=[somatic_mut, somatic_amp, somatic_del],
            zero_impute_germline=ZERO_IMPUTE_GERMLINE,
        )
    )

    somatic_mut, somatic_amp, somatic_del = prostate_data_loaders.zero_impute_somatic_datasets(
        germline_datasets=[germline_rare_lof, germline_rare_missense, germline_common_lof, germline_common_missense],
        somatic_datasets=[somatic_mut, somatic_amp, somatic_del],
        zero_impute_somatic=ZERO_IMPUTE_SOMATIC,
    )

    logging.info(
        "Now that we have processed all our datasets, we restrict to just the overlapping genes (the columns) (and further filter to those that fit some TCGA criteria)"
    )
    (
        somatic_mut,
        somatic_amp,
        somatic_del,
        germline_rare_lof,
        germline_rare_missense,
        germline_common_lof,
        germline_common_missense,
    ) = prostate_data_loaders.restrict_to_genes_in_common(
        somatic_mut,
        somatic_amp,
        somatic_del,
        germline_rare_lof,
        germline_rare_missense,
        germline_common_lof,
        germline_common_missense,
    )

    logging.info("Printing some basic info for each of our datasets")
    df_dict = dict(
        zip(
            [
                "somatic_mut",
                "somatic_amp",
                "somatic_del",
                "germline_rare_high-impact",
                "germline_rare_moderate-impact",
                "germline_common_high-impact",
                "germline_common_moderate-impact",
                "additional",
                "y",
            ],
            [
                somatic_mut,
                somatic_amp,
                somatic_del,
                germline_rare_lof,
                germline_rare_missense,
                germline_common_lof,
                germline_common_missense,
                additional,
                y,
            ],
        )
    )

    report_and_eval.report_df_info_with_names(df_dict, n=5)

    if SAVE_DIR != "":
        logging.info("Saving each df to {}".format(SAVE_DIR))
        for name, df in df_dict.items():
            save_name = os.path.join(SAVE_DIR, name + ".csv")
            df.to_csv(save_name, index=True)
            wandb.config.update({name + "_output_f": save_name})

    logging.info("Ending wandb run")
    wandb.finish()
    return wandb_run_id


if __name__ == "__main__":
    main()
