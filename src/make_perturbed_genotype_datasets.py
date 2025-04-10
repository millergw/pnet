"""
Script: make_perturbed_genotype_datasets.py

What this script will do:

1. Load the original binary genotype matrix (`somatic_mut.csv`).

2. For each combination of parameters:
   - Randomly choose a set of samples to affect.
   - Randomly choose a set of features (genes) to perturb.
   - For each selected gene, randomly select a fraction of its rows (within the selected samples) to flip to `1`.

3. Save each perturbed dataset with an informative filename that includes:
   - Fraction of samples affected (10, 50, 100%)
   - Number of features (1, 10, 100, or all)
   - Perturbation strength (1%, 10%, or 100% of selected sample rows per feature set to 1)
"""

import os
import logging
import numpy as np
import pandas as pd
import wandb
from itertools import product

# === LOGGING SETUP === #
logging.basicConfig(
    filename="make_perturbed_genotype_datasets.log",
    encoding="utf-8",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# === WANDB INIT === #
wandb.login()
wandb.init(
    project="pnet_data_simulation",
    name="generate_perturbed_genotypes",
    notes="Generate perturbed genotype datasets for testing model performance",
)


# === HELPER FUNCTION === #
def create_target_dataframe(df_index, sample_subset):
    """
    Create a target DataFrame with an 'is_met' column indicating whether
    each sample is in the sample_subset.

    Args:
        df_index (pd.Index): The index of the original DataFrame (sample names).
        sample_subset (list): A list of samples (or indices) to mark as 'met'.

    Returns:
        pd.DataFrame: A DataFrame with the same index as the input and an 'is_met' column.
    """
    target = pd.DataFrame({"Tumor_Sample_Barcode": df_index}).set_index("Tumor_Sample_Barcode")
    target["is_met"] = target.index.map(lambda x: 1 if x in sample_subset else 0)
    return target


# === CONFIGURATION === #
DATA_DIR = "../../pnet_germline/processed/wandb-group-data_prep_germline_tier12_and_somatic/converted-IDs-to-somatic_imputed-germline_True_imputed-somatic_False_paired-samples-True/wandb-run-id-q151d0zw"
SAVE_DIR = "../../pnet_germline/processed/perturbed_genotype_datasets/p1000_somatic_mut"
DATA_FILENAME = "somatic_mut.csv"
summary_csv_path = os.path.join(SAVE_DIR, "perturbation_summary.csv")
SEED = 42
np.random.seed(SEED)

logging.info("Setting random seed to {}".format(SEED))

# === LOAD DATA === #
logger.info(f"Loading original dataset from: {DATA_DIR}")
data_f = os.path.join(DATA_DIR, DATA_FILENAME)
df = pd.read_csv(data_f, index_col=0)
samples = df.index.tolist()
genes = df.columns.tolist()
n_samples = len(samples)
n_genes = len(genes)
logger.info(f"Loaded dataset with shape: {df.shape}")

# === PARAMETERS === #
fractions_samples = [0.3, 0.5]  # % of samples to affect (this is how many samples we put into the perturbed class)
n_features_list = [1, 10, 100, "all"]  # number of features to affect
perturb_strengths = [
    0.1,
    0.5,
    1.0,
]  # % of selected column to set to 1 (this is the portion of our perturbed class that we actually alter)

# === WANDB CONFIG UPDATE === #
hparams = {
    "data_dir": DATA_DIR,
    "save_dir": SAVE_DIR,
    "data_filename": DATA_FILENAME,
    "seed": SEED,
    "fractions_samples": fractions_samples,
    "n_features_list": n_features_list,
    "perturb_strengths": perturb_strengths,
    "n_samples": n_samples,
    "n_genes": n_genes,
    "summary_csv_path": summary_csv_path,
}

wandb.config.update(hparams)

# === OUTPUT DIR === #
try:
    os.makedirs(SAVE_DIR, exist_ok=True)
    logger.info(f"Output directory created or already exists: {SAVE_DIR}")
except Exception as e:
    logger.error(f"Failed to create output directory {SAVE_DIR}: {e}")
    raise

# === METADATA TRACKING === #
summary_records = []

# === MAIN LOOP === #
for frac_sample, n_features, strength in product(fractions_samples, n_features_list, perturb_strengths):
    df_copy = df.copy()
    sample_subset = np.random.choice(samples, size=int(n_samples * frac_sample), replace=False)

    # Use the new function to create the target DataFrame
    target = create_target_dataframe(df_copy.index, sample_subset)

    selected_genes = (
        genes if n_features == "all" else np.random.choice(genes, size=min(n_features, n_genes), replace=False)
    )
    # The concept is that: "For this gene and for this group of affected samples, inject a synthetic signal by flipping a proportion of their values to 1"
    for gene in selected_genes:
        rows_to_perturb = np.random.choice(
            sample_subset,
            size=max(1, int(len(sample_subset) * strength)),
            replace=False,  # We use max(1, ...) to ensure at least one row gets perturbed, even if strength is very small
        )
        df_copy.loc[rows_to_perturb, gene] = 1

    suffix = f"samplePrcnt{int(frac_sample*100)}_features{n_features}_strengthPrcnt{int(strength*100)}"
    out_path = os.path.join(SAVE_DIR, f"somatic_mut_{suffix}.csv")
    df_copy.to_csv(out_path)

    target_out_path = os.path.join(SAVE_DIR, f"y_{suffix}.csv")
    target.to_csv(target_out_path, index=True)

    logger.info(f"Saved perturbed dataset: {out_path}")

    summary_records.append(
        {
            "out_data_file": out_path,
            "out_target_file": target_out_path,
            "fraction_samples": frac_sample,
            "n_features": n_features,
            "perturb_strength": strength,
            "n_samples_affected": len(sample_subset),
            "n_genes_perturbed": len(selected_genes),
        }
    )

# === SAVE SUMMARY TO WANDB === #
summary_df = pd.DataFrame(summary_records)
summary_df.to_csv(summary_csv_path, index=False)
logger.info(f"Saved perturbation summary to {summary_csv_path}")
logger.info("Finished generating perturbed datasets.")
wandb.finish()
