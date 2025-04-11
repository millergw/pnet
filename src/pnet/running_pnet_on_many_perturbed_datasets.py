import os
import pandas as pd
import subprocess

# Path to the perturbation summary CSV
summary_csv_path = (
    "../../pnet_germline/processed/perturbed_genotype_datasets/p1000_somatic_mut/perturbation_summary.csv"
)

# Load the perturbation summary CSV
summary_df = pd.read_csv(summary_csv_path)

# Filter the rows based on the desired conditions
filtered_df = summary_df[
    (summary_df["fraction_samples"] == 0.5)
    & (summary_df["perturb_strength"] == 1.0)
    & (summary_df["n_features"].isin(["10", "100"]))
]

# Define the perturbed data directory
perturbed_data_dir = "../../pnet_germline/processed/perturbed_genotype_datasets/p1000_somatic_mut"

# Define other parameters for the PNET model
data_config_f = "data.yaml"
evaluation_set = "validation"
model_type = "pnet"
wandb_project = "prostate_met_status"
wandb_group = "pnet_perturbed_data"
seed = 42
cpus = 8
input_data_dir = "../../pnet_germline/processed/wandb-group-data_prep_germline_tier12_and_somatic/converted-IDs-to-somatic_imputed-germline_True_imputed-somatic_False_paired-samples-True/wandb-run-id-q151d0zw"

# Iterate over the filtered rows and run the PNET model
for _, row in filtered_df.iterrows():
    perturbed_somatic_mut = os.path.basename(row["out_data_file"])  # Get the filename of the perturbed somatic_mut
    perturbed_target = os.path.basename(row["out_target_file"])  # Get the filename of the perturbed target

    # Construct the command to run the PNET model
    command = [
        "python",
        "run_model_on_perturbed_data.py",
        "--data_config_f",
        data_config_f,
        "--datasets",
        "somatic_amp somatic_del somatic_mut",
        "--evaluation_set",
        evaluation_set,
        "--model_type",
        model_type,
        "--wandb_project",
        wandb_project,
        "--wandb_group",
        wandb_group,
        "--seed",
        str(seed),
        "--input_data_dir",
        input_data_dir,
        "--perturbed_data_dir",
        perturbed_data_dir,
        "--perturbed_somatic_mut",
        perturbed_somatic_mut,
        "--perturbed_target",
        perturbed_target,
        "--cpus",
        str(cpus),
    ]

    # Print the command for debugging
    print(f"Running command: {' '.join(command)}")

    # Run the command
    subprocess.run(command)
