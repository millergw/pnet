import subprocess
from collections import defaultdict

import wandb


def fetch_grouped_runs(project_name, sweep_id, model_type):
    """
    Fetch WandB run IDs from a specific sweep and group by shared train/eval indices.

    Parameters:
        project_name (str): The name of the WandB project.
        sweep_id (str): The WandB sweep ID.
        model_type (str): The model type to filter for.

    Returns:
        dict: Dictionary where keys are (train_set, eval_set) pairs and values are lists of run IDs.
    """
    api = wandb.Api()
    runs = api.runs(project_name, filters={"sweep": sweep_id, "state": "finished"})

    grouped_runs = defaultdict(list)

    for run in runs:
        if run.config.get("model_type") == model_type:
            train_set = run.config["train_set_indices_f"]
            eval_set = run.config["evaluation_set_indices_f"]
            grouped_runs[(train_set, eval_set)].append(run.id)

    return grouped_runs


def run_script_for_each_group(script_path, project_name, grouped_runs):
    """
    Execute the script for each group of runs that share the same train/eval indices.

    Parameters:
        script_path (str): Path to the script to execute.
        project_name (str): The name of the WandB project.
        grouped_runs (dict): Dictionary of grouped run IDs.
    """
    for (train_set, eval_set), run_ids in grouped_runs.items():
        run_ids_str = ",".join(run_ids)  # Convert list of IDs to comma-separated string
        command = [
            "python",
            script_path,
            "--wandb_project",
            project_name,
            "--wandb_run_ids",
            run_ids_str,  # Pass all grouped run IDs at once
            "--train_set",
            train_set,
            "--eval_set",
            eval_set,
        ]
        print(f"Executing: {' '.join(command)}")
        subprocess.run(command)


if __name__ == "__main__":
    # Parameters
    PROJECT_NAME = "prostate_met_status"
    SWEEP_ID = "cmlmrw2s"
    MODEL_TYPE = "pnet"
    SCRIPT_PATH = "./analyze_misclassifications.py"

    # Step 1: Fetch grouped runs
    print(f"Fetching grouped runs for project '{PROJECT_NAME}', sweep '{SWEEP_ID}', model type '{MODEL_TYPE}'...")
    grouped_runs = fetch_grouped_runs(PROJECT_NAME, SWEEP_ID, MODEL_TYPE)
    print(f"Found {len(grouped_runs)} unique training/evaluation index set combination(s).")

    # Step 2: Execute script once per group
    print(f"Executing script '{SCRIPT_PATH}' for each group...")
    run_script_for_each_group(SCRIPT_PATH, PROJECT_NAME, grouped_runs)
