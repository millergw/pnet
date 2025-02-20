import os
import subprocess
import wandb


def fetch_relevant_runs(project_name, sweep_id, model_type):
    """
    Fetch WandB run IDs from a specific sweep and filter by model type.

    Parameters:
        project_name (str): The name of the WandB project.
        sweep_id (str): The WandB sweep ID.
        model_type (str): The model type to filter for.

    Returns:
        list: A list of WandB run IDs that match the criteria.
    """
    api = wandb.Api()
    runs = api.runs(project_name, filters={"sweep": sweep_id, "state": "finished"})

    relevant_run_ids = []
    for run in runs:
        if run.config.get("model_type") == model_type:
            relevant_run_ids.append(run.id)

    return relevant_run_ids


def run_script_for_each_run(script_path, project_name, relevant_run_ids):
    """
    Execute the script for each relevant WandB run ID.

    Parameters:
        script_path (str): Path to the script to execute.
        project_name (str): The name of the WandB project.
        relevant_run_ids (list): List of WandB run IDs to process.
    """
    for run_id in relevant_run_ids:
        command = ["python", script_path, "--wandb_project", project_name, "--wandb_run_id", run_id]
        print(f"Executing: {' '.join(command)}")
        subprocess.run(command)


if __name__ == "__main__":
    # Parameters
    PROJECT_NAME = "prostate_met_status"
    SWEEP_ID = "cmlmrw2s"
    MODEL_TYPE = "pnet"
    SCRIPT_PATH = "./analyze_misclassifications.py"  # Path to your script that takes --wandb_run_id

    # Step 1: Fetch relevant run IDs
    print(f"Fetching runs for project '{PROJECT_NAME}', sweep '{SWEEP_ID}', model type '{MODEL_TYPE}'...")
    relevant_run_ids = fetch_relevant_runs(PROJECT_NAME, SWEEP_ID, MODEL_TYPE)
    print(f"Found {len(relevant_run_ids)} relevant runs.")

    # Step 2: Run the script for each relevant WandB run ID
    print(f"Executing script '{SCRIPT_PATH}' for each run...")
    run_script_for_each_run(SCRIPT_PATH, PROJECT_NAME, relevant_run_ids)
