"""
Functions for 
- reporting information during running of a program, eg details about DFs
- reporting evaluation information for models
Author: Gwen Miller <gwen_miller@g.harvard.edu>
"""

import numpy as np
import pandas as pd
import logging
import matplotlib
import matplotlib.pyplot as plt
import wandb
import os
import json

# Importing packages related to model performance
from sklearn.metrics import confusion_matrix # expects true_labels, predicted_labels
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report # expects true_labels, predicted_labels
from sklearn.metrics import roc_auc_score # expects true_labels, predicted_probs
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

import torch

logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger()
# logger.setLevel(logging.INFO)


#################
# Reporting
#################

def make_path_if_needed(file_path):
    directory = os.path.dirname(file_path)
    make_dir_if_needed(directory)
    return


def make_dir_if_needed(directory):
    if not os.path.exists(directory):
        logging.debug(f"Path did not exist; making directory {directory}")
        os.makedirs(directory)
    return


def report_df_info(*dataframes, n=5): # TODO: use this instead of load_df_verbose
    """
    Report information about an arbitrary number of dataframes.

    Parameters:
    *dataframes (pd.DataFrame): Arbitrary number of dataframes to report information about.
    n (int): Number of columns and indices to display.

    Returns:
    None

    # Example usage:
    data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    data2 = {'X': [7, 8, 9], 'Y': [10, 11, 12]}

    df1 = pd.DataFrame(data1, index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame(data2, index=['row4', 'row5', 'row6'])

    # Call the function to report information about the dataframes
    report_df_info(df1, df2)
    """

    for idx, df in enumerate(dataframes, start=1):
        print(f"----- DataFrame {idx} Info -----")
        print(f"Shape: {df.shape}")
        print(f"First {n} columns: {df.columns[:n].tolist()}")
        print(f"First {n} indices: {df.index[:n].tolist()}")
        print("-----")
    return


def report_df_info_with_names(df_dict, n=5):
    """
    Report information about dataframes (with DF names provided in a dictionary for convenience).

    Parameters:
    df_dict (dict): A dictionary where keys are names and values are dataframes.
    n (int): Number of columns and indices to display.

    Returns:
    None

    # Example usage:
    data1 = {'A': [1, 2, 3], 'B': [4, 5, 6]}
    data2 = {'X': [7, 8, 9], 'Y': [10, 11, 12]}

    df1 = pd.DataFrame(data1, index=['row1', 'row2', 'row3'])
    df2 = pd.DataFrame(data2, index=['row4', 'row5', 'row6'])

    # Create a dictionary with dataframe names
    dataframes_dict = {"DataFrame 1": df1, "DataFrame 2": df2}

    # Call the function with the dictionary of dataframes
    report_df_info_with_names(dataframes_dict)

    # Alternatively, use dict(zip()) instead of writing out a dictionary
    names = ['Dataframe 1', 'DF2']
    dfs = [df1, df2]
    report_df_info_with_names(dict(zip(names, dfs)))
    """

    for name, df in df_dict.items():
        print(f"----- DataFrame {name} Info -----")
        print(f"Shape: {df.shape}")
        print(f"First {n} columns: {df.columns[:n].tolist()}")
        print(f"First {n} indices: {df.index[:n].tolist()}")
        print("-----")
    return


#################
# Evaluation of model
#################

def get_loss_plot(train_losses, test_losses, 
                  train_label="Train loss", test_label="Validation loss"):
    logging.info("Making a loss plot over time")
    # Sample data
    epochs = range(1, len(train_losses)+1)

    # Plotting the lines
    plt.plot(epochs, train_losses, label=train_label)
    plt.plot(epochs, test_losses, label=test_label)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Model Loss')

    # Adding a legend
    plt.legend()
    return plt
    

def get_pnet_preds_and_probs(model, pnet_dataset):
    model.to('cpu') # TODO: is this needed?
    x = pnet_dataset.x
    additional = pnet_dataset.additional
    pred_probas = model.predict_proba(x, additional).detach()
    preds = model.predict(x, additional).detach()
    return preds, pred_probas


def get_performance_metrics(who, y_trues, y_preds, y_probas, save_dir=None):
    """
    Get and log useful performance metrics for a given data split (designated by 'who' parameter)
    """
    assert who in ['train', 'test', 'val', 'validation'], f"Expected one of train, test, val, or validation but got '{who}'"

    # TODO: add F1, auc_prc, and auc curve pdf as in Pnet.evaluate_interpret_save
    metric_dict = {
        f"{who}_acc": accuracy_score(y_trues, y_preds, normalize=True),
        f"{who}_balanced_acc": balanced_accuracy_score(y_trues, y_preds),
        f"{who}_roc_auc_score": roc_auc_score(y_trues, y_probas),
        f"{who}_cm": confusion_matrix(y_trues, y_preds),
        f"{who}classification report": classification_report(y_trues, y_preds)
    }

    logging.info(f"{who} set metrics:")
    for k,v in metric_dict:
        logging.info(f"{k}: {v}")

    if save_dir is not None:
        make_dir_if_needed(save_dir)
        p = os.path.join(save_dir, f'{who}_performance_metrics.json')
        logging.info(f"Saving dictionary of {who} set metrics to {p}")
        with open(p, 'w') as json_file:
            json.dump(metric_dict, json_file)

    # TODO: should I break the W&B pieces into a different function?
    logging.info(f"Saving {who} set metrics to W&B:")
    for k,v in metric_dict:
        wandb.run.summary[k] = v
    logging.info(f"Logging {who} set confusion matrix plot to W&B")
    wandb.sklearn.plot_confusion_matrix(y_trues, y_preds)

    return metric_dict


def get_summary_metrics_wandb(model, x_train, y_train, x_test, y_test):
    logging.info("Logging plot summary metrics to W&B")
    wandb.sklearn.plot_summary_metrics(model, x_train, y_train, x_test, y_test)
    return
    

def get_model_preds_and_probs(model, train_dataset, test_dataset, model_type = "pnet"):
    if model_type == "pnet":
        logging.info(f"Working with model_type = {model_type}")
        logging.info("Computing model predictions on train set")
        y_train_preds, y_train_probas = get_pnet_preds_and_probs(model, train_dataset)
        logging.info("Hist of model predictions on train set")
        plt.hist(y_train_preds)

        logging.info("Computing model predictions on test set")
        y_test_preds, y_test_probas = get_pnet_preds_and_probs(model, test_dataset)
        logging.info("Hist of model predictions on test set")
        plt.hist(y_test_preds)   
    else:
        logging.error(f"We haven't implemented for the model type you specified, which was {model_type}")
    return  y_train_preds, y_train_probas, y_test_preds, y_test_probas


def get_pnet_feature_importances(model, who, pnet_dataset, save_dir = None):
    """
    Args:
    - model: this is a Pnet model object
    - who: train, test, validation, or val. Which dataset are you using?
    - pnet_dataset: this is a Pnet dataset object (not a DF), and has attributes x, additional, y, etc.
    """
    gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = model.interpret(pnet_dataset)
    
    if save_dir is not None:
        make_dir_if_needed(save_dir)
        logging.info(f"Saving feature importance information to {save_dir}")
        gene_feature_importances.to_csv(os.path.join(save_dir, f'{who}_gene_feature_importances.csv'))
        additional_feature_importances.to_csv(os.path.join(save_dir, f'{who}_additional_feature_importances.csv'))
        gene_importances.to_csv(os.path.join(save_dir, f'{who}_gene_importances.csv'))
        for i, layer in enumerate(layer_importance_scores):
            layer.to_csv(os.path.join(save_dir, '{}_layer_{}_importances.csv'.format(who, i)))

    return gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores


def evaluate_interpret_save(model, pnet_dataset, who, save_dir): # TODO: start working here 10/12
    x = pnet_dataset.x
    additional = pnet_dataset.additional
    y_trues = pnet_dataset.y
    if save_dir is not None:
        make_dir_if_needed(save_dir)

    y_preds, y_probas = get_pnet_preds_and_probs(model, pnet_dataset)
    metric_dict = get_performance_metrics(who, y_trues, y_preds, y_probas, save_dir)    
    gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = get_pnet_feature_importances(model, who, pnet_dataset, save_dir)
    return
