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
# from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import (
    confusion_matrix, # expects true_labels, predicted_labels
    classification_report, # expects true_labels, predicted_labels
    roc_auc_score, # expects true_labels, predicted_probs
    average_precision_score, # aka the AUC-PRC score
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    accuracy_score,
    mean_squared_error,
)
import torch

logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()


#################
# Reporting
#################

def make_path_if_needed(file_path):
    directory = os.path.dirname(file_path)
    make_dir_if_needed(directory)
    return


def make_dir_if_needed(directory):
    if not os.path.isdir(directory) and directory != '':
        logging.debug(f"Directory did not exist; making directory {directory}")
        os.makedirs(directory)
    return


def savefig(plt, save_path, png=True, pdf=True):
    make_path_if_needed(save_path)
    logging.info(f"saving plot to {save_path}")
    if png:
        plt.savefig(save_path, bbox_inches='tight')
    if pdf:
        plt.savefig(f"{save_path}.pdf", format="pdf", bbox_inches='tight')


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
                  train_label="Train loss", test_label="Validation loss",
                  title="Model Loss", ylabel="Loss", xlabel="Epochs"):
    logging.info("Making a loss plot over time")
    # Sample data
    epochs = range(1, len(train_losses)+1)

    # Plotting the lines
    plt.plot(epochs, train_losses, label=train_label)
    plt.plot(epochs, test_losses, label=test_label)

    # Adding labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

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

    # TODO: add auc curve pdf as in Pnet.evaluate_interpret_save
    # logging.info(f"Logging {who} set ROC plot to W&B")

    metric_dict = {
        f"{who}_acc": accuracy_score(y_trues, y_preds, normalize=True),
        f"{who}_balanced_acc": balanced_accuracy_score(y_trues, y_preds),
        f"{who}_roc_auc_score": roc_auc_score(y_trues, y_probas),
        f"{who}_average_precision_score": average_precision_score(y_trues, y_probas),
        f"{who}_f1_score": f1_score(y_trues, y_preds),
        f"{who}_confusion_matrix\n": confusion_matrix(y_trues, y_preds).tolist(),
        f"{who}_classification report\n": classification_report(y_trues, y_preds)
    }

    logging.info(f"{who} set metrics:")
    for k,v in metric_dict.items():
        logging.info(f"{k}: {v}")

    if save_dir is not None:
        make_dir_if_needed(save_dir)
        p = os.path.join(save_dir, f'{who}_performance_metrics.json')
        logging.info(f"Saving dictionary of {who} set metrics to {p}")
        with open(p, 'w') as json_file:
            json.dump(metric_dict, json_file)
            
    # TODO: should I break the W&B pieces into a different function?
    logging.info(f"Saving {who} set metrics to W&B:")
    for k,v in metric_dict.items():
        wandb.run.summary[k] = v
    logging.info(f"Logging {who} set confusion matrix plot to W&B")
    wandb.sklearn.plot_confusion_matrix(y_trues, y_preds)

    return metric_dict


def get_summary_metrics_wandb(model, x_train, y_train, x_test, y_test):
    logging.info("Logging plot summary metrics to W&B")
    wandb.sklearn.plot_summary_metrics(model, x_train, y_train, x_test, y_test)
    return
    

def get_train_test_manual_split(x, y, train_inds, test_inds):
    """
    Get manual train-test split based on specified indices.

    Parameters:
    - x: Features (input data, e.g., NumPy array or pandas DataFrame).
    - y: Labels (output data, e.g., NumPy array or pandas Series).
    - train_inds: List of indices for the training set.
    - test_inds: List of indices for the testing set.

    Returns:
    - X_train: Features for the training set.
    - X_test: Features for the testing set.
    - y_train: Labels for the training set.
    - y_test: Labels for the testing set.
    """
    logging.info("CAUTION: this `get_train_test_manual_split` is an untested function. TODO: check functionality.")

    # Assuming 'x' is a NumPy array or pandas DataFrame
    X_train = x[train_inds]
    X_test = x[test_inds]

    # Assuming 'y' is a NumPy array or pandas Series
    y_train = y[train_inds]
    y_test = y[test_inds]

    return X_train, X_test, y_train, y_test



def get_model_preds_and_probs(model, who, model_type = "pnet", pnet_dataset=None, x=None, verbose=False):
    logging.info(f"Model_type = {model_type}. Computing model predictions on {who} set")
    if model_type == 'pnet':
        y_preds, y_probas = get_pnet_preds_and_probs(model, pnet_dataset)
    elif model_type in ['rf', 'bdt']:
        y_preds, y_probas = get_sklearn_model_preds_and_probs(model, x)
    else:
        logging.error(f"We haven't implemented for the model type you specified, which was {model_type}")
    if verbose:
        logging.info(f"Hist of model prediction probabilities on {who} set")
        plt.hist(y_probas)
        plt.show()
    return  y_preds, y_probas


def get_sklearn_model_preds_and_probs(sklearn_model, x):
    preds = sklearn_model.predict(x)
    pred_probs = sklearn_model.predict_proba(x)
    return preds, pred_probs


def get_sklearn_feature_importances(sklearn_model, who, input_df, save_dir=None):
    importances = sklearn_model.feature_importances_
    gene_feature_importances = pd.Series(importances, index=input_df.columns) # TODO: check if this is the correct index
    if save_dir is not None:
        make_dir_if_needed(save_dir)
        logging.info(f"Saving feature importance information to {save_dir}")
        gene_feature_importances.to_csv(os.path.join(save_dir,f'{who}_gene_feature_importances.csv'))
        # wandb.save(f'{who}_gene_feature_importances.csv', base_path=save_dir, policy="end") # TODO: problem. save_dir is above current dir, and this isn't allowed.
    return gene_feature_importances


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


def evaluate_interpret_save(model, who, model_type, pnet_dataset=None, x=None, y=None, save_dir=None):
    """
    For a given trained model of type `model_type' (e.g. P-NET, RF, BDT), get the model predictions, performance metrics, feature importances, and (optionally) save the results.
    The model is evaluated on the `dataset`.
    """
    if save_dir is not None:
        make_dir_if_needed(save_dir)
        logging.info(f"Results will be saved to {save_dir}.")
    
    # TODO: need a universal way to get the X vs y components of the `pnet_dataset`
    if model_type == 'pnet':
        y = pnet_dataset.y
        logging.info(f"Getting the {model_type} model predictions on the {who} set, performance metrics, and feature importances (if applicable)")
        y_preds, y_probas = get_model_preds_and_probs(model=model, pnet_dataset=pnet_dataset, who=who, model_type=model_type)
        metric_dict = get_performance_metrics(who, y, y_preds, y_probas, save_dir) # TODO: this is universal, and should get pulled out   
        gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores = get_pnet_feature_importances(model, who, pnet_dataset, save_dir)
        importances = {
            'gene_feature_importances':gene_feature_importances, 
            'additional_feature_importances':additional_feature_importances, 
            'gene_importances':gene_importances, 
            'layer_importance_scores':layer_importance_scores,
        }
        for name,dat in importances.items():
            save_as_file_to_wandb(dat, f'{who}_{name}.csv')
            # wandb.run.summary[k] = v.to_dict() # TODO: haven't tested and I expect it might not work for everything, some some might be a list?
        return metric_dict, gene_feature_importances, additional_feature_importances, gene_importances, layer_importance_scores

    elif model_type in ['rf', 'bdt']:
        logging.info(f"Getting the {model_type} model predictions on the {who} set, performance metrics, and feature importances (if applicable)")
        x = pnet_dataset.x
        y = pnet_dataset.y.ravel()
        input_df = pnet_dataset.input_df

        y_preds, y_probas = get_model_preds_and_probs(model=model, x=x, who=who, model_type=model_type)
        # If the positive class is class 1, use the probabilities for that class
        y_probas = y_probas[:, 1]

        metric_dict = get_performance_metrics(who, y, y_preds, y_probas, save_dir)    
        gene_feature_importances = get_sklearn_feature_importances(model, who=who, input_df=input_df, save_dir=save_dir)
        # TODO: trying different ways of saving the feature importances (as a dict, a file, and as two separate lists)
        wandb.run.summary['gene_feature_importances'] = gene_feature_importances.to_dict()
        wandb.run.summary['gene_feature_importances_names'] = gene_feature_importances.keys()
        wandb.run.summary['gene_feature_importances_values'] = gene_feature_importances.values()
        save_as_file_to_wandb(gene_feature_importances, f'{who}_gene_feature_importances.csv')
        return gene_feature_importances
    
    else:
        logging.error(f"We haven't implemented for the model type you specified, which was {model_type}")
    return


def save_as_file_to_wandb(data, filename, policy='now', delete_local=True):
    logging.info("Temporarily save down to {filename}, upload to WandB.")
    data.to_csv(filename)
    wandb.save(filename, policy=policy)
    if delete_local:
        logging.info(f"Deleting the temporary file at {filename}")
        os.remove(filename)
    return


def get_deviance(clf, x_test, y_test):
    """
    TODO: check functionality for RF. Works for BDT. 
    Thinking about the "Plot training deviance" section from https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py
    """
    test_score = np.zeros((clf.n_estimators_,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_predict(x_test)):
        test_score[i] = mean_squared_error(y_test, y_pred)
    return clf.train_score_, test_score