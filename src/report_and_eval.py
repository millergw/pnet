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
    

def get_pnet_preds_and_probs(pnet_dataset, model):
    model.to('cpu') # TODO: is this needed?
    x = pnet_dataset.x
    additional = pnet_dataset.additional
    pred_probas = model.predict_proba(x, additional).detach()
    preds = model.predict(x, additional).detach()
    return preds, pred_probas


def get_performance_metrics_old(model, train_dataset, test_dataset, config):
    
    x_train = train_dataset.x
    additional_train = train_dataset.additional
    y_train = train_dataset.y
    x_test = test_dataset.x
    additional_test = test_dataset.additional
    y_test = test_dataset.y

    logging.info("# making predictions on train and test sets")  
    logging.info("computing model predictions on train set")
    y_train_preds, y_train_probas = get_pnet_preds_and_probs(train_dataset, model)
    plt.hist(y_train_preds)

    logging.info("computing model predictions on test set")
    y_test_preds, y_test_probas = get_pnet_preds_and_probs(test_dataset, model)
    plt.hist(y_test_preds)


    logging.info("# calculating and logging useful performance metrics")
    train_score = accuracy_score(y_train, y_train_preds, normalize=True)
    test_score = accuracy_score(y_test, y_test_preds, normalize=True)
    wandb.run.summary["train_score"] = train_score # we use wandb.run.summary instead of wandb.log when we don't want multiple time steps
    wandb.run.summary["test_score"] = test_score
    wandb.run.summary["test_roc_auc_score"] = roc_auc_score(y_test, y_test_probas) # expects true_labels, predicted_probs
    wandb.run.summary['train_balanced_acc'] = balanced_accuracy_score(y_train, y_train_preds)
    wandb.run.summary['test_balanced_acc'] = balanced_accuracy_score(y_test, y_test_preds)
    
    logging.info(f"train score: {train_score}, test_score: {test_score}")
    logging.info(confusion_matrix(y_test, y_test_preds))
    
    wandb.log({
        "train_confusion_matrix": plot_confusion_matrix(y_train, y_train_preds), # expects true_labels, predicted_labels
        "test_confusion_matrix": confusion_matrix(y_test, y_test_preds),
        "test_classification_report": classification_report(y_test, y_test_preds) # expects true_labels, predicted_labels
              })
    
    wandb.sklearn.plot_confusion_matrix(y_test, y_test_preds)
    wandb.sklearn.plot_summary_metrics(model, x_train, y_train, x_test, y_test)
    
    # TODO: delete the stuff below here; I think it is redundant
    logging.info("train set metrics")
    cm_train = confusion_matrix(y_train, y_train_preds)
    print(classification_report(y_train, y_train_preds))
    print(roc_auc_score(y_train, y_train_probas))
    print(cm_train)

    logging.info("validation set metrics")
    cm_test = confusion_matrix(y_test, y_test_preds) # can also do (test_preds >= 0.5) instead of the list comprehension
    print(classification_report(y_test, y_test_preds)) 
    print(roc_auc_score(y_test, y_test_probas))
    print(cm_test)

    logging.info("# returning the predictions and prediction probabilities for train/test sets")
    return y_train_preds, y_test_preds, y_train_probas, y_test_probas


def get_performance_metrics(who, y_trues, y_preds, y_probas):
    """
    Get and log useful performance metrics for a given data split (designated by 'who' parameter)
    """
    assert who in ['train', 'test', 'val', 'validation'], f"Expected one of train, test, val, or validation but got '{who}'"

    logging.info("{who} set performance metrics calculation and logging")
    acc = accuracy_score(y_trues, y_preds, normalize=True)
    wandb.run.summary[f"{who}_acc"] = acc # we use wandb.run.summary instead of wandb.log when we don't want multiple time steps
    wandb.run.summary[f"{who}_roc_auc_score"] = roc_auc_score(y_trues, y_probas) # expects true_labels, predicted_probs
    wandb.run.summary[f'{who}_balanced_acc'] = balanced_accuracy_score(y_trues, y_preds)
    wandb.run.summary[f'{who}_cm'] = confusion_matrix(y_trues, y_preds)
    wandb.run.summary[f"{who}_classification_report"] = classification_report(y_trues, y_preds) # expects true_labels, predicted_labels
    
    # TODO: delete the stuff below here; I think it is redundant
    logging.info(f"{who} set metrics")
    cm = confusion_matrix(y_trues, y_preds)
    logging.info(f"{who}classification report \n{classification_report(y_trues, y_preds)}")
    logging.info(f"{who}_roc_auc_score \n{roc_auc_score(y_trues, y_probas)}")
    logging.info(f"{who} set: \nacc = {round(acc, 3)} \ncm = {cm}")
    return acc, cm


def get_performance_metrics_wandb(model, x_train, y_train, y_train_preds, x_test, y_test, y_test_preds):
    logging.info("Logging train set confusion matrix to W&B")
    wandb.sklearn.plot_confusion_matrix(y_train, y_train_preds)
    logging.info("Logging test set summary metrics to W&B")
    wandb.sklearn.plot_confusion_matrix(y_test, y_test_preds)
    logging.info("Logging plot summary metrics to W&B")
    wandb.sklearn.plot_summary_metrics(model, x_train, y_train, x_test, y_test)
    return
    

def get_model_preds_and_probs(model, train_dataset, test_dataset, model_type = "pnet"):
    if model_type == "pnet":
        logging.info(f"Working with model_type = {model_type}")
        logging.info("Computing model predictions on train set")
        y_train_preds, y_train_probas = get_pnet_preds_and_probs(train_dataset, model)
        logging.info("Hist of model predictions on train set")
        plt.hist(y_train_preds)

        logging.info("Computing model predictions on test set")
        y_test_preds, y_test_probas = get_pnet_preds_and_probs(test_dataset, model)
        logging.info("Hist of model predictions on test set")
        plt.hist(y_test_preds)   
    else:
        logging.error(f"We haven't implemented for the model type you specified, which was {model_type}")
    return  y_train_preds, y_train_probas, y_test_preds, y_test_probas