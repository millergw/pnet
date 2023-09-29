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
# General DF manipulations and filtering
##############################
def find_overlapping_columns(dataframes): # TODO: check if we should have a *dataframes here or not. Same for overlapping indicies function.
    logging.info("Finding overlapping columns in the given list of {len(dataframes)} datasets")
    logging.debug("Ensure that at least two DataFrames are provided")
    if len(dataframes) < 2:
        raise ValueError("At least two DataFrames are required for finding overlaps.")

    logging.debug("Get the columns of the first DataFrame")
    overlapping_columns = set(dataframes[0].columns)

    logging.debug("Find the intersection of columns with each subsequent DataFrame")
    for df in dataframes[1:]:
        overlapping_columns = overlapping_columns.intersection(df.columns)

    logging.info(f"We found {len(overlapping_columns)} overlapping columns")
    return list(overlapping_columns)


def find_overlapping_indices(*dataframes):
    logging.info("Finding overlapping indicies in the given {len(dataframes)} datasets")
    logging.debug("Ensure that at least two DataFrames are provided")    
    if len(dataframes) < 2:
        raise ValueError("At least two DataFrames are required for finding overlaps.")

    logging.debug("Get the indices of the first DataFrame")
    overlapping_indices = set(dataframes[0].index)

    logging.debug("Find the intersection of indices with each subsequent DataFrame")
    for df in dataframes[1:]:
        overlapping_indices = overlapping_indices.intersection(df.index)
        
    logging.info(f"We found {len(overlapping_indices)} overlapping indices")
    return list(overlapping_indices)


def find_overlapping_elements(*arrays):
    logging.debug("Ensure that at least two arrays are provided")
    if len(arrays) < 2:
        raise ValueError("At least two arrays are required for finding overlaps.")

    logging.debug("Get the elements of the first array")
    overlapping_elements = set(arrays[0])

    logging.debug("Find the intersection with each subsequent array")
    for a in arrays[1:]:
        overlapping_elements = overlapping_elements.intersection(a)

    logging.info(f"We found {len(overlapping_elements)} overlapping elements")
    return list(overlapping_elements)


def restrict_to_overlapping_indices(*dataframes):
    logging.debug("Find the overlapping indices among all DataFrames")
    overlapping_indices = find_overlapping_indices(*dataframes)
    logging.info(f"The number of overlapping indices amoung the {len(dataframes)} dataframes is {len(overlapping_indices)}.")

    logging.info(f"Restricting each DataFrame to the {len(overlapping_indices)} overlapping indices")
    restricted_dataframes = []
    for df in dataframes:
        logging.debug(f"Shape before: {df.shape}")
        restricted_df = df.loc[overlapping_indices]
        logging.debug(f"Shape after: {restricted_df.shape}")
        restricted_dataframes.append(restricted_df)
    return restricted_dataframes


def restrict_to_overlapping_columns(*dataframes):
    logging.debug("Find the overlapping columns among all DataFrames")
    overlapping_columns = find_overlapping_columns(*dataframes)
    logging.info(f"The number of overlapping columns amoung the {len(dataframes)} dataframes is {len(overlapping_columns)}.")

    logging.debug("Restricting each DataFrame to the overlapping columns")
    restricted_dataframes = []
    for df in dataframes:
        logging.debug(f"Shape before: {df.shape}")
        restricted_df = df[overlapping_columns]
        logging.debug(f"Shape after: {restricted_df.shape}")
        restricted_dataframes.append(restricted_df)
    return restricted_dataframes


def filter_to_specified_indices(indices, *dataframes):
    # Restrict each DataFrame to the specified indices
    restricted_dataframes = []
    for df in dataframes:
        logging.debug(f"Shape before: {df.shape}")
        restricted_df = df.loc[indices]
        logging.debug(f"Shape after: {restricted_df.shape}")
        restricted_dataframes.append(restricted_df)
    return restricted_dataframes


def filter_to_specified_columns(columns, *dataframes):
    # Restrict each DataFrame to the specified columns
    restricted_dataframes = []
    for df in dataframes:
        logging.debug(f"Shape before: {df.shape}")
        restricted_df = df[columns]
        logging.debug(f"Shape after: {restricted_df.shape}")
        restricted_dataframes.append(restricted_df)
    return restricted_dataframes


def load_df_verbose(f):
    logging.info(f"loading file at {f}")
    df = pd.read_csv(f)
    logging.debug(df.head())
    logging.debug(df.shape)
    return df


def is_binarized(df):
    """
    Ex:
    data = {'A': [0, 1, 1, 0],
    'B': [0, 0, 1, 1],
    'C': [1, 1, 0, 0]}

    binarized_df = pd.DataFrame(data)
    is_binarized(binarized_df)
    """
    return np.all((df.values == 0.) | (df.values == 1.))


def drop_na_index_rows(df):
    """
    Drop rows with missing (NaN) index values from a pandas DataFrame. This includes np.nan and None.
    
    Parameters:
    df (pd.DataFrame): The DataFrame from which to drop rows with missing index values.
    
    Returns:
    pd.DataFrame: A new DataFrame with rows containing missing index values removed.

    # Example usage:
    data = {'A': [1, 2, 3, 4, 5]}
    index_values = [np.nan, 'row2', 'row3', None, 'row5']
    df = pd.DataFrame(data, index=index_values)

    # Drop rows with missing index values from the DataFrame
    cleaned_df = drop_na_index_rows(df)

    print("Original DataFrame:")
    print(df)

    print("\nDataFrame after dropping rows with missing index values:")
    print(cleaned_df)
    """
    logging.info("Use boolean indexing to drop rows with NaN index values")
    logging.debug(f"Shape before: {df.shape}")
    cleaned_df = df[~df.index.isna()]
    logging.debug(f"Shape after: {cleaned_df.shape}")
    return cleaned_df


############################## 
# Value conversion, value mapping, value imputation
##############################

def impute_cols_with_a_constant(df, new_col_names, fill=0):
    """
    Fill new columns with specified `fill` value. Inputs: arbitrary number of dataframes.
    The point here is to be able to pass mutiple dataframes in
    """
    new_col_names = list(new_col_names)
    logging.info(f"Shape before {fill}-imputation: {df.shape}")
    logging.info(f"We have {len(new_col_names)} features to add as a column of all {fill}'s")
    df = df.reindex(columns=df.columns.tolist() + new_col_names).fillna(fill)
    logging.info(f"Shape after {fill}-imputation: {df.shape}")
    return df


def impute_cols_with_a_constant_v2(new_col_names, fill=0, *dataframes):
    """
    Fill new columns with specified `fill` value. Inputs: arbitrary number of dataframes.
    The point here is to be able to pass mutiple dataframes in
    """
    new_col_names = list(new_col_names)
    imputed_dataframes = []
    for df in dataframes:
        logging.info(f"Shape before {fill}-imputation: {df.shape}")
        logging.info(f"We have {len(new_col_names)} features to add as a column of all {fill}'s")
        imputed_df = df.reindex(columns=df.columns.tolist() + new_col_names).fillna(fill)
        logging.info(f"Shape after {fill}-imputation: {imputed_df.shape}")
        imputed_dataframes.append(imputed_df)
    return dataframes


def convert_values(input_value, source, target):
    """
    # Example usage:
    value1_list = ['apple', 'banana', 'cherry']
    value2_list = ['red', 'yellow', 'red']

    # Convert a single value
    conversion_result = convert_values('kiwi', value1_list, value2_list)
    print(f"Converted List: {conversion_result}")
    # > Converted List: 'yellow'
    # Convert a list of values
    input_list = ['apple', 'banana', 'kiwi']
    conversion_result = convert_values(input_list, value1_list, value2_list)
    print(f"Converted List: {conversion_result}")
    # > Converted List: ['red', 'yellow', 'kiwi']
    """

    logging.debug("Ensure source and target have the same length")
    if len(source) != len(target):
        raise ValueError("Input lists must have the same length.")

    logging.info("Converting input by creating a dictionary to map values from 'source' to 'target'")
    value_mapping = dict(zip(source, target))

    logging.debug("Initialize lists to track converted and unconverted items")
    converted_items = []
    unconverted_items = []

    if isinstance(input_value, list):
        logging.debug("If input_value is a list, convert each element")
        for item in input_value:
            converted_value = value_mapping.get(item, None)
            if converted_value is not None:
                converted_items.append(converted_value)
            else:
                converted_items.append(np.nan)
                unconverted_items.append(item)
    else:
        logging.debug("If input_value is a single value, convert it")
        converted_value = value_mapping.get(input_value, None)
        if converted_value is not None:
            converted_items.append(converted_value)
        else:
            converted_items.append(np.nan)
            unconverted_items.append(input_value)

    logging.debug("{len(converted_items)} converted: {converted_items}")
    if len(unconverted_items)>0:
        logging.warn("{len(unconverted_items)} couldn't be converted: {unconverted_items}")
    return converted_items


def convert_values_old(input_value, source, target):
    """
    # Example usage
    source = ['apple', 'banana', 'cherry']
    target = ['red', 'yellow', 'red']

    # Convert a single value
    converted_value = convert_values('banana', source, target)
    print(f"Converted Value: {converted_value}") 
    # > Converted Value: 'yellow'

    # Convert a list of values
    input_list = ['apple', 'banana', 'kiwi']
    converted_list = convert_values(input_list, source, target)
    print(f"Converted List: {converted_list}")
    # > Converted List: ['red', 'yellow', 'kiwi']
    """
    logging.info("Converting input by creating a dictionary to map values from 'source' to 'target'")
    value_mapping = dict(zip(source, target))

    if isinstance(input_value, list):
        logging.debug("If input_value is a list, convert each element")
        output = list(map(lambda x: value_mapping.get(x, np.nan), input_value))
    else:
        logging.debug("If input_value is a single value, convert it")
        output =  value_mapping.get(input_value, np.nan)
    
    if np.nan in output:
        logging.warn("The converted list contains np.nan values.")
    return output



