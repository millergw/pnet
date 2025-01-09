import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import random
import numpy as np


# DataLoader object for pytorch. Constructing single loader for all data input modalities.
class PnetDataset(Dataset):
    def __init__(self, genetic_data, target, indicies, additional_data=None, gene_set=None):
        """
        A dataset class for PyTorch that handles the loading and integration of multiple genetic data modalities.

        This class combines data from different genetic modalities (e.g., `mut`, `cnv`), links them to target labels,
        and supports batching for PyTorch. It ensures consistent handling of genes across modalities and can incorporate
        additional sample-specific features.

        Parameters:
        ----------
        genetic_data : Dict(str: pd.DataFrame)
            A dictionary of genetic modalities, where keys are modality names (e.g., 'mut', 'cnv') and values are
            pandas DataFrames with samples as rows and genes as columns. Paired samples must have matching indices
            across all modalities.
        target : pd.Series or pd.DataFrame
            The target variable for each sample. Can be binary or continuous, provided as a pandas Series or DataFrame
            with samples as the index.
        indicies : list of str
            A list of sample indices to include in the dataset.
        additional_data : pd.DataFrame, optional
            Additional features for each sample, indexed by sample names. Default is None.
        gene_set : list of str, optional
            A list of genes to be considered. By default, all overlapping genes across modalities are included.
        """

        assert isinstance(genetic_data, dict), f"input data expected to be a dict, got {type(genetic_data)}"
        for inp in genetic_data:
            assert isinstance(inp, str), f"input data keys expected to be str, got {type(inp)}"
            assert isinstance(genetic_data[inp], pd.DataFrame), (
                f"input data values expected to be a dict, got" f" {type(genetic_data[inp])}"
            )
        self.genetic_data = genetic_data
        self.target = target
        self.gene_set = gene_set
        self.altered_inputs = []
        self.inds = indicies
        if additional_data is not None:
            self.additional_data = additional_data.loc[self.inds]
        else:
            self.additional_data = pd.DataFrame(index=self.inds)  # create empty dummy dataframe if no additional data
        self.target = self.target.loc[self.inds]
        self.genes = self.get_genes()
        self.input_df = self.unpack_input()
        assert self.input_df.index.equals(self.target.index)
        self.x = torch.tensor(self.input_df.values, dtype=torch.float)
        self.y = torch.tensor(self.target.values, dtype=torch.float)
        self.additional = torch.tensor(self.additional_data.values, dtype=torch.float)

    def __len__(self):
        return self.input_df.shape[0]

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        additional = self.additional[index]
        return x, additional, y

    def get_genes(self):
        """
        Generate list of genes which are present in all data modalities and in the list of genes to be considered
        :return: List(str); List of gene names
        """
        # drop duplicated columns:
        for inp in self.genetic_data:
            self.genetic_data[inp] = self.genetic_data[inp].loc[:, ~self.genetic_data[inp].columns.duplicated()].copy()
        gene_sets = [set(self.genetic_data[inp].columns) for inp in self.genetic_data]
        if self.gene_set:
            gene_sets.append(self.gene_set)
        genes = list(set.intersection(*gene_sets))
        print("Found {} overlapping genes".format(len(genes)))
        return genes

    def unpack_input(self):
        """
        Unpacks data modalities into one joint pd.DataFrame. Suffixing gene names by their modality name.
        :return: pd.DataFrame; containing n*m columns, where n is the number of modalities and m the number of genes
        considered.
        """
        input_df = pd.DataFrame(index=self.inds)
        for inp in self.genetic_data:
            temp_df = self.genetic_data[inp][self.genes]
            temp_df.columns = temp_df.columns + "_" + inp
            input_df = input_df.join(temp_df, how="inner", rsuffix="_" + inp)
        print("generated input DataFrame of size {}".format(input_df.shape))
        return input_df.loc[self.inds]

    def save_indicies(self, path):
        df = pd.DataFrame(data={"indicies": self.inds})
        df.to_csv(path, sep=",", index=False)


# Dataset class that extends PnetDataset to include global gene embeddings.
class PnetDatasetWithGlobalEmbeddings(PnetDataset):
    """
    A dataset class that extends PnetDataset to include global gene embeddings.

    This class integrates global embeddings for each gene into the dataset.
    Each sample's feature vector contains:
        - Modality-specific features for each gene (e.g., `mut`, `cnv`).
        - A single global embedding for each gene, shared across modalities.

    Attributes:
    ----------
    genetic_data : Dict(str: pd.DataFrame)
        A dictionary where keys are modality names (e.g., 'mut', 'cnv') and values are
        pandas DataFrames containing samples as rows and genes as columns.
    target : pd.Series
        A Series containing the target variable for each sample.
    indices : list
        A list of sample indices to include in the dataset.
    gene_embeddings : pd.DataFrame
        A DataFrame where rows are genes and columns represent the global embedding features.
    additional_data : pd.DataFrame, optional
        Additional features for each sample, indexed by sample names.
    gene_set : list, optional
        A list of genes to include; if None, all overlapping genes across modalities are included.

    Methods:
    -------
    unpack_input():
        Combines genetic data and global embeddings into a unified input DataFrame for the model.
    """

    def __init__(self, genetic_data, target, indicies, gene_embeddings, additional_data=None, gene_set=None):
        self.gene_embeddings = gene_embeddings
        super().__init__(genetic_data, target, indicies, additional_data, gene_set)

    def unpack_input(self):
        """
        Combines modality-specific genetic data and global gene embeddings into a single DataFrame.

        The resulting DataFrame includes:
        - Modality-specific features for each gene.
        - A single global embedding for each gene, expanded to match the number of samples.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the combined genetic data and global embeddings for all samples.
            Shape: (samples, n_genes x (n_modalities+embedding_length))
        """
        # Initialize the combined input DataFrame
        input_df = pd.DataFrame(index=self.inds)

        # Add modality-specific genetic data
        for modality_name in self.genetic_data:
            modality_data = self.genetic_data[modality_name][self.genes]
            modality_data.columns = [f"{col}_{modality_name}" for col in modality_data.columns]
            input_df = input_df.join(modality_data, how="inner")

        # Expand global embeddings to match the number of samples
        gene_emb_expanded = pd.DataFrame(
            np.tile(self.gene_embeddings.loc[self.genes].values.flatten(), len(input_df.index)).reshape(
                len(input_df.index), -1
            ),
            index=input_df.index,
            columns=[f"{gene}_embedding{i+1}" for gene in self.genes for i in range(self.gene_embeddings.shape[1])],
        )

        # Combine genetic data and global embeddings
        input_df = pd.concat([input_df, gene_emb_expanded], axis=1)
        return input_df


def get_indicies(genetic_data, target, additional_data=None):
    """
    Generates a list of indicies which are present in all data modalities. Drops duplicated indicies.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
     genetic data.
    :return: List(str); List of sample names found in all data modalities
    """
    for gd in genetic_data:
        genetic_data[gd].dropna(inplace=True)
    target.dropna(inplace=True)
    ind_sets = [set(genetic_data[inp].index.drop_duplicates(keep=False)) for inp in genetic_data]
    ind_sets.append(target.index.drop_duplicates(keep=False))
    if additional_data is not None:
        ind_sets.append(additional_data.index.drop_duplicates(keep=False))
    inds = list(set.intersection(*ind_sets))
    print("Found {} overlapping indicies".format(len(inds)))
    return inds


def generate_train_test(
    genetic_data,
    target,
    gene_set=None,
    additional_data=None,
    test_split=0.3,
    seed=None,
    train_inds=None,
    test_inds=None,
    collinear_features=0,
    shuffle_labels=False,
):
    """
    Takes all data modalities to be used and generates a train and test DataSet with a given split.
    :param genetic_data: Dict(str: pd.DataFrame); requires a dict containing a pd.DataFrame for each data modality
         and the str identifier. Paired samples should have matching indicies across Dataframes.
    :param target: pd.DataFrame or pd.Series; requires a single pandas Dataframe or Series with target variable
        paired per sample index. Target can be binary or continuous.
    :param gene_set: List(str); List of genes to be considered, default is None and considers all genes found in every
        data modality.
    :param additional_data: pd.DataFrame; Dataframe with additional information per sample. Sample IDs should match
    :param test_split: float; Fraction of samples to be used for testing.
    :param seed: int; Random seed to be used for train/test splits.
    :return:
    """
    print("Given {} Input modalities".format(len(genetic_data)))
    inds = get_indicies(genetic_data, target, additional_data)
    random.seed(seed)
    random.shuffle(inds)
    if train_inds and test_inds:
        train_inds = list(set(inds).intersection(train_inds))
        test_inds = list(set(inds).intersection(test_inds))
    elif train_inds:
        train_inds = list(set(inds).intersection(train_inds))
        test_inds = [i for i in inds if i not in train_inds]
    elif test_inds:
        test_inds = list(set(inds).intersection(test_inds))
        train_inds = [i for i in inds if i not in test_inds]
    else:
        test_inds = inds[int((len(inds) + 1) * (1 - test_split)) :]
        train_inds = inds[: int((len(inds) + 1) * (1 - test_split))]
    print("Initializing Train Dataset")
    train_dataset = PnetDataset(genetic_data, target, train_inds, additional_data=additional_data, gene_set=gene_set)
    print("Initializing Test Dataset")
    test_dataset = PnetDataset(genetic_data, target, test_inds, additional_data=additional_data, gene_set=gene_set)

    # Positive control: Replace a gene's values with values collinear to the target
    train_dataset, test_dataset = add_collinear(train_dataset, test_dataset, collinear_features)
    # Positive control: Shuffle labels for prediction
    if shuffle_labels:
        train_dataset = shuffle_data_labels(train_dataset)
        test_dataset = shuffle_data_labels(test_dataset)
    return train_dataset, test_dataset


def to_dataloader(train_dataset, test_dataset, batch_size):
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(123)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,)
    # val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,)
    # based on https://pytorch.org/docs/stable/notes/randomness.html for reproducibility of DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return train_loader, val_loader


def add_collinear(train_dataset, test_dataset, collinear_features):
    if isinstance(collinear_features, list):
        for f in collinear_features:
            replace_collinear(train_dataset, test_dataset, f)
    else:
        for n in range(collinear_features):
            r = random.randint(0, len(train_dataset.input_df.columns))
            altered_input_col = train_dataset.input_df.columns[r]
            train_dataset, test_dataset = replace_collinear(train_dataset, test_dataset, altered_input_col)
    return train_dataset, test_dataset


def shuffle_data_labels(dataset):
    print("shuffling {} labels".format(dataset.target.shape[0]))
    target_copy = dataset.target.copy()
    target_copy[target_copy.columns[0]] = dataset.target.sample(frac=1).reset_index(drop=True).values
    dataset.target = target_copy
    return dataset


def replace_collinear(train_dataset, test_dataset, altered_input_col):
    train_dataset.altered_inputs.append(altered_input_col)
    test_dataset.altered_inputs.append(altered_input_col)
    print("Replace input of: {} with collinear feature.".format(altered_input_col))
    train_dataset.input_df[altered_input_col] = train_dataset.target
    test_dataset.input_df[altered_input_col] = test_dataset.target
    return train_dataset, test_dataset
