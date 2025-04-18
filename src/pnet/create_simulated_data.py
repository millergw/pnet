import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_probabilities(df):
    """
    Calculate gene-level mutation probabilities from a binary dataset.

    Parameters:
        df (pd.DataFrame): Binary dataset (samples x genes).

    Returns:
        pd.Series: Mutation probabilities for each gene.
    """
    logging.info("Calculating gene-level probabilities...")
    probabilities = df.mean(axis=0)
    logging.info("Probabilities calculated:\n%s", probabilities.head())
    return probabilities


def plot_probabilities(probabilities, title="Gene Mutation Probabilities"):
    """
    Plot gene-level mutation probabilities.

    Parameters:
        probabilities (pd.Series): Mutation probabilities for each gene.
        title (str): Plot title.
    """
    logging.info("Plotting probabilities: %s", title)
    plt.figure(figsize=(10, 6))
    plt.bar(probabilities.index, probabilities.values, color="skyblue")
    plt.title(title)
    plt.xlabel("Genes")
    plt.ylabel("Probability")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_probability_histogram(probabilities, bins=10, title="Histogram of Gene Mutation Probabilities"):
    """
    Plot a histogram of gene-level mutation probabilities.

    Parameters:
        probabilities (pd.Series): Mutation probabilities for each gene.
        bins (int): Number of bins for the histogram.
        title (str): Plot title.
    """
    logging.info("Plotting histogram: %s", title)
    plt.figure(figsize=(10, 6))
    plt.hist(probabilities.values, bins=bins, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def plot_probability_dotplot(probabilities, title="Dot Plot of Gene Mutation Probabilities"):
    """
    Plot a dot plot of gene-level mutation probabilities where x-axis is the probability
    and y-axis is jittered to visualize distribution.

    Parameters:
        probabilities (pd.Series): Mutation probabilities for each gene.
        title (str): Plot title.
    """
    logging.info("Plotting dot plot: %s", title)
    plt.figure(figsize=(10, 6))
    x_values = probabilities.values
    y_values = [0] * len(x_values)  # Add jitter to y-axis
    plt.scatter(x_values, y_values, color="skyblue", alpha=0.7)
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.xlabel("Probability")
    plt.ylabel("Jitter")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.show()


def generate_perturbed_data(df, perturb_type="shift", k=3, sample_fraction=0.5, gene_fraction=0.5):
    """
    Generate a perturbed binary dataset with controlled perturbations.

    Parameters:
        df (pd.DataFrame): Original binary dataset (samples x genes).
        perturb_type (str): Type of perturbation ('shift', 'scale', 'constant_offset').
        k (float): Perturbation factor (e.g., number of std devs for 'shift', constant offset value).
        sample_fraction (float): Fraction of samples to perturb (0-1).
        gene_fraction (float): Fraction of genes to perturb (0-1).

    Returns:
        pd.DataFrame: Perturbed dataset.
    """
    logging.info("Starting perturbation process...")
    np.random.seed(42)  # For reproducibility

    # Calculate original probabilities
    logging.info("Calculating original probabilities...")
    gene_probabilities = calculate_probabilities(df)

    # Select genes and samples to perturb
    num_genes = int(gene_fraction * df.shape[1])
    num_samples = int(sample_fraction * df.shape[0])
    logging.info("Perturbing %d genes and %d samples...", num_genes, num_samples)
    perturbed_genes = np.random.choice(df.columns, size=num_genes, replace=False)
    perturbed_samples = np.random.choice(df.index, size=num_samples, replace=False)
    logging.info("Selected genes for perturbation:\n%s", perturbed_genes)
    logging.info("Selected samples for perturbation:\n%s", perturbed_samples)

    # Perturb probabilities
    perturbed_probabilities = gene_probabilities.copy()
    if perturb_type == "shift":
        logging.info("Applying 'shift' perturbation...")
        perturbed_probabilities[perturbed_genes] = np.minimum(
            gene_probabilities[perturbed_genes]
            + k * np.sqrt(gene_probabilities[perturbed_genes] * (1 - gene_probabilities[perturbed_genes])),
            1,
        )
    elif perturb_type == "scale":
        logging.info("Applying 'scale' perturbation...")
        perturbed_probabilities[perturbed_genes] = np.minimum(gene_probabilities[perturbed_genes] * k, 1)
    elif perturb_type == "constant_offset":
        logging.info("Applying 'constant_offset' perturbation...")
        perturbed_probabilities[perturbed_genes] = np.minimum(gene_probabilities[perturbed_genes] + k, 1)

    logging.info("Perturbed probabilities:\n%s", perturbed_probabilities.head())

    # Generate perturbed dataset
    perturbed_df = df.copy()
    logging.info("Generating perturbed dataset...")
    for gene in perturbed_genes:
        perturbed_df.loc[perturbed_samples, gene] = np.random.binomial(
            1, perturbed_probabilities[gene], size=num_samples
        )

    logging.info("Perturbed dataset generated.")
    return perturbed_df


# Example usage
if __name__ == "__main__":
    # Generate a toy dataset
    np.random.seed(42)
    logging.info("Generating original dataset...")
    original_df = pd.DataFrame(np.random.binomial(1, 0.1, size=(1000, 50)), columns=[f"Gene_{i}" for i in range(50)])
    logging.info("Original dataset generated.")

    # Calculate probabilities
    probabilities = calculate_probabilities(original_df)

    # Plot probabilities
    plot_probabilities(probabilities, title="Original Gene Mutation Probabilities")

    # Plot histogram of probabilities
    plot_probability_histogram(probabilities, bins=10, title="Histogram of Original Gene Mutation Probabilities")

    # Plot dot plot of probabilities
    plot_probability_dotplot(probabilities, title="Dot Plot of Original Gene Mutation Probabilities")

    # Generate perturbed dataset
    perturbed_df = generate_perturbed_data(
        original_df, perturb_type="shift", k=3, sample_fraction=0.5, gene_fraction=0.5
    )

    # Calculate and plot perturbed probabilities
    perturbed_probabilities = calculate_probabilities(perturbed_df)
    plot_probabilities(perturbed_probabilities, title="Perturbed Gene Mutation Probabilities")
    plot_probability_histogram(
        perturbed_probabilities, bins=10, title="Histogram of Perturbed Gene Mutation Probabilities"
    )
    plot_probability_dotplot(perturbed_probabilities, title="Dot Plot of Perturbed Gene Mutation Probabilities")
