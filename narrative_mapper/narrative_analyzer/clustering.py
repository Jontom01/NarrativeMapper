from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import umap.umap_ as umap
import hdbscan
import pandas as pd

def cluster_embeddings(
    df, 
    n_components=20, 
    n_neighbors=20, 
    min_cluster_size=40, 
    min_samples=15,
    verbose=False,
    umap_kwargs=None,
    hdbscan_kwargs=None
    ) -> pd.DataFrame:
    """
    Reduces dimensionality of embedding vectors using UMAP and clusters them using HDBSCAN.

    Each dictionary must include an 'embedding_vector' key. After clustering, a 'cluster' label is added
    to each dictionary. The function returns a DataFrame with all original keys (excluding 'embedding_vector') 
    and the assigned 'cluster' label, excluding the noise cluster (cluster = -1).

    Parameters:
        df (DataFrame): DataFrame with embeddings column.
        n_components (int): Target number of dimensions for UMAP reduction.
        n_neighbors (int): Number of neighbors for UMAP.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        min_samples (int): Minimum samples for HDBSCAN.
        umap_kwargs (dict): Allows for more UMAP input parameters
        hdbscan_kwargs (dict): Allows for more HDBSCAN input parameters

    Returns:
        pd.DataFrame: DataFrame of clustered items with a 'cluster' column.
    """
    embeddings = df['embeddings'].tolist()

    embeddings = normalize(embeddings, norm='l2') #since both UMAP + HDBSCAN are setup for euclidean

    #PCA so UMAP doesn't assassinate my memory
    pca = PCA(n_components=100)
    reduced_embeddings = pca.fit_transform(embeddings)

    #UMAP dimensionality:
    progress_context_umap = (
    Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn()
        ) if verbose else nullcontext()
    )
    with progress_context_umap as progress:
        if verbose:
            task = progress.add_task("[cyan]UMAP reducing dimensions...", total=1)

        if umap_kwargs == None:
            umap_kwargs = {}

        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            metric='euclidean',
            **umap_kwargs       
        )
        reduced_embeddings = umap_reducer.fit_transform(reduced_embeddings)

        if verbose:
            progress.update(task, advance=1)
    
    #HDBSCAN clustering:
    progress_context_hdbscan = (
    Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TimeElapsedColumn()
        ) if verbose else nullcontext()
    )
    with progress_context_hdbscan as progress:
        if verbose:
            task = progress.add_task("[cyan]HDBSCAN clustering...", total=1)

        if hdbscan_kwargs == None:
            hdbscan_kwargs = {}

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            **hdbscan_kwargs
        )
        cluster_labels = clusterer.fit_predict(reduced_embeddings)

        if verbose:
            progress.update(task, advance=1)

    df = df.copy()
    df['cluster'] = cluster_labels.tolist()
    df = df[df['cluster'] != -1]

    return df