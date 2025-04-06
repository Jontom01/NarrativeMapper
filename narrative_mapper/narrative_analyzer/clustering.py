import umap.umap_ as umap
import hdbscan
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import csv

def cluster_embeddings(
    embeddings_dict: list[dict], 
    n_components=20, 
    n_neighbors=20, 
    min_cluster_size=40, 
    min_samples=15,
    umap_kwargs=None,
    hdbscan_kwargs=None
    ) -> pd.DataFrame:
    """
    Reduces dimensionality of embedding vectors using UMAP and clusters them using HDBSCAN.

    Each dictionary must include an 'embedding_vector' key. After clustering, a 'cluster' label is added
    to each dictionary. The function returns a DataFrame with all original keys (excluding 'embedding_vector') 
    and the assigned 'cluster' label, excluding the noise cluster (cluster = -1).

    Parameters:
        embeddings_dict (list of dict): List of dictionaries with 'embedding_vector' keys.
        n_components (int): Target number of dimensions for UMAP reduction.
        n_neighbors (int): Number of neighbors for UMAP.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        min_samples (int): Minimum samples for HDBSCAN.
        umap_kwargs (dict): Allows for more UMAP input parameters
        hdbscan_kwargs (dict): Allows for more HDBSCAN input parameters

    Returns:
        pd.DataFrame: DataFrame of clustered items with a 'cluster' column.
    """
    embeddings = []
    for item in embeddings_dict:
        embeddings.append(item['embedding_vector'])
        
    #UMAP dimensionality:
    if umap_kwargs == None:
        umap_kwargs = {}

    umap_reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric='cosine',
        **umap_kwargs       
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)

    distance_matrix = pairwise_distances(reduced_embeddings, metric='cosine')
    distance_matrix = distance_matrix.astype('float64')

    #HDBSCAN clustering:
    if hdbscan_kwargs == None:
        hdbscan_kwargs = {}

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        **hdbscan_kwargs
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)

    i = 0
    for item in embeddings_dict:
        item['cluster'] = cluster_labels[i]
        i += 1

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"HDBSCAN found {num_clusters} clusters.")

    return_df = pd.DataFrame(embeddings_dict)

    return return_df