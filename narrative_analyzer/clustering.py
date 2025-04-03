import umap.umap_ as umap
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd

def cluster_embeddings(embeddings_dict: list[dict]):

    embeddings = []
    for item in embeddings_dict:
        embeddings.append(item['embedding_vector'])
    # UMAP dimensionality reduction with tuned parameters:
    umap_reducer = umap.UMAP(
        n_neighbors=70,       # Controls local versus global structure; lower values capture local structure
        n_components=50,       # Reduce embeddings to 2 dimensions (use 3 for 3D visualizations)
        metric='cosine',      # Use cosine similarity; you can change to 'euclidean' if preferred
        min_dist=0.0,         # Minimum distance between points in the low-dimensional space
        random_state=42       # For reproducibility
    )
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    distance_matrix = pairwise_distances(reduced_embeddings, metric='cosine')
    distance_matrix = distance_matrix.astype('float64')

    # HDBSCAN clustering with additional tuning:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=50,            # Minimum number of points to form a cluster
        min_samples=5,                  # How conservative the clustering is; adjust based on your data
        metric='precomputed',             # Distance metric used for clustering
        cluster_selection_method='eom'  # Method to select clusters ('eom' or 'leaf')
    )
    cluster_labels = clusterer.fit_predict(distance_matrix)
    i = 0
    for item in embeddings_dict:
        item['cluster'] = cluster_labels[i]
        i += 1

    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"HDBSCAN found {num_clusters} clusters.")

    return_df = pd.DataFrame(embeddings_dict)
    return_df = return_df.drop(columns=['embedding_vector'])
    return return_df