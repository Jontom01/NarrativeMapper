from .embeddings import get_embeddings
from .clustering import cluster_embeddings
from .summarize import summarize_clusters
from .formatters import format_by_text, format_by_cluster, format_to_dict
import pandas as pd

class NarrativeMapper:
    """
    A pipeline for processing, clustering, summarizing, and formatting text data.
    
    Methods allow you to load embeddings from a file, perform clustering,
    generate cluster summaries (using OpenAI Chat Completions and sentiment analysis),
    and format the results into various output structures.
    """
    
    def __init__(self, online_group_name: str):
        """
        Initializes the NarrativeMapper instance.
        
        Parameters:
            online_group_name (str): Name of the online community (e.g. subreddit) to label outputs.
        """
        self.online_group_name = online_group_name
        self.embeddings = None
        self.cluster_df = None
        self.summary_df = None

    def load_embeddings(self, file_path: str, chunk_size: int=500) -> "NarrativeMapper":
        """
        Loads and processes text data to obtain OpenAI embeddings.
        
        Parameters:
            file_path (str): Path to the .csv file containing text data.
            chunk_size (int): length of text-list chunks being send to OpenAI embeddings
        
        Returns:
            NarrativeMapper: Self, with embeddings loaded.
        """
        self.embeddings = get_embeddings(file_path, chunk_size)
        return self

    def cluster(self, n_components: int = 20, n_neighbors: int = 20,
                min_cluster_size: int = 40, min_samples: int = 15) -> "NarrativeMapper":
        """
        Applies UMAP for dimensionality reduction and HDBSCAN for clustering
        on the loaded embeddings.
        
        Parameters:
            n_components (int): Target dimensions for UMAP.
            n_neighbors (int): Number of neighbors for UMAP.
            min_cluster_size (int): Minimum cluster size for HDBSCAN.
            min_samples (int): Minimum samples for HDBSCAN.
        
        Returns:
            NarrativeMapper: Self, with clustering results stored.
        """
        self.cluster_df = cluster_embeddings(
            self.embeddings,
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        return self

    def summarize(self, max_sample_size: int=500) -> "NarrativeMapper":
        """
        Summarizes each cluster using GPT-based keyword extraction and sentiment analysis.

        Parameters:
            max_sample_size (int): max length of text list for each cluster being sampled
        
        Returns:
            NarrativeMapper: Self, with summarized clusters stored.
        """
        self.summary_df = summarize_clusters(self.cluster_df, max_sample_size)
        return self

    def format_by_text(self) -> pd.DataFrame:
        """
        Returns a DataFrame where each row represents an individual comment with its sentiment.
        
        Returns:
            pd.DataFrame: Text-level formatted output.
        """
        return format_by_text(self.summary_df, self.online_group_name)

    def format_by_cluster(self) -> pd.DataFrame:
        """
        Returns a DataFrame summarizing clusters with overall sentiment and comment counts.
        
        Returns:
            pd.DataFrame: Cluster-level formatted output.
        """
        return format_by_cluster(self.summary_df, self.online_group_name)

    def format_to_dict(self) -> dict:
        """
        Returns the summarized clusters in a dictionary format.
        
        Returns:
            dict: A dictionary with cluster summaries, suitable for JSON export.
        """
        return format_to_dict(self.summary_df, self.online_group_name)
