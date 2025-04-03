import ast
from embeddings import batch_embeddings
from clustering import cluster_embeddings
from summarize import summarize_clusters, summarizedf_to_dict

#TEST USE

if __name__ == "__main__":
    embeddings_dict = batch_embeddings("comment_data/comment_data_conservative_1400.csv")

    df = cluster_embeddings(embeddings_dict)

    summarize_df = summarize_clusters(df)

    fin = summarizedf_to_dict(summarize_df)

    print(fin)

