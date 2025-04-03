import ast
from embeddings import batch_embeddings
from clustering import cluster_embeddings
from summarize import summarize_clusters, format_to_dict, format_by_cluster, format_by_text

#TEST USE

if __name__ == "__main__":

    embeddings_dict = batch_embeddings("comment_data/comment_data_antiwork_1800.csv")

    df = cluster_embeddings(embeddings_dict,n_components_var=20, n_neighbors_var=20, min_cluster_size_var=40)

    summarize_df = summarize_clusters(df)

    fin = format_to_dict(summarize_df, "r/antiwork")

    csv_df = format_by_text(summarize_df, "r/antiwork")
    
    csv2_df = format_by_cluster(summarize_df, "r/antiwork")

    csv_df.to_csv("idkwhattonamethis_1.csv", index=False)

    csv2_df.to_csv("idkwhattonamethis_2.csv")

    print(fin)

