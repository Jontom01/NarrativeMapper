import ast
from narrative_mapper import *

#TEST USE

if __name__ == "__main__":

    embeddings_dict = get_embeddings("unrelated_to_package/comment_data/comment_data_antiwork_1800.csv")

    df = cluster_embeddings(embeddings_dict, n_components=20, n_neighbors=20, min_cluster_size=40, min_samples=15)

    summarize_df = summarize_clusters(df)

    fin = format_to_dict(summarize_df, "r/antiwork")

    csv_df = format_by_text(summarize_df, "r/antiwork")
    
    csv2_df = format_by_cluster(summarize_df, "r/antiwork")

    csv_df.to_csv("test_1.csv", index=False)

    csv2_df.to_csv("test_2.csv")

    print(fin)

