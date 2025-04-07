from narrative_mapper import *

if __name__ == "__main__":
    #Function Use Version
    
    '''
    embeddings_dict = get_embeddings("unrelated_to_package/comment_data/comment_data_antiwork_1800.csv")

    df = cluster_embeddings(embeddings_dict, n_components=20, n_neighbors=20, min_cluster_size=40, min_samples=15)

    summarize_df = summarize_clusters(df)

    fin = format_to_dict(summarize_df, "r/antiwork")

    csv_df = format_by_text(summarize_df, "r/antiwork")
    
    csv2_df = format_by_cluster(summarize_df, "r/antiwork")

    csv_df.to_csv("test_1.csv", index=False)

    csv2_df.to_csv("test_2.csv", index=False)

    print(fin)
    '''
    #OO Version
    
    mapper = NarrativeMapper("r/antiwork")

    mapper.load_embeddings("unrelated_to_package/comment_data/comment_data_antiwork_1800.csv", chunk_size=1000)

    umap_kwargs =  {'min_dist': 0.0}
    mapper.cluster(n_components=15, n_neighbors=15, min_cluster_size=70, min_samples=20, umap_kwargs=umap_kwargs)
    mapper.summarize(max_sample_size=600)
    
    df_text = mapper.format_by_text()
    df_cluster = mapper.format_by_cluster()
    dict_output = mapper.format_to_dict()

    df_text.to_csv("test_1.csv", index=False)
    df_cluster.to_csv("test_2.csv", index=False)

    print(dict_output)