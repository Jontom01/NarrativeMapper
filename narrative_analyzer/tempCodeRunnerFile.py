    embeddings_dict = batch_embeddings("comment_data_politics_1200.csv")

    df = cluster_embeddings(embeddings_dict)

    keywords_df = get_keywords(df)

    sentiment_df = get_sentiment(keywords_df)