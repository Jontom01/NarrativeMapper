import pandas as pd

def format_by_cluster(df, online_group_name="") -> pd.DataFrame:
    """
    Formats the summarized cluster output into a compact DataFrame where each row represents a cluster.

    Includes the cluster label, sentiment info, and total comment count for each cluster.

    Parameters:
        df (pd.DataFrame): Output from summarize_clusters().
        online_group_name (str): Label identifying the source community (e.g. subreddit name).

    Returns:
        pd.DataFrame: Cluster-level summary with one row per cluster.
    """
    df = df.copy()
    text_count = []
    for _, row in df.iterrows():
        text_count.append(len(row['text']))
    df['text_count'] = text_count   
    df['online_group_name'] = online_group_name
    df = df[['online_group_name', 'cluster', 'cluster_summary', 'text_count', 'aggregated_sentiment', 'text', 'all_sentiments']]
    
    return df

def format_by_text(df, online_group_name="") -> pd.DataFrame:
    """
    Flattens the summarized cluster output into a DataFrame where each row is an individual comment.

    Includes the comment text, its cluster label, and associated sentiment.

    Parameters:
        df (pd.DataFrame): Output from summarize_clusters().
        online_group_name (str): Label identifying the source community.

    Returns:
        pd.DataFrame: Text-level DataFrame with one row per message.
    """

    #This can eventually be remade using strictly dataframe manipulation, to be faster on larger datasets
    text_col = []
    sentiment_col = []
    online_group_name_col = []
    cluster_col = []
    cluster_summary_col = []

    for _, row in df.iterrows():
        texts = row['text']

        sentiments = row['all_sentiments']

        text_col += texts
        sentiment_col += sentiments

        tmp = len(texts) #since these columns need to match the length of the others
        online_group_name_col += [online_group_name] * tmp
        cluster_col += [row['cluster']] * tmp
        cluster_summary_col += [row['cluster_summary']] * tmp

    return_df = pd.DataFrame({
        'online_group_name': online_group_name_col, 
        'cluster': cluster_col, 
        'cluster_summary': cluster_summary_col,
        'text': text_col,
        'sentiment': sentiment_col})

    return return_df

def format_to_dict(df, online_group_name="") -> dict:
    """
    Converts the summarized cluster output into a dictionary format useful for JSON export.

    Each cluster includes its label, sentiment, and comment count.

    Parameters:
        df (pd.DataFrame): Output from summarize_clusters().
        online_group_name (str): Label identifying the source community.

    Returns:
        dict: A structured dictionary with cluster summaries.
    """
    df = df.copy()
    final = {"online_group_name": online_group_name, "clusters": []}

    for _, row in df.iterrows():
        cluster_summary = row["cluster_summary"]
        sentiment = row["aggregated_sentiment"]
        text_count = len(row['text'])
        cluster = row["cluster"]
        final["clusters"].append({"cluster": cluster, "cluster_summary": cluster_summary, "sentiment": sentiment, "text_count": text_count})

    return final
