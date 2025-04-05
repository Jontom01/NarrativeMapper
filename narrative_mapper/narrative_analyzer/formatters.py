import pandas as pd
import ast

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
    comment_count = []
    for _, row in df.iterrows():
        comment_count.append(len(row['text']))
    df['comment_count'] = comment_count   
    df['online_group_name'] = online_group_name
    df = df[['online_group_name', 'cluster', 'cluster_summary', 'comment_count', 'aggregated_sentiment', 'text', 'all_sentiments']]
    
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
    df = df.copy()
    rows = []
    for _, row in df.iterrows():

        cluster_summary = row["cluster_summary"]

        text_list = row['text']
        if isinstance(text_list, str):
            try:
                text_list = ast.literal_eval(text_list)
            except Exception as e:
                print("Error evaluating row['text']:", text_list)
                raise e

        sentiment_list = row['all_sentiments']
        if isinstance(sentiment_list, str):
            try:
                sentiment_list = ast.literal_eval(sentiment_list)
            except Exception as e:
                print("Error evaluating row['all_sentiments']:", sentiment_list)
                raise e
        cluster = row['cluster']
        for index, message in enumerate(text_list): #did enumerate so i can index sentiment_list
            tmp_dict = {
                'online_group_name': online_group_name,
                'cluster': cluster, 
                'cluster_summary': cluster_summary,
                'text': message, 
                'sentiment': sentiment_list[index], 
                }
            rows.append(tmp_dict)

    formatted_df = pd.DataFrame(rows)
    return formatted_df

def format_to_dict(df, online_group_name="") -> dict:
    """
    Converts the summarized cluster output into a dictionary format useful for JSON export.

    Each cluster includes its label, sentiment tone, and comment count.

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
        tone = row["aggregated_sentiment"]
        comment_count = len(row['text'])
        cluster = row["cluster"]
        final["clusters"].append({"cluster": cluster, "cluster_summary": cluster_summary, "tone": tone, "comment_count": comment_count})

    return final
