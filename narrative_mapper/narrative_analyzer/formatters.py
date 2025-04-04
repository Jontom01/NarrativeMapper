import pandas as pd
import ast

def format_by_cluster(df, online_group_name=""):
    #This can eventually be remade using strictly dataframe manipulation, to be faster on larger datasets
    rows = []
    for _, row in df.iterrows():
        talking_points = row["main_talking_points"]
        tmp_dict = {
            'online_group_name': online_group_name, 
            'cluster_label': talking_points,
            'comment_count': len(row['text']), 
            'aggregated_sentiment': row["aggregated_sentiment"], 
            'all_sentiments': row["all_sentiments"]
            }
        rows.append(tmp_dict)

    formatted_df = pd.DataFrame(rows)
    return formatted_df

def format_by_text(df, online_group_name=""):
    #This can eventually be remade using strictly dataframe manipulation, to be faster on larger datasets
    rows = []
    for _, row in df.iterrows():

        talking_points = row["main_talking_points"]

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

        for index, message in enumerate(text_list):
            tmp_dict = {
                'online_group_name': online_group_name, 
                'cluster_label': talking_points,
                'text': message, 
                'sentiment': sentiment_list[index], 
                }
            rows.append(tmp_dict)

    formatted_df = pd.DataFrame(rows)
    return formatted_df


def format_to_dict(df, online_group_name=""):
    final = {"online_group_name": online_group_name, "clusters": []}

    for _, row in df.iterrows():
        talking_points = row["main_talking_points"]
        tone = row["aggregated_sentiment"]
        comment_count = len(row['text'])
        final["clusters"].append({"label": talking_points, "tone": tone, "comment_count": comment_count})

    return final
