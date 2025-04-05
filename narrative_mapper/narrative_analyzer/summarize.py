import pandas as pd
from transformers import pipeline
import ast
from dotenv import load_dotenv
from openai import OpenAI
import os 

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiments_for_texts(texts) -> (str, list[dict]):
    """
    Analyze sentiment for a list of texts using the Hugging Face sentiment pipeline.
    Returns an overall aggregated sentiment and a list of individual sentiment results.
    """
    sentiments = []
    for text in texts:
        try:
            result = sentiment_analyzer(text, truncation=True)
            #result is typically a list with one dict: [{'label': 'POSITIVE', 'score': 0.99}]
            sentiments.append(result[0])
        except Exception as e:
            #an case of error, mark it as unknown
            sentiments.append({"label": "UNKNOWN", "score": 0})
    #aggregate by majority label: count POSITIVE and NEGATIVE, then decide overall
    pos_count = sum(1 for s in sentiments if s["label"] == "POSITIVE")
    neg_count = sum(1 for s in sentiments if s["label"] == "NEGATIVE")
    count_ratio = pos_count/neg_count
    if count_ratio > 2:
        overall = "POSITIVE"
    elif count_ratio < 0.5:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"
    return overall, sentiments

def extract_keywords_for_cluster(texts) -> str:
    """
    Uses OpenAI Chat Completions to summarize the main theme of a cluster of texts.
    Returns a concise 1-sentence summary string.
    """

    prompt = f"""
        Here are comments/messages from the same topic cluster (after using embeddings to vectorize the text-semantics and then a clustering algorithm to group them):
        ---
        {texts}
        ---
        Please summarize the core topic or themes of this cluster in 1 sentence (brief, no filler words).
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return str(response.choices[0].message.content)

def summarize_clusters(df: pd.DataFrame, max_sample_size: int=500) -> pd.DataFrame:

    """
    Summarizes each text cluster by extracting the narrative and sentiment analysis of each cluster.

    Given a DataFrame of clustered text (as returned by `cluster_embeddings`), this function:
    - Samples up to 500 comments per cluster
    - Uses OpenAI Chat Completions to generate a one-line summary of each cluster's main theme
    - Applies a Hugging Face sentiment model to determine overall cluster sentiment

    Parameters:
        df (pd.DataFrame): DataFrame containing clustered text data with a 'cluster' and 'text' column.
        max_sample_size (int): max length of text list for each cluster being sampled

    Returns:
        pd.DataFrame: A new DataFrame with columns:
            - 'cluster': Cluster ID
            - 'text': List of sampled texts
            - 'cluster_summary': Cluster summary (from GPT)
            - 'aggregated_sentiment': Overall sentiment label
            - 'all_sentiments': List of individual sentiment results per text
    """

    #group texts by cluster and sample up to 500 texts per cluster
    grouped_texts = {}
    grouped = df.groupby('cluster')
    for cluster, group in grouped:
        sample_size = min(max_sample_size, len(group))
        grouped_texts[cluster] = group['text'].sample(n=sample_size, random_state=42).tolist()
    

    grouped_df = pd.DataFrame(list(grouped_texts.items()), columns=['cluster', 'text'])
    
    #use OpenAI Chat Completions to extract a concise summary (cluster label) for each cluster
    cluster_summary = []
    for texts in grouped_df['text']:
        summary = extract_keywords_for_cluster(texts)
        cluster_summary.append(summary)
    grouped_df['cluster_summary'] = cluster_summary
    
    #analyze sentiments for each cluster
    aggregated_sentiments = []
    all_sentiments = []
    for texts in grouped_df['text']:
        overall, sentiments = analyze_sentiments_for_texts(texts)
        aggregated_sentiments.append(overall)
        all_sentiments.append(sentiments)
    
    grouped_df['aggregated_sentiment'] = aggregated_sentiments
    grouped_df['all_sentiments'] = all_sentiments
    
    return grouped_df