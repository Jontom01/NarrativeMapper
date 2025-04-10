from transformers import pipeline
from openai import OpenAI
from .utils import get_openai_key, batch_list
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import pandas as pd
import torch

device= 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)


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

    if neg_count == 0 and pos_count == 0: raise Exception("No sentiments calculated in batch")
    count_ratio = 2 if (neg_count == 0) else pos_count/neg_count
    
    if count_ratio >= 2:
        overall = "POSITIVE"
    elif count_ratio <= 0.5:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"
    return overall, sentiments

def extract_summary_for_cluster(texts: list[str]) -> str:
    """
    Summarizes a cluster of semantically similar texts into one precise sentence.
    Uses a two-stage summarization strategy to handle token limits and improve accuracy.
    """
    client = OpenAI(api_key=get_openai_key())
    
    summary_batches = []
    batches = batch_list(texts, model="gpt-4o-mini", max_tokens=7000)
    
    for batch in batches:
        joined_batch = "\n".join(batch)

        prompt = f"""
        You are an expert in discourse analysis and topic summarization.

        Your task is to analyze the following user-generated messages, which were grouped together by semantic similarity using embeddings and clustering.

        Summarize the *recurring themes or central topic(s)* discussed in this cluster using **one short sentence**.

        Be specific. Avoid vague or generic summaries. Use concrete nouns. If multiple recurring themes are present, combine them concisely. Avoid filler words.

        ---
        {joined_batch}
        ---
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        summary = response.choices[0].message.content.strip()
        summary_batches.append(summary)

    combined_summaries = "\n".join(summary_batches)

    final_prompt = f"""
    You are an expert in summarization.

    Here are partial summaries of different batches from a single conversation cluster:
    ---
    {combined_summaries}
    ---
    Synthesize them into **one precise sentence** summarizing the main topic(s).
    Avoid redundancy and avoid vague language. Be specific.
    """

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.2
    )

    return final_response.choices[0].message.content.strip()


def summarize_clusters(df: pd.DataFrame, max_sample_size: int=500, verbose=False) -> pd.DataFrame:
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
    df = df.copy()
    df = df.drop(columns=['embeddings'])
    #group texts by cluster and sample up to 500 texts per cluster
    grouped_texts = {}
    grouped = df.groupby('cluster')
    for cluster, group in grouped:
        sample_size = min(max_sample_size, len(group))
        grouped_texts[cluster] = group['text'].sample(n=sample_size, random_state=42).tolist()
    

    grouped_df = pd.DataFrame(list(grouped_texts.items()), columns=['cluster', 'text'])
    
    #use OpenAI Chat Completions to extract a concise summary (cluster label) for each cluster
    cluster_summary = []
    
    progress_context_summary = (
    Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
        ) if verbose else nullcontext()
    )
    with progress_context_summary as progress:
        if verbose:
            task = progress.add_task("[cyan]Extracting summaries...", total=len(grouped_df['text']))
            
        for texts in grouped_df['text']:
            summary = extract_summary_for_cluster(texts)
            cluster_summary.append(summary)
            if verbose:
                progress.update(task, advance=1)

    grouped_df['cluster_summary'] = cluster_summary
    
    #analyze sentiments for each cluster
    aggregated_sentiments = []
    all_sentiments = []

    progress_context_sentiment = (
    Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
        ) if verbose else nullcontext()
    )
    with progress_context_sentiment as progress:
        if verbose:
            task = progress.add_task("[cyan]Extracting sentiments...", total=len(grouped_df['text']))

        for texts in grouped_df['text']:
            overall, sentiments = analyze_sentiments_for_texts(texts)
            aggregated_sentiments.append(overall)
            all_sentiments.append(sentiments)

            if verbose:
                progress.update(task, advance=1)
    
    grouped_df['aggregated_sentiment'] = aggregated_sentiments
    grouped_df['all_sentiments'] = all_sentiments
    
    return grouped_df