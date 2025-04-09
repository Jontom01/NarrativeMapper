from dotenv import load_dotenv
import os

dotenv_loaded = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

if not dotenv_loaded:
    print("WARNING: .env file not found in current directory.")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please provide it in a .env file.")

from narrative_mapper.narrative_analyzer.narrative_mapper import NarrativeMapper
from rich.logging import RichHandler
from datetime import datetime
import logging
import argparse
import tiktoken
import csv
import pandas as pd

#better cluster param calculations, flag options (sample size limiter, batch_size, output file directory)
def calculate_token_stats(text_list, model="text-embedding-3-large"):
    """
    Calculates average and total tokens for a list of messages.

    Args:
        text_list (List[str]): List of textual messages.
        model (str): Model name to load correct tokenizer.

    Returns:
        dict: {'average_tokens': float, 'total_tokens': int}
    """
    encoding = tiktoken.encoding_for_model(model)

    token_counts = [len(encoding.encode(text)) for text in text_list]
    total_tokens = sum(token_counts)
    average_tokens = total_tokens / len(text_list) if text_list else 0

    return {
        "average_tokens": round(average_tokens, 2),
        "total_tokens": total_tokens
    }

def get_cluster_params(df):
    text_list = df['text'].tolist()

    token_stats = calculate_token_stats(text_list)
    total_tokens = token_stats['total_tokens']
    avg_tokens = token_stats['average_tokens']

    base = {'n_components': 10, 'n_neighbors': 20, 'min_cluster_size': 50, 'min_samples': 10}
    if total_tokens < 25000: #~65000 for 1000 reddit comments
        return base
    else:
        #scale n_neighbors and min_cluster_size up with total tokens
        scale_factor_1 = max(1, (total_tokens / 25000) * 0.75)
        #scale down min_cluster_size slightly with avg_tokens
        scale_factor_2 = max(0.6, min(1.0, 40 / avg_tokens))  # Cap at 0.6 to 1.0

        return {
            "n_neighbors": int(base["n_neighbors"] * scale_factor_1),
            "n_components": base["n_components"],
            "min_cluster_size": int(base["min_cluster_size"] * scale_factor_1 * scale_factor_2),
            "min_samples": int(base["min_samples"] * scale_factor_2),
        }

def main():
    parser = argparse.ArgumentParser(description="Run NarrativeMapper on this file.")
    parser.add_argument("file_name", type=str, help="file path")
    parser.add_argument("online_group_name", type=str, help="online group name")
    args = parser.parse_args()

    df = pd.read_csv(args.file_name)
    cluster_params = get_cluster_params(df)
    #print(cluster_params)
    mapper = NarrativeMapper(df, args.online_group_name)
    mapper.load_embeddings()
    umap_kwargs =  {'min_dist': 0.0}
    mapper.cluster(
        n_components=cluster_params['n_components'], 
        n_neighbors=cluster_params['n_neighbors'], 
        min_cluster_size=cluster_params['min_cluster_size'], 
        min_samples=cluster_params['min_samples'], 
        umap_kwargs=umap_kwargs
        )
    mapper.summarize(max_sample_size=500)
    output = mapper.format_to_dict()["clusters"]
    mapper.format_by_cluster().to_csv("testing.csv", index=False)

    with open(f"{args.online_group_name}_NarrativeMapper.txt", "w", encoding="utf-8") as f:
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Online Group Name: {args.online_group_name}\n\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",  # You can add timestamps with "%(asctime)s - %(message)s"
        handlers=[
            logging.FileHandler(f"{args.online_group_name}_NarrativeMapper.txt", mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

    for cluster in output:
        summary = cluster["cluster_summary"]
        sentiment = cluster["sentiment"]
        count = cluster["text_count"]

        message = f"Summary: {summary}\nSentiment: {sentiment}\nComments: {count}\n---\n"
        logging.info(message)
