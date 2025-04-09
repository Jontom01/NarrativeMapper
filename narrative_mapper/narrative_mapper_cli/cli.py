from dotenv import load_dotenv
import os

dotenv_loaded = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

if not dotenv_loaded:
    print("WARNING: .env file not found in current directory.")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please provide it in a .env file.")

from narrative_mapper.narrative_analyzer.narrative_mapper import NarrativeMapper
from .cluster_config import BASELINE_MODES
from rich.logging import RichHandler
from datetime import datetime
import logging
import argparse
import tiktoken
import csv
import pandas as pd
import math

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
        "total_tokens": total_tokens,
        "num_texts": len(text_list)
    }

def get_cluster_params(df, mode, verbose):
    text_list = df['text'].tolist()
    num_texts = len(text_list)
    token_stats = calculate_token_stats(text_list)

    total_tokens = token_stats['total_tokens']
    avg_tokens = token_stats['average_tokens']

    baseline = BASELINE_MODES.get(mode, BASELINE_MODES["standard"])

    return baseline.scale_params(
        total_tokens=total_tokens,
        avg_tokens=avg_tokens,
        num_texts=num_texts,
        verbose=verbose
    )

def main():
    parser = argparse.ArgumentParser(description="Run NarrativeMapper on this file.")
    parser.add_argument("file_name", type=str, help="file path")
    parser.add_argument("online_group_name", type=str, help="online group name")

    #FLAGS
    parser.add_argument(
        "--cluster-mode",
        type=str,
        choices=["standard", "long_form", "short_form"],
        default="standard",
        help="Choose a clustering mode (default: standard)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed parameter scaling info"
    )
    '''
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview scaled parameters without running clustering"
    )
    '''
    args = parser.parse_args()

    df = pd.read_csv(args.file_name)

    mode = args.cluster_mode or "standard"
    verbose = args.verbose or False

    cluster_params = get_cluster_params(df, mode, verbose)

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
