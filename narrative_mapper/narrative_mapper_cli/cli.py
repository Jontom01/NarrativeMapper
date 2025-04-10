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
from math import sqrt, log2
import logging
import argparse
import csv
import pandas as pd

#better cluster param calculations, flag options (sample size limiter, batch_size, output file directory)
def get_cluster_params(df, verbose=False):
    text_list = df['text'].tolist()
    num_texts = len(text_list)
    base_num_texts = 500
    N = max(1, num_texts / base_num_texts)

    #n_components ~ constant to N. 
    n_components = 10

    #n_neighbors ~ sqrt(N). range [10, 60]
    n_neighbors = int(min(60, max(10, 10*sqrt(N))))

    #min_cluster_size ~ sqrt(N). range [15, 200]
    min_cluster_size = int(min(200, max(15, 15*sqrt(N))))

    #min_samples ~ log2(N). range [5, 20]
    min_samples = int(min(20, max(5, 5*log2(N))))

    params = {
        "n_neighbors": n_neighbors,
        "n_components": n_components,
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
    }

    if verbose:
        print(f"[PARAM SCALING]")
        print(f"Text count: {num_texts}")
        print(f"n_components: {params['n_components']}")
        print(f"n_neighbors: {params['n_neighbors']}")
        print(f"min_cluster_size: {params['min_cluster_size']}")
        print(f"min_samples: {params['min_samples']}")

    return params

def main():
    parser = argparse.ArgumentParser(description="Run NarrativeMapper on this file.")
    parser.add_argument("file_name", type=str, help="file path")
    parser.add_argument("online_group_name", type=str, help="online group name")

    #FLAGS
    # gotta add one for file or no file
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
    verbose = args.verbose or False

    cluster_params = get_cluster_params(df, verbose)

    mapper = NarrativeMapper(df, args.online_group_name, verbose)
    mapper.load_embeddings()
    umap_kwargs =  {'min_dist': 0.0, 'low_memory': True}
    mapper.cluster(
        n_components=cluster_params['n_components'], 
        n_neighbors=cluster_params['n_neighbors'], 
        min_cluster_size=cluster_params['min_cluster_size'], 
        min_samples=cluster_params['min_samples'], 
        umap_kwargs=umap_kwargs
        )
    mapper.summarize(max_sample_size=500)
    output = mapper.format_to_dict()["clusters"]
   # mapper.format_by_cluster().to_csv("testing.csv", index=False)

    with open(f"{args.online_group_name}_NarrativeMapper.txt", "w", encoding="utf-8") as f:
        f.write(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Online Group Name: {args.online_group_name}\n\n")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
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
