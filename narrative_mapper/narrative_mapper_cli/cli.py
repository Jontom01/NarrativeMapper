from dotenv import load_dotenv
import os

dotenv_loaded = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

if not dotenv_loaded:
    print("WARNING: .env file not found in current directory.")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please provide it in a .env file.")

from narrative_mapper.narrative_analyzer.narrative_mapper import NarrativeMapper
from datetime import datetime
from math import sqrt, log2
import logging
import argparse
import csv
import pandas as pd
import sys

#better cluster param calculations, flag options (sample size limiter, batch_size, output file directory)
def parse_args():
    #INPUT ARGUMENTS
    parser = argparse.ArgumentParser(description="Run NarrativeMapper on this file.")
    parser.add_argument("file_name", type=str, help="file path")
    parser.add_argument("online_group_name", type=str, help="online group name")
    #FLAGS
    parser.add_argument("--verbose", action="store_true", help="Print/show detailed parameter scaling info and progress bars.")
    parser.add_argument("--file-output", action="store_true", help="Output summaries to text file in working directory.")
    parser.add_argument("--max-samples", type=int, default=500, help="Max amount of texts samples from clusters being used in summarization. Default is 500.")
    parser.add_argument("--random-state", type=int, default=None, help="Sets value to UMAP and PCA random state. Default value is None.")
    parser.add_argument("--no-pca", action="store_true", help="Allows user to skip PCA and go straight to UMAP.")
    parser.add_argument("--dim-pca", type=int, default=100, help="Allows user to change PCA dim. Default is 100.")
    return parser.parse_args()


def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'text' not in df.columns:
            raise ValueError("Input file must contain a 'text' column.")
        return df
        
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")

    if 'text' not in df.columns:
        raise ValueError("Input file must contain a 'text' column.")
    return df

def get_param_calcs(df, verbose=False):
    '''
    Autocalculates UMAP and HDBSCAN parameters based off of 'text' col size.
    '''
    text_list = df['text'].tolist()
    num_texts = len(text_list)
    base_num_texts = 500
    N = max(1, num_texts / base_num_texts)

    #n_components ~ constant to N. 
    n_components = 10

    #n_neighbors ~ cube root of N. range [20, 75]
    n_neighbors = int(min(75, max(20, 20*(N**(1/3)))))

    #min_cluster_size ~ sqrt(N). range [30, 200]
    min_cluster_size = int(min(200, max(30, 30*sqrt(N))))

    #min_samples ~ log2(N). range [5, 30]
    min_samples = int(min(30, max(5, 5*log2(N))))

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

def run_mapper(df, group_name, param_calcs, verbose, **mapper_args):
    '''
    Runs NarrativeMapper logic to obtain main narratives/topics and sentiments.
    '''
    mapper = NarrativeMapper(df, group_name, verbose)
    mapper.load_embeddings()

    pca_kwargs = {
        'n_components': mapper_args['dim_pca'], 
        'random_state': mapper_args['random_state']
        }
    umap_kwargs = {
        'n_components': param_calcs['n_components'],
        'n_neighbors': param_calcs['n_neighbors'],
        'random_state': mapper_args['random_state'],
        "min_dist": 0.0, 
        "low_memory": True
        }
    hdbscan_kwargs = {
        'min_cluster_size': param_calcs['min_cluster_size'],
        'min_samples': param_calcs['min_samples']
    }
    
    mapper.cluster(
        umap_kwargs=umap_kwargs,
        hdbscan_kwargs=hdbscan_kwargs,
        pca_kwargs=pca_kwargs,
        use_pca= not mapper_args['no_pca'] #since no_pca == True means we dont want PCA
    )
    mapper.summarize(max_sample_size=mapper_args['max_sample_size'])
    return mapper

def write_log(output, group_name, file_output):
    '''
    Output logic. Prints to file if user uses --file-output flag.
    '''
    log_path = f"{group_name}_NarrativeMapper.txt"
    handlers = [logging.StreamHandler()]
    if file_output:
        handlers.append(logging.FileHandler(log_path, mode='w', encoding='utf-8'))

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    logging.info(f"Run Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"Online Group Name: {group_name}\n")

    for cluster in output:
        logging.info(f"Summary: {cluster['cluster_summary']}")
        logging.info(f"Sentiment: {cluster['sentiment']}")
        logging.info(f"Text Samples: {cluster['text_count']}")
        logging.info("---")

def main():
    '''
    Main function that runs pipeline.
    '''
    try:
        args = parse_args()
        df = load_data(args.file_name)
        param_calcs = get_param_calcs(df, verbose=args.verbose)
        mapper_args = {
            'random_state': args.random_state,
            'max_sample_size': args.max_samples,
            'no_pca': args.no_pca,
            'dim_pca': args.dim_pca
            }
        mapper = run_mapper(df, args.online_group_name, param_calcs, verbose=args.verbose, **mapper_args)
        output = mapper.format_to_dict()["clusters"]
        write_log(output, args.online_group_name, args.file_output)

    except Exception as e:
        print(f"Error running CLI: {e}")
        sys.exit(1)