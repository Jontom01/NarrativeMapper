from dotenv import load_dotenv
import os

dotenv_loaded = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))

if not dotenv_loaded:
    print("WARNING: .env file not found in current directory.")

openai_key = os.environ.get("OPENAI_API_KEY")
if not openai_key:
    raise RuntimeError("OPENAI_API_KEY not set. Please provide it in a .env file.")

from narrative_mapper.narrative_analyzer.embeddings import get_embeddings
from narrative_mapper.narrative_analyzer.clustering import cluster_embeddings
from narrative_mapper.narrative_analyzer.summarize import summarize_clusters
from narrative_mapper.narrative_analyzer.formatters import format_to_dict
from .cli_utils import scrape_subreddit, create_map
from datetime import datetime
import logging
import argparse
import csv
import pandas as pd
import pickle

def parse_args():
    #INPUT ARGUMENTS
    parser = argparse.ArgumentParser(description="Run NarrativeMapper on this file.")
    parser.add_argument("file_name", type=str, help="file path")
    
    #FLAGS
    parser.add_argument("--verbose", action="store_true", help="Print/show detailed parameter scaling info and progress bars.")
    parser.add_argument("--cache", action="store_true", help="Cache embeddings and summary pkl files to working directory.")
    parser.add_argument("--load-embeddings", action="store_true", help="Use embeddings pkl as file-path. Skips previous parts of the pipeline.")
    parser.add_argument("--load-summary", action="store_true", help="Use summary pkl as file-path. Skips previous parts of the pipeline.")
    parser.add_argument("--file-output", action="store_true", help="Output summaries to text file in working directory.")
    parser.add_argument("--max-samples", type=int, default=500, help="Max amount of texts samples from clusters being used in summarization. Default is 500.")
    parser.add_argument("--random-state", type=int, default=42, help="Changes value to UMAP and PCA random state. Default value is 42.")
    parser.add_argument("--no-pca", action="store_true", help="Allows user to skip PCA and go straight to UMAP.")
    parser.add_argument("--dim-pca", type=int, default=100, help="Allows user to change PCA dim. Default is 100.")
    parser.add_argument("--reddit", action="store_true", help="Full reddit pipeline. Replace file-path with subreddit name.")

    return parser.parse_args()

def load_data(file_path, load_embeddings=False, load_summary=False, is_reddit_scrape=False):
    try:
        if load_embeddings: 
            df = pd.read_pickle(file_path)
            if 'embeddings' not in df.columns:
                raise ValueError("Input file must contain a 'embeddings' column.")

        elif load_summary: 
            df = pd.read_pickle(file_path)
            if not all(col in df.columns for col in ['cluster', 'cluster_summary', 'aggregated_sentiment']):
                raise ValueError("Input file missing one of following cols: 'cluster', 'cluster_summary', 'aggregated_sentiment'.")

        elif is_reddit_scrape:
            df = scrape_subreddit([file_path],submission_limit=250, comment_limit=10)
       
    
        else: 
            df = pd.read_csv(file_path)

        if 'text' not in df.columns:
            raise ValueError("Input file must contain a 'text' column.")
        return df

    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file: {e}")

    return df

def run_mapper(df, group_name, verbose, **mapper_args):
    '''
    Runs NarrativeMapper logic to obtain main narratives/topics and sentiments.
    '''
    if mapper_args['load_summary']: 
        summary_df = df #skip pipeline if user loads summary_df
    else:
        if mapper_args['load_embeddings']: 
            embeddings_df = df #skip embeddings if user loads embeddings df
        else: 
            embeddings_df = get_embeddings(df, verbose=verbose)

            if mapper_args['cache']:
                embeddings_df.to_pickle(f"{group_name}_embeddings.pkl") #cache embeddings df

        pca_kwargs = {
            'n_components': mapper_args['dim_pca'], 
            'random_state': mapper_args['random_state']
            }
        umap_kwargs = {
            'random_state': mapper_args['random_state'],
            'min_dist': 0.0, 
            'low_memory': True,
            'metric': 'euclidean'
            }
        hdbscan_kwargs = {
            'metric': 'euclidean',
            'cluster_selection_method': 'leaf'
        }
        
        cluster_df = cluster_embeddings(
            df=embeddings_df,
            verbose=verbose,
            umap_kwargs=umap_kwargs,
            hdbscan_kwargs=hdbscan_kwargs,
            pca_kwargs=pca_kwargs,
            use_pca= not mapper_args['no_pca'] #since no_pca == True means we dont want PCA
        )
    
        summary_df = summarize_clusters(df=cluster_df, verbose=verbose, max_sample_size=mapper_args['max_sample_size'])

        if mapper_args['cache']:
            summary_df.to_pickle(f"{group_name}_summary.pkl") #cache summary df

    return summary_df

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
        load_embeddings = args.load_embeddings
        load_summary = args.load_summary
        is_reddit_scrape = args.reddit
        mapper_args = {
            'random_state': args.random_state,
            'max_sample_size': args.max_samples,
            'no_pca': args.no_pca,
            'dim_pca': args.dim_pca,
            'cache': args.cache,
            'load_embeddings': load_embeddings,
            'load_summary': load_summary
            }
        online_group_name = os.path.splitext(args.file_name)[0]

        df = load_data(args.file_name, load_embeddings=load_embeddings, load_summary=load_summary, is_reddit_scrape=args.reddit)
        summary_df = run_mapper(df, online_group_name, verbose=args.verbose, **mapper_args)
        output = format_to_dict(summary_df)['clusters']
        write_log(output, online_group_name, args.file_output)

        map_df = pd.DataFrame(output)
        create_map(map_df, online_group_name)

    except Exception as e:
        raise RuntimeError(f"Error running CLI") from e