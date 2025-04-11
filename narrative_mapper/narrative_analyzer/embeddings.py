from openai import OpenAI, OpenAIError
from .utils import get_openai_key, batch_list
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import pandas as pd
import re
import sys

def clean_texts(text_list: list[str]):
    #can eventually make this more robust
    return [
        re.sub(r'@\w+', '', re.sub(r'<.*?>', '', re.sub(r'https?://\S+', '', text.strip())))
        for text in text_list
    ]

def get_embeddings(df, verbose=False) -> pd.DataFrame:
    """
    Generates OpenAI text embeddings.

    The input DataFrame must contain 'text' column. The function sends
    each 'text' value to the OpenAI embedding API in batches and then adds a new 'embeddings' 
    column to output DataFrame containing the 1536-dimensional semantic embedding.

    Parameters:
        DataFrame: Must include 'text' column
        verbose (bool): Shows progress bar and timer if True.

    Returns:
        DataFrame: contains origin columns in file_name, but with the added 'embeddings' column
    """
    if 'text' not in df.columns:
        print("Error: DataFrame must contain a 'text' column.")
        sys.exit(1)
    try:
        client = OpenAI(api_key=get_openai_key())
        df = df.copy()
        text_list = df['text'].tolist()

        if not text_list:
            print("Error: 'text' column is empty after removing nulls.")
            sys.exit(1)

        embeddings_list = []
        batches = batch_list(text_list, model="text-embedding-3-small", max_tokens=8000) #used to send multiple requests to bypass token limit. This works because the vector space is the same each call.

        progress_context = (
        Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
            ) if verbose else nullcontext()
        )
        with progress_context as progress:
            if verbose:
                task = progress.add_task("[cyan]Embedding texts...", total=len(text_list))
            for batch in batches:
                batch = clean_texts(batch) #clean text input
                response = client.embeddings.create(
                    input=batch,
                    model="text-embedding-3-small"
                )
                for item in response.data:
                    embeddings_list.append(item.embedding)
                if verbose:
                    progress.update(task, advance=len(batch))
        
        df['embeddings'] = embeddings_list
        return df

    except OpenAIError as e:
        print(f"OpenAI API error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error during embedding generation: {e}")
        sys.exit(1)