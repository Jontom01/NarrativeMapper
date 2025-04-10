from openai import OpenAI
from .utils import get_openai_key, batch_list
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from contextlib import nullcontext
import pandas as pd
import re

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
    column to output DataFrame containing the 3072-dimensional semantic embedding.

    Parameters:
        DataFrame: Must include 'text' column

    Returns:
        DataFrame: contains origin columns in file_name, but with the added 'embeddings' column
    """
    client = OpenAI(api_key=get_openai_key())
    df = df.copy()
    #make try catch to see if text exists
    try:
        text_list = df['text'].tolist()
    except Exception as e:
        print("Error converting text column to list")
        raise e

    embeddings_list = []
    batches = batch_list(text_list, model="text-embedding-3-large", max_tokens=8000) #used to send multiple requests to bypass token limit. This works because the vector space is the same each call.
    
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
                model="text-embedding-3-large"
            )
            for item in response.data:
                embeddings_list.append(item.embedding)
            if verbose:
                progress.update(task, advance=len(batch))

    df['embeddings'] = embeddings_list

    return df