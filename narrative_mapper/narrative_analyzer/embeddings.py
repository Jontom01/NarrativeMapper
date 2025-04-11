from openai import OpenAI, OpenAIError
from .utils import get_openai_key, batch_list, progress_bars
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
    column to output DataFrame containing the 1536-dimensional semantic embedding.

    Parameters:
        DataFrame: Must include 'text' column
        verbose (bool): Shows progress bar and timer if True.

    Returns:
        DataFrame: contains origin columns in file_name, but with the added 'embeddings' column
    """
    if 'text' not in df.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")

    try:
        client = OpenAI(api_key=get_openai_key())
        df = df.copy()
        text_list = df['text'].tolist()

        if not text_list:
            raise RuntimeError("The 'text' column is empty after removing null values.")

        embeddings_list = []
        batches = batch_list(text_list, model="text-embedding-3-small", max_tokens=8000) #used to send multiple requests to bypass token limit. This works because the vector space is the same each call.

        progress_context = progress_bars(verbose, bars=True)
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

    except openai.error.RateLimitError:
        raise RuntimeError("Rate limit exceeded. Please wait or lower request frequency.")

    except openai.error.OpenAIError as e:
        raise RuntimeError(f"OpenAI request failed") from e

    except Exception as e:
        raise RuntimeError(f"Unexpected error during embedding generation") from e