from openai import OpenAI
from sklearn.preprocessing import normalize
from .utils import get_openai_key, batch_list
from rich.progress import Progress
import pandas as pd
import re

def clean_texts(text_list: list[str]):
    for text in text_list:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
    return text_list

def get_embeddings(df) -> pd.DataFrame:
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
    with Progress() as progress:
        task = progress.add_task("[cyan]Embedding texts...", total=len(text_list))
        for batch in batches:
            batch = clean_texts(batch) #clean text input
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-large"
            )
            for item in response.data:
                embeddings_list.append(item.embedding)
            progress.update(task, advance=len(batch))

    normalize_embeddings = normalize(embeddings_list, norm='l2') #since both UMAP + HDBSCAN are setup for cosine similarity
    df['embeddings'] = normalize_embeddings.tolist()

    return df