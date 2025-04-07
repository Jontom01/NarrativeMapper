from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
import csv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_list(big_list, chunk_size=500):
    return [big_list[i:i + chunk_size] for i in range(0, len(big_list), chunk_size)]

def get_embeddings(file_name: str, chunk_size: int=500) -> pd.DataFrame:
    """
    Generates OpenAI text embeddings.

    The input csv file must contain 'text' column. The function sends
    each 'text' value to the OpenAI embedding API in batches and then adds a new 'embeddings' 
    column to output DataFrame containing the 3072-dimensional semantic embedding.

    Parameters:
        data (list of dict): A list of dictionaries, each representing a row of data.
                             Each dictionary must include a 'text' field.

    Returns:
        DataFrame: contains origin columns in file_name, but with the added 'embeddings' column
    """
    #make try catch to see if text exists
    df = pd.read_csv(file_name)
    try:
        text_list = df['text'].tolist()
    except Exception as e:
        print("Error converting text column to list")
        raise e

    embeddings_list = []
    batch_list = chunk_list(text_list, chunk_size) #used to send multiple requests to bypass token limit. This works because the vector space is the same each call.

    for batch in batch_list:
        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )
        for item in response.data:
            embeddings_list.append(item.embedding)

    df['embeddings'] = embeddings_list

    return df