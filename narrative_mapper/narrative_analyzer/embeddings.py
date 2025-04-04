from openai import OpenAI
import ast
import pandas as pd
from dotenv import load_dotenv
import os
import csv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_list(big_list, chunk_size=500):
    return [big_list[i:i + chunk_size] for i in range(0, len(big_list), chunk_size)]

def get_embeddings(file_name: str) -> list[dict]:
    """
    Retrieves OpenAI embeddings for a large list of text entries by chunking the input to stay within token limits.

    Each dictionary must contain a 'text' field. The function processes the data in chunks using 
    `get_embeddings_requests` and returns the full list with an added 'embedding_vector' key in each item.

    Parameters:
        file_name (str): Path to .csv file containing the input data.

    Returns:
        list of dict: Same dictionaries with added 'embedding_vector' key.
    """
    text_list = []
    with open(file_name, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        text_list = list(reader)

    text_list_2d = chunk_list(text_list) #this chunks the list of texts dicts into 500 by N 2D list. This lets us batch call Open AI embeddings
    embedding_dict = [] #this will hold a list of all the text dict items after they have added 'embedded_vector'
    for comment_list in text_list_2d:
        embedding_dict_tmp = get_embeddings_requests(comment_list) #the text list with embeddings for just 1 row of the 2D 500 by N list
        embedding_dict += embedding_dict_tmp #add these dict items to the total embedding dict list
    return embedding_dict

def get_embeddings_requests(d: list[dict]) -> list[dict]:
    """
    Generates OpenAI embeddings for a list of dictionary items containing text data.

    Each input dictionary must contain a 'text' key with a string value. The function sends
    each 'text' value to the OpenAI embedding API and appends a new key called 'embedding_vector'
    containing the 3072-dimensional semantic embedding.

    Parameters:
        data (list of dict): A list of dictionaries, each representing a row of data.
                             Each dictionary must include a 'text' field.

    Returns:
        list of dict: The same list of dictionaries, but with each dictionary now including
                      an 'embedding_vector' key containing the embedding as a list of floats.
    """
    text_list = []
    for item in d:
        text_list.append(item['text'])
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-large"
    )
   
    for i in range(len(response.data)):
        d[i]['embedding_vector'] = response.data[i].embedding

    return d