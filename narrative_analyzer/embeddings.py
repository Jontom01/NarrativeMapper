# features/feature_extraction.py
from openai import OpenAI
import ast
import pandas as pd
import json
from dotenv import load_dotenv
import os
import csv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chunk_list(big_list, chunk_size=1000):
    return [big_list[i:i + chunk_size] for i in range(0, len(big_list), chunk_size)]

def batch_embeddings(file_name: str) -> list[dict]:
    comments = []
    with open(file_name, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        comments = list(reader)

    comment_list_2d = chunk_list(comments) #this chunks the list of comment dicts into 1000 by N 2D list. This lets us batch call Open AI embeddings
    embedding_dict = [] #this will hold a list of all the comment dict items after they have added 'embedded_vector'
    for comment_list in comment_list_2d:
        embedding_dict_tmp = get_embeddings(comment_list) #the comment list with embeddings for just 1 row of the 2D 1000 by N list
        embedding_dict += embedding_dict_tmp #add these dict items to the total embedding dict list
    return embedding_dict

def get_embeddings(d: list[dict]) -> list[dict]:
    comments = []
    for item in d:
        comments.append(item['text'])
    response = client.embeddings.create(
        input=comments,
        model="text-embedding-3-large"
    )
   
    for i in range(len(response.data)):
        d[i]['embedding_vector'] = response.data[i].embedding

    return d