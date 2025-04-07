from openai import OpenAI
import pandas as pd
from dotenv import load_dotenv
import os
from sklearn.preprocessing import normalize
import tiktoken
import re

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def batch_list(big_list, batch_size=500):
    return [big_list[i:i + batch_size] for i in range(0, len(big_list), batch_size)]

def clean_texts(text_list: list[str]):
    for text in text_list:
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
    return text_list

def get_embeddings(df, batch_size: int=50) -> pd.DataFrame:
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
    df = df.copy()
    #make try catch to see if text exists
    try:
        text_list = df['text'].tolist()
    except Exception as e:
        print("Error converting text column to list")
        raise e

    embeddings_list = []
    batches = batch_list(text_list, batch_size) #used to send multiple requests to bypass token limit. This works because the vector space is the same each call.
    encoding = tiktoken.encoding_for_model("text-embedding-3-large")
    for batch in batches:
        batch = clean_texts(batch) #clean text input
        tokens = encoding.encode(str(batch))
        if len(tokens) > 8191 + 2: #the extra 2 is because of the '[]' that wont be passed into the actual model
            print("A batch exceeded token limit. Decrease batch_size parameter value.")

        response = client.embeddings.create(
            input=batch,
            model="text-embedding-3-large"
        )
        for item in response.data:
            embeddings_list.append(item.embedding)

    normalize_embeddings = normalize(embeddings_list, norm='l2') #since both UMAP + HDBSCAN are setup for cosine similarity
    df['embeddings'] = normalize_embeddings.tolist()

    return df