import pandas as pd
from transformers import pipeline
import ast
from dotenv import load_dotenv
from openai import OpenAI
import os 

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiments_for_texts(texts):
    """
    Analyze sentiment for a list of texts using the Hugging Face sentiment pipeline.
    Returns an overall aggregated sentiment and a list of individual sentiment results.
    """
    sentiments = []
    for text in texts:
        try:
            result = sentiment_analyzer(text, truncation=True)
            #result is typically a list with one dict: [{'label': 'POSITIVE', 'score': 0.99}]
            sentiments.append(result[0])
        except Exception as e:
            #an case of error, mark it as unknown
            sentiments.append({"label": "UNKNOWN", "score": 0})
    #aggregate by majority label: count POSITIVE and NEGATIVE, then decide overall
    pos_count = sum(1 for s in sentiments if s["label"] == "POSITIVE")
    neg_count = sum(1 for s in sentiments if s["label"] == "NEGATIVE")
    count_ratio = pos_count/neg_count
    if count_ratio > 2:
        overall = "POSITIVE"
    elif count_ratio < 0.5:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"
    return overall, sentiments

def extract_keywords_for_cluster(texts):
    prompt = f"""
        Here are comments/messages from the same topic cluster (after using embeddings to vectorize the text-semantics and then a clustering algorithm to group them):
        ---
        {texts}
        ---
        Please summarize the core topic or themes of this cluster in 1 sentence (brief, no filler words).
        """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.5
    )
    return str(response.choices[0].message.content)

def summarize_clusters(df):
    #group texts by cluster and sample up to 500 texts per cluster
    grouped_texts = {}
    grouped = df.groupby('cluster')
    for cluster, group in grouped:
        sample_size = min(500, len(group))
        grouped_texts[cluster] = group['text'].sample(n=sample_size, random_state=42).tolist()
    

    grouped_df = pd.DataFrame(list(grouped_texts.items()), columns=['cluster', 'text'])
    
    #use OpenAI Chat Completions to extract a concise summary (cluster label) for each cluster
    main_talking_points = []
    for texts in grouped_df['text']:
        summary = extract_keywords_for_cluster(texts)
        main_talking_points.append(summary)
    grouped_df['main_talking_points'] = main_talking_points
    
    #analyze sentiments for each cluster
    aggregated_sentiments = []
    all_sentiments = []
    for texts in grouped_df['text']:
        overall, sentiments = analyze_sentiments_for_texts(texts)
        aggregated_sentiments.append(overall)
        all_sentiments.append(sentiments)
    
    grouped_df['aggregated_sentiment'] = aggregated_sentiments
    grouped_df['all_sentiments'] = all_sentiments
    
    return grouped_df


