from keybert import KeyBERT
import pandas as pd
from transformers import pipeline


kw_model = KeyBERT()
sentiment_analyzer = pipeline("sentiment-analysis")

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

def extract_keywords_for_cluster(texts, top_n=10):
    joined = " ".join(texts)
    keywords = kw_model.extract_keywords(
        joined,
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=top_n
    )
    result = []
    for kw, _ in keywords:
        result.append(kw)
    return result

def summarize_clusters(df):
    #firstly for keyword extraction
    #sample texts in cluster
    sampled_list = []
    grouped_obj = df.groupby('cluster')
    for cluster, group in grouped_obj:
        sample_size = min(500, len(group))
        sampled_group = group.sample(n=sample_size, random_state=42)
        sampled_list.append(sampled_group)
    sampled_df = pd.concat(sampled_list).reset_index(drop=True)

    #group sample text by cluster
    grouped_texts = {}
    grouped_obj2 = sampled_df.groupby('cluster')
    for cluster, group in grouped_obj2:
        text_list = group['text'].tolist()
        grouped_texts[cluster] = text_list
    grouped_df = pd.DataFrame(list(grouped_texts.items()), columns=['cluster', 'text'])

    #apply keyword extraction for each cluster
    main_talking_points = []
    for texts in grouped_df['text']:
        keywords = extract_keywords_for_cluster(texts, top_n=10)
        main_talking_points.append(keywords)
    grouped_df['main_talking_points'] = main_talking_points


    #now to analyze sentiment for each cluster
    aggregated_sentiments = []
    all_sentiments = []
    for texts in grouped_df['text']:
        overall, sentiments = analyze_sentiments_for_texts(texts)
        aggregated_sentiments.append(overall)
        all_sentiments.append(sentiments)

    grouped_df['aggregated_sentiment'] = aggregated_sentiments
    grouped_df['all_sentiments'] = all_sentiments

    print(grouped_df[['cluster', 'main_talking_points', 'aggregated_sentiment']])
   # final_df = grouped_df.drop(columns=['text'])
    return grouped_df
    #final_df.to_csv("cluster_keywords_and_sentiments.csv", index=False)

def summarizedf_to_dict(df, subreddit):
    final = {"subreddit": subreddit, "clusters": []}

    for _, row in df.iterrows():
        talking_points = row["main_talking_points"]
        #Convert to list only if it's a string representation
        if isinstance(talking_points, str):
            talking_points = ast.literal_eval(talking_points)
        if isinstance(talking_points, list) and len(talking_points) > 0:
            label = talking_points[:3] #top 3 main talking points
        else: 
            label = []
        tone = row["aggregated_sentiment"]
        comment_count = len(row['text'])
        final["clusters"].append({"label": label, "tone": tone, "comment_count": comment_count})

    return final
