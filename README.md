# NarrativeMapper

**Overview:**

NarrativeMapper is a discourse analysis pipeline that uncovers the dominant narratives and emotional tones within online communities.

This project uses the Reddit API to scrape user comments from top posts in specific subreddits, then applies OpenAI’s embedding API (text-embedding-3-large) to convert text into semantic vectors. These embeddings are clustered using UMAP for dimensionality reduction and HDBSCAN for density-based clustering.

For each discovered cluster, the tool:

- Extracts the main talking points OpenAI Chat Completions

- Analyzes the emotional tone using a Hugging Face sentiment classifier

- Outputs structured summaries of the narrative + emotion pairs


**Example Output:**

This example is based off of 1800 r/conservatives comments from the top 300 posts within the last year (Date of Writing: 2025-04-03). Output using summarize.format_to_dict() method

<details>
<summary>click to view output example</summary>

```python
{
  "online_group_name": "r/conservative",
  "clusters": [
    {
      "label": "The comments in this cluster primarily discuss economic challenges, including inflation, government spending, and the impact of political policies on personal finances and the middle class.",
      "tone": "NEGATIVE",
      "comment_count": 49
    },
    {
      "label": "The core theme of this comment cluster revolves around concerns and criticisms regarding election integrity, voter ID laws, and the political strategies of the Republican Party in the context of upcoming elections.",
      "tone": "NEGATIVE",
      "comment_count": 56
    },
    {
      "label": "The core themes of this cluster revolve around criticisms of leftist ideologies, particularly regarding race, gender identity, and social policies, highlighting perceived hypocrisy, the impact of these ideologies on society, and concerns about the implications for children and traditional values.",
      "tone": "NEGATIVE",
      "comment_count": 102
    },
    {
      "label": "The core themes of this comment cluster revolve around immigration, national identity, cultural integration, and perceptions of societal decline attributed to immigration policies and practices.",
      "tone": "NEUTRAL",
      "comment_count": 49
    },
    {
      "label": "The comments in this cluster revolve around polarized perspectives on the Israel-Palestine conflict, encompassing issues of free speech, cultural values, religious intolerance, and the perceived hypocrisy of various political ideologies.",
      "tone": "NEGATIVE",
      "comment_count": 101
    },
    {
      "label": "The comments in this cluster primarily discuss political opinions and reactions surrounding recent events involving Donald Trump, Kamala Harris, and the media's portrayal of them, highlighting themes of political bias, public perception, and the implications of an assassination attempt on Trump.",
      "tone": "NEGATIVE",
      "comment_count": 375
    },
    {
      "label": "The core theme of this comment cluster revolves around perceived bias and manipulation on Reddit, particularly regarding political discussions, with users expressing frustration over censorship, propaganda, and the dominance of anti-Trump sentiment across various subreddits.",
      "tone": "NEGATIVE",
      "comment_count": 55
    }
  ]
}

```

</details>

Other formatting methods are available. Both summarize.format_by_text() and summarize.format_by_cluster() return dataframes that offer better format for data analysis. 

**Architecture:**

----------------------------------------------------------------------------------------------------------------------------

Text Data (reddit_scraper.py) ----> CSV Text Data

CSV Text Data ----> Embeddings (embeddings.py) ----> Cluster (clustering.py) ----> Summarize (summarize.py)

----------------------------------------------------------------------------------------------------------------------------

*reddit_scraper.py:*
Used to scrape reddit comments (Reddit PRAW). This data is then turned into a csv file.

*embeddings.py:*
Converts comments into 3072 dimensional vectors (OPEN AI's text-embedding-3-large).

*clustering.py:*
Clusters embedding vectors using UMAP + HDBSCAN. I reduced the dimensions to 50 during reduction (after some attempts and fails). The other variables related to dimension reduction and clustering are dependent on the ammount of vectors (comments).

*summarize.py:*
Extracts cluster summaries (4o-gpt-mini Chat Completion) and semantics (distilbert-base-uncased-finetuned-sst-2-english) for each cluster. I assign cluster label names based off the following OpenAI Chat Completions prompt. 

<details>
<summary>click to view prompt</summary>

```python
prompt = f"""
        Here are comments/messages from the same topic cluster (after using embeddings to vectorize the text-semantics and then a clustering algorithm to group them):
        ---
        {texts}
        ---
        Please summarize the core topic or themes of this cluster in 1 sentence (brief, no filler words).
        """
```

</details>

Using the sentiment classifier on each comment, I determine a cluster to be "POSITIVE" if there are 2 times more (or greater) positive comments than negative, and vice-versa for clusters that are determined to be "NEGATIVE".
