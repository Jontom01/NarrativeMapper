# NarrativeMapper

### Overview:

The NarrativeMapper package is a discourse analysis pipeline that uncovers the dominant narratives and emotional tones within online communities.

This project processes textual messages from .csv files, then applies OpenAI’s embedding API (text-embedding-3-large) to convert each message into semantic vectors. These embeddings are clustered using UMAP for dimensionality reduction and HDBSCAN for density-based clustering.

For each discovered cluster, the tool:

- Extracts the main talking points OpenAI Chat Completions

- Analyzes the emotional tone using a Hugging Face sentiment classifier

- Outputs structured summaries of the narrative + emotion pairs

---
### Installation:

Install via [PyPI](https://pypi.org/project/NarrativeMapper/): 

```bash
pip install NarrativeMapper
```

---
### Output Formats:

This example is based off of 1800 r/antiwork comments from the top 300 posts within the last year (Date of Writing: 2025-04-03). 

Output using **format_to_dict()** function. Useful for JSON export.

```python
{
    "online_group_name": "r/antiwork",
    "clusters": [
        {
            "cluster": 0,
            "cluster_summary": "The core theme of this cluster revolves around the frustrations and challenges of the modern job application and interview process, highlighting issues such as discrimination, exploitative practices, and the disconnect between employers and candidates.",
            "tone": "NEGATIVE",
            "comment_count": 76
        },
        {
            "cluster": 1,
            "cluster_summary": "The core theme of this cluster revolves around the debate over low wages in the fast food industry, the impact of wage increases on business practices and pricing, and the broader implications for workers' livelihoods and economic conditions.",
            "tone": "NEGATIVE",
            "comment_count": 100
        },
        {
            "cluster": 2,
            "cluster_summary": "The cluster reflects widespread frustration and despair among younger generations regarding economic instability, unaffordable living costs, inadequate healthcare, and the perceived indifference of older generations towards their struggles.",
            "tone": "NEGATIVE",
            "comment_count": 112
        },
        {
            "cluster": 3,
            "cluster_summary": "The core theme of this cluster revolves around employee dissatisfaction with management practices, workplace exploitation, and the importance of asserting one's rights and boundaries in a toxic work environment.",
            "tone": "NEGATIVE",
            "comment_count": 464
        },
        {
            "cluster": 4,
            "cluster_summary": "The core theme of this cluster revolves around dissatisfaction with traditional work structures, advocating for reduced work hours, better work-life balance, and criticism of corporate exploitation and the lack of employee rights.",
            "tone": "NEGATIVE",
            "comment_count": 95
        },
        {
            "cluster": 5,
            "cluster_summary": "The core theme of this cluster revolves around wealth inequality, criticizing the hoarding of wealth by billionaires and the systemic issues that perpetuate economic disparity and exploitation of the working class.",
            "tone": "NEGATIVE",
            "comment_count": 95
        },
        {
            "cluster": 6,
            "cluster_summary": "The comments express strong criticism of capitalism, highlighting themes of exploitation, corporate greed, and the detrimental impact of billionaires and CEOs on workers and society.",
            "tone": "NEGATIVE",
            "comment_count": 89
        }
    ]
}
```

Two other formatting functions are available, format_by_text() and format_by_cluster(), both return pandas DataFrames that are well-suited for CSV export.

**format_by_cluster()** returns columns:

- online_group_name - online group name

- cluster - numeric cluster number

- cluster_summary - summary of the cluster

- comment_count - sampled textual messages per cluster

- aggregated_sentiment - net sentiment, of form 'NEGATIVE', 'POSITIVE', 'NEUTRAL'

- all_sentiments - this is a list containing dict items of the form '{'label': 'NEGATIVE', 'score': 0.9896971583366394}' for each message (sentiment calculated by distilbert-base-uncased-finetuned-sst-2-english).

example to showcase output format:

[click to view CSV](unrelated_to_package/example_outputs/test_2.csv)

**format_by_text()** returns columns:

- online_group_name - online group name

- cluster - numeric cluster number

- cluster_summary - summary of the cluster

- text - the sampled textual message (this function returns all of them row by row)

- sentiment - dict item holding sentiment calculation, of the form '{'label': 'NEGATIVE', 'score': 0.9896971583366394}' (sentiment calculated by distilbert-base-uncased-finetuned-sst-2-english).

example output to showcase output format:

[click to view CSV](unrelated_to_package/example_outputs/test_1.csv)

---
### Pipeline Architecture:

----------------------------------------------------------------------------------------------------------------------------

CSV Text Data --> Embeddings (embeddings.py) --> Cluster (clustering.py) --> Summarize (summarize.py)  --> Formatting (formatters.py)

----------------------------------------------------------------------------------------------------------------------------

**embeddings.py:**
Converts textual messages into 3072 dimensional vectors (OPEN AI's text-embedding-3-large).

**clustering.py:**
Clusters embedding vectors using UMAP for reduction and HDBSCAN for clustering.

**summarize.py:**
Determines summaries/label-names (4o-gpt-mini Chat Completion) and sentiment (distilbert-base-uncased-finetuned-sst-2-english) for each cluster. 

**formatters.py:**
Formats summarized clusters into useful forms for data analysis.

---
### How to Use:

**IMPORTANT:**

To use this package, you'll need an OpenAI API key. Create a .env file in your root directory (same folder where your script runs).

Inside the .env file, add your API key like this:

```dotenv
OPENAI_API_KEY="your-api-key-here"
```

The package will automatically load your key using python-dotenv. (Make sure to keep your .env file private and add it to your .gitignore if you're using Git.)


**Option 1: High-Level Class-Based Interface**

```python
#initialize NarrativeMapper object
mapper = NarrativeMapper("r/antiwork")

#embeds semantic vectors
mapper.load_embeddings("path/to/your/file.csv", chunk_size=1000)

#clustering: n_components, n_neighbors are UMAP variables. min_cluser_size, min_samples are HDBSCAN variables.
mapper.cluster(n_components=20, n_neighbors=20, min_cluster_size=40, min_samples=15)

#summarize each cluster's topic and sentiment
mapper.summarize(max_sample_size=500)

#export in your preferred format
summary_dict = mapper.format_to_dict()
text_df = mapper.format_by_text()
cluster_df = mapper.format_by_cluster()

#saving DataFrames to csv
text_df.to_csv("comments_by_cluster.csv", index=False)
cluster_df.to_csv("cluster_summary.csv", index=False)
```

**Option 2: Low-Level Functional Interface**

```python
#manual control over each step:
embeddings = get_embeddings("path/to/your/file.csv")
cluster_df = cluster_embeddings(embeddings, n_components=20, n_neighbors=20, min_cluster_size=40, min_samples=15)
summary_df = summarize_clusters(cluster_df)

#export/format options
summary_dict = format_to_dict(summary_df, online_group_name="r/antiwork")
text_df = format_by_text(summary_df, online_group_name="r/antiwork")
cluster_df = format_by_cluster(summary_df, online_group_name="r/antiwork")
```