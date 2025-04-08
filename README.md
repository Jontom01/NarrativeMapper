# NarrativeMapper


## Overview

Whether you're coding in Python or simply running a single command in your terminal, NarrativeMapper gives you instant insight into the dominant stories behind the noise.

Ever wonder what stories are dominating Reddit, Twitter, or any corner of the internet? NarrativeMapper clusters similar online discussions and uses OpenAI’s GPT to summarize the dominant narratives, tone, and sentiment. Built for researchers, journalists, analysts, and anyone trying to make sense of the chaos.

- Extracts dominant narratives from messy text data

- Clusters similar posts using embeddings + UMAP + HDBSCAN

- Summarizes each cluster with GPT

- Analyzes sentiment per narrative

- Plug-and-play pipeline: **CLI**, class-based, or functional

<details>
<summary><strong>Click to view actual models being used</strong></summary>

- Uses OpenAI Embeddings ([OpenAI's text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings))

- Dimensionality reduction ([UMAP](https://umap-learn.readthedocs.io/en/latest/))

- Density-based clustering ([HDBSCAN](https://hdbscan.readthedocs.io/en/latest/))

- Topic summary + sentiment extraction ([OpenAI's Chat Completions API](https://platform.openai.com/docs/guides/gpt), model gpt-4o-mini + [Hugging Face's distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english))
</details>

## Installation and Setup

**Installation:**

<details>
<summary><strong>Click to view installation process</strong></summary>

Install via [PyPI](https://pypi.org/project/NarrativeMapper/): 

```bash
pip install NarrativeMapper
```
</details>

**Setup:**

<details>
<summary><strong>Click to view setup process</strong></summary>

1. Create a .env file in your root directory (same folder where your script runs).

2. Inside the .env file, add your OpenAI API key like this:

```dotenv
OPENAI_API_KEY=your-api-key-here
```

3. Before importing narrative_mapper, make sure to load your .env like this:

```python
from dotenv import load_dotenv
load_dotenv()

from narrative_mapper import *
```

(Make sure to keep your .env file private and add it to your .gitignore if you're using Git.)
</details>

## How to Use

### Option 1: CLI (zero code)

Run NarrativeMapper directly from the terminal:

```bash
narrativemapper path/to/your.csv online_group_name
```
This will:

- Load the CSV

- Automatically embed, cluster, and summarize the comments

- Output a formatted results file in the current directory (output_summary.txt)

- Print the summarized narratives and sentiment to the terminal

File output example from [this dataset](https://github.com/Jontom01/NarrativeMapper/blob/main/unrelated_to_package/comment_data/comment_data_space.csv):

```txt
Run Timestamp: 2025-04-08 19:12:45
Online Group Name: reddit_space_subreddit

Summary: The core themes of this cluster revolve around the awe and nostalgia associated with the Voyager missions, the challenges of long-term space exploration, and imaginative reflections on extraterrestrial life and science fiction.
Sentiment: NEUTRAL
Comments: 25
---

Summary: The core theme of this cluster revolves around discussions of space, galaxies, the universe's vastness, and the implications of astronomical phenomena on our understanding of life and existence.
Sentiment: NEGATIVE
Comments: 95
---

Summary: The cluster revolves around personal experiences and emotions related to witnessing solar eclipses and auroras, highlighting the awe, excitement, and challenges of viewing these celestial events.
Sentiment: NEUTRAL
Comments: 63
---

Summary: The core theme of this cluster revolves around admiration and appreciation for a stunning astronomical photograph, with many comments expressing curiosity about the techniques used to capture it and requests for high-resolution versions for personal use.
Sentiment: POSITIVE
Comments: 48
---

Summary: The cluster primarily discusses concerns and criticisms regarding Boeing's safety record, management practices, and the implications for NASA's reliance on Boeing for crewed space missions, juxtaposed with support for SpaceX's advancements in space travel.
Sentiment: NEGATIVE
Comments: 53
---

Summary: The cluster discusses concerns about space debris, the irresponsibility of space agencies, particularly China, and the need for international cooperation and regulation to address the growing problem of litter in Earth's orbit.
Sentiment: NEGATIVE
Comments: 20
---
```

**Note:** Make sure you're running the CLI from the same directory where your .env file is located (Unless you have set OPENAI_API_KEY globally in your environment).

### Option 2: Class-Based Interface

```python
from dotenv import load_dotenv
load_dotenv()

from narrative_mapper import *
import pandas as pd

file_df = pd.read_csv("file-path")

#initialize NarrativeMapper object
mapper = NarrativeMapper(file_df, "r/antiwork")

#embeds semantic vectors
mapper.load_embeddings(batch_size=100)

#clustering: main UMAP and HDBSCAN variables along with kwargs for more customizability.
umap_kwargs =  {'min_dist': 0.0}
mapper.cluster(n_components=20, n_neighbors=20, min_cluster_size=40, min_samples=15, umap_kwargs=umap_kwargs)

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

### Option 3: Functional Interface

```python
from dotenv import load_dotenv
load_dotenv()

from narrative_mapper import *
import pandas as pd

df = pd.read_csv("file-path")

#manual control over each step:
embeddings = get_embeddings(file_df, batch_size=100)
cluster_df = cluster_embeddings(embeddings, n_components=20, n_neighbors=20, min_cluster_size=40, min_samples=15)
summary_df = summarize_clusters(cluster_df, max_sample_size=500)

#export/format options
summary_dict = format_to_dict(summary_df, online_group_name="r/antiwork")
text_df = format_by_text(summary_df, online_group_name="r/antiwork")
cluster_df = format_by_cluster(summary_df, online_group_name="r/antiwork")
```

## Output Formats

This example is based off of 1800 r/antiwork comments from the top 300 posts within the last year (Date of Writing: 2025-04-03).

The three formatter functions return the following:

**format_to_dict()** returns dict with following format:

<details>
<summary><strong>format_to_dict output example</strong></summary>

```json
{
    "online_group_name": "r/antiwork",
    "clusters": [
        {
            "cluster": 0,
            "cluster_summary": "The core theme of this cluster revolves around the frustrations and challenges of the modern job application and interview process, highlighting issues such as discrimination, exploitative practices, and the disconnect between employers and candidates.",
            "sentiment": "NEGATIVE",
            "text_count": 76
        },
        {
            "cluster": 1,
            "cluster_summary": "The core theme of this cluster revolves around the debate over low wages in the fast food industry, the impact of wage increases on business practices and pricing, and the broader implications for workers' livelihoods and economic conditions.",
            "sentiment": "NEGATIVE",
            "text_count": 100
        },
        {
            "cluster": 2,
            "cluster_summary": "The cluster reflects widespread frustration and despair among younger generations regarding economic instability, unaffordable living costs, inadequate healthcare, and the perceived indifference of older generations towards their struggles.",
            "sentiment": "NEGATIVE",
            "text_count": 112
        },
        {
            "cluster": 3,
            "cluster_summary": "The core theme of this cluster revolves around employee dissatisfaction with management practices, workplace exploitation, and the importance of asserting one's rights and boundaries in a toxic work environment.",
            "sentiment": "NEGATIVE",
            "text_count": 464
        },
        {
            "cluster": 4,
            "cluster_summary": "The core theme of this cluster revolves around dissatisfaction with traditional work structures, advocating for reduced work hours, better work-life balance, and criticism of corporate exploitation and the lack of employee rights.",
            "sentiment": "NEGATIVE",
            "text_count": 95
        },
        {
            "cluster": 5,
            "cluster_summary": "The core theme of this cluster revolves around wealth inequality, criticizing the hoarding of wealth by billionaires and the systemic issues that perpetuate economic disparity and exploitation of the working class.",
            "sentiment": "NEGATIVE",
            "text_count": 95
        },
        {
            "cluster": 6,
            "cluster_summary": "The comments express strong criticism of capitalism, highlighting themes of exploitation, corporate greed, and the detrimental impact of billionaires and CEOs on workers and society.",
            "sentiment": "NEGATIVE",
            "text_count": 89
        }
    ]
} 
```
</details>

**format_by_cluster()** returns pandas DataFrame with columns:

<details>
<summary><strong>format_by_cluster columns</strong></summary>

- **online_group_name:** online group name

- **cluster:** numeric cluster number

- **cluster_summary:** summary of the cluster

- **text_count:** sampled textual messages per cluster

- **aggregated_sentiment:** net sentiment, of form 'NEGATIVE', 'POSITIVE', 'NEUTRAL'

- **text:** the list of textual messages that are part of the cluster

- **all_sentiments:** this is a list containing dict items of the form '{'label': 'NEGATIVE', 'score': 0.9896971583366394}' for each message (sentiment calculated by distilbert-base-uncased-finetuned-sst-2-english).

</details>

[CSV to show output format](https://github.com/Jontom01/NarrativeMapper/blob/main/unrelated_to_package/example_outputs/test_2.csv)

**format_by_text()** returns pandas DataFrame with columns:

<details>
<summary><strong>format_by_text columns</strong></summary>

- **online_group_name**: online group name

- **cluster**: numeric cluster number

- **cluster_summary:** summary of the cluster

- **text:** the sampled textual message (this function returns all of them row by row)

- **sentiment:** dict item holding sentiment calculation, of the form '{'label': 'NEGATIVE', 'score': 0.9896971583366394}' (sentiment calculated by distilbert-base-uncased-finetuned-sst-2-english).

</details>

[CSV to show output format](https://github.com/Jontom01/NarrativeMapper/blob/main/unrelated_to_package/example_outputs/test_1.csv)


## Pipeline Architecture & API Overview

**Pipeline:**

```txt
CSV Text Data → Embeddings → Clustering → Summarization → Formatting
```
**Functions:**

```python

#Converts each message into a 3072-dimensional vector using OpenAI's text-embedding-3-large.
get_embeddings(file_df, batch_size=...)

#Clusters the embeddings using UMAP (for reduction) and HDBSCAN (for density-based clustering).
cluster_embeddings(
    embeddings, 
    n_components=..., 
    n_neighbors=..., 
    min_cluster_size=..., 
    min_samples=..., 
    umap_kwargs=..., 
    hdbscan_kwags=...
    )

#Uses GPT (via Chat Completions) for cluster summaries and Hugging Face for sentiment analysis.
summarize_clusters(clustered_df, max_sample_size=...)

#Returns structured output as a dictionary (ideal for JSON export).
format_to_dict(summary_df)

#Returns a DataFrame where each row summarizes a cluster.
format_by_cluster(summary_df)

#Returns a DataFrame where each row is an individual comment with its sentiment and cluster label.
format_by_text(summary_df)

```
### NarrativeMapper Class

**Instance Attributes:**

```python
class NarrativeMapper:
    def __init__(self, df, online_group_name: str):
        self.file_df               # DataFrame of csv file
        self.online_group_name     # Name of the online community or data source
        self.embeddings_df         # DataFrame after embedding
        self.cluster_df            # DataFrame after clustering
        self.summary_df            # DataFrame after summarization

```

**Methods:**
```python
load_embeddings(batch_size=...)
cluster(
    n_components=..., 
    n_neighbors=..., 
    min_cluster_size=..., 
    min_samples=..., 
    umap_kwargs=..., 
    hdbscan_kwargs=...
    )
summarize(max_sample_size=...)
format_by_text()
format_by_cluster()
format_to_dict()
```

### Parameter Reference

<details>
<summary><strong>Click to expand</strong></summary>

- **n_components:** The number of dimensions UMAP reduces the embedding vectors to. Lower values simplify the data for clustering.

- **n_neighbors:** Influences UMAP’s balance between local and global structure. Higher values emphasize global relationships.

- **min_cluster_size:** In HDBSCAN, the minimum number of points required to form a cluster. Smaller values allow more granular clusters.

- **min_samples:** A density sensitivity parameter in HDBSCAN. Higher values make clustering more conservative.

- **umap_kwargs:** Allows for input of other UMAP parameters.

- **hdbscan_kwags:** Allows for input of other HDBSCAN parameters.

- **batch_size:** Number of messages processed per API request to avoid token limits. Choose smaller values the larger your textual messages are.

- **max_sample_size:** Maximum number of comments sampled per cluster for summarization.

</details>


## Estimated Cost (OpenAI Pricing)

Estimated cost: **$0.13 to $0.28 per 1 million tokens**.

Example: A CSV containing 1,000 Reddit comments costs approximately **$0.01** to process.

<details>
<summary><strong>Click for pricing details</strong></summary>

The OpenAI text-embedding-3-large model costs approximately $0.13 per 1 million input tokens. Determined by the total tokens of your input textual messages.

The Chat Completions model used for summarization (gpt-4o-mini) is $0.15 per 1 million input tokens. The max_sample_size parameter (referenced later) helps reduce costs by limiting how many comments are passed into gpt-4o-mini for each cluster. This can significantly reduce the Chat Completions token usage.

The gpt-4o-mini input prompt (excluding the text) and output summary are both very short (<100 tokens), so their cost contribution is negligible.

</details>