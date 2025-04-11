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
<summary><strong>Click to view actual models and algorithms</strong></summary>

- Uses OpenAI Embeddings: [OpenAI's text-embedding-3-small](https://platform.openai.com/docs/guides/embeddings)

- Preprocessing: L2 Normalization (for Euclidean distance) + [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)(to reduce UMAP memory usage) 

- Dimensionality reduction: [UMAP](https://umap-learn.readthedocs.io/en/latest/)

- Clustering: [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/)

- Cluster Merging: Union-find algorithm

- Topic summary + sentiment extraction: [OpenAI's Chat Completions API](https://platform.openai.com/docs/guides/gpt), model gpt-4o-mini + [Hugging Face's distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
</details>

## Installation and Setup

**Installation:**

<details>
<summary>Click to view installation process</summary>

Install via [PyPI](https://pypi.org/project/NarrativeMapper/): 

```bash
pip install NarrativeMapper
```
</details>

**Setup:**

<details>
<summary>Click to view setup process</summary>

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
narrativemapper path/to/your.csv online_group_name --flag-options
```
This will:

- Load the CSV

- Automatically embed, cluster, and summarize the comments (with pretty progress bars if using --verbose)

- Output a formatted results file in the current directory (output_summary.txt)

- Print the summarized narratives and sentiment to the terminal

**Output example** from [this dataset](https://github.com/Jontom01/NarrativeMapper/blob/main/sample_data/comment_data/comment_data_space.csv):

```txt
Run Timestamp: 2025-04-10 20:42:45
Online Group Name: reddit_space_subreddit

Summary: The cluster addresses concerns regarding the reliability of SpaceX and Boeing in space missions, the implications of space debris on safety, and the need for corporate accountability in aerospace within the context of human exploration and technological advancement in space.
Sentiment: NEGATIVE
Text Samples: 244
---
Summary: The cluster focuses on personal experiences and emotions tied to witnessing solar eclipses, encompassing travel efforts, photography techniques, and the profound awe these celestial events evoke.
Sentiment: NEUTRAL
Text Samples: 139
---
```
**Flag Options:**

```bash
  --verbose             Print/show detailed parameter scaling info and progress bars.
  --file-output         Output summaries to text file in working directory.
  --max-samples         Max amount of texts samples from clusters being used in summarization. Default is 500.
  --random-state        Sets value to UMAP and PCA random state. Default value is None.
  --no-pca              Skip PCA and go straight to UMAP.
  --dim-pca             Change PCA dim. Default is 100.
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
mapper = NarrativeMapper(file_df, "r/antiwork", verbose=True)

#embeds semantic vectors
mapper.load_embeddings()

#clustering: has default UMAP and HDBSCAN parameters set but has kwargs for more customizability.
umap_kwargs =  {'n_components': 10, 'min_dist': 0.0}
mapper.cluster(umap_kwargs=**umap_kwargs, use_pca=False)

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
embeddings = get_embeddings(file_df)
cluster_df = cluster_embeddings(embeddings)
summary_df = summarize_clusters(cluster_df)

#export/format options
summary_dict = format_to_dict(summary_df, online_group_name="r/antiwork")
text_df = format_by_text(summary_df, online_group_name="r/antiwork")
cluster_df = format_by_cluster(summary_df, online_group_name="r/antiwork")
```

## Output Formats

This example is based off of [this dataset](https://github.com/Jontom01/NarrativeMapper/blob/main/sample_data/comment_data/comment_data_antiwork_1800.csv)

The three formatter functions return the following:

**format_to_dict()** returns dict with following format:

<details>
<summary>Click to view</summary>

```python

{
    'online_group_name': 'r/antiwork',
    'clusters': [
        {
            'cluster': 2,
            'cluster_summary': 'The cluster focuses on the exploitation of workers under capitalism, highlighting the growing wealth disparity driven by corporate greed, the manipulation of housing markets, and the urgent need for systemic reforms to improve living conditions, wages, and labor rights.',
            'sentiment': 'NEGATIVE',
            'text_count': 483
        },
        {
            'cluster': 4,
            'cluster_summary': 'The conversation cluster centers on critiques of remote work policies, reflections on privilege and inequality, and humorous observations about daily frustrations and absurdities.',
            'sentiment': 'NEGATIVE',
            'text_count': 80
        },
        {
            'cluster': 5,
            'cluster_summary': 'This cluster highlights the frustrations and absurdities of modern job application processes, focusing on discriminatory hiring practices, excessive interview demands, and the dehumanizing effects of AI and psychometric testing on candidates.',
            'sentiment': 'NEGATIVE',
            'text_count': 76
        },
        {
            'cluster': 7,
            'cluster_summary': 'The conversation focuses on the low wages and poor treatment of fast food workers, emphasizing the urgent need for improved compensation and benefits in relation to living costs.',
            'sentiment': 'NEGATIVE',
            'text_count': 58
        },
        {
            'cluster': 8,
            'cluster_summary': 'The conversation cluster highlights pervasive issues of employee dissatisfaction stemming from wage theft, workplace exploitation, toxic environments, harassment, and inadequate labor rights, alongside the struggle for work-life balance and the necessity for legal recourse in employment disputes.',
            'sentiment': 'NEGATIVE',
            'text_count': 392
        }
    ]
}
```
</details>

**format_by_cluster()** returns pandas DataFrame with columns:

<details>
<summary>Click to view</summary>

- **online_group_name:** online group name

- **cluster:** numeric cluster number

- **cluster_summary:** summary of the cluster

- **text_count:** sampled textual messages per cluster

- **aggregated_sentiment:** net sentiment, of form 'NEGATIVE', 'POSITIVE', 'NEUTRAL'

- **text:** the list of textual messages that are part of the cluster

- **all_sentiments:** this is a list containing dict items of the form '{'label': 'NEGATIVE', 'score': 0.9896971583366394}' for each message (sentiment calculated by distilbert-base-uncased-finetuned-sst-2-english).

</details>

[CSV to show output format](https://github.com/Jontom01/NarrativeMapper/blob/main/sample_data/example_outputs/test_2.csv)

**format_by_text()** returns pandas DataFrame with columns:

<details>
<summary>Click to view</summary>

- **online_group_name**: online group name

- **cluster**: numeric cluster number

- **cluster_summary:** summary of the cluster

- **text:** the sampled textual message (this function returns all of them row by row)

- **sentiment:** dict item holding sentiment calculation, of the form '{'label': 'NEGATIVE', 'score': 0.9896971583366394}' (sentiment calculated by distilbert-base-uncased-finetuned-sst-2-english).

</details>

[CSV to show output format](https://github.com/Jontom01/NarrativeMapper/blob/main/sample_data/example_outputs/test_1.csv)


## Pipeline Architecture & API Overview

**Pipeline:**

```txt
CSV Text Data → Embeddings → Clustering → Summarization → Formatting
```
**Functions:**

```python

#Converts each message into a 1536-dimensional vector using OpenAI's text-embedding-3-small.
get_embeddings(file_df, verbose=...)

#Clusters the embeddings using UMAP (for reduction) and HDBSCAN (for density-based clustering).
cluster_embeddings(
    embeddings, 
    verbose=..., 
    use_pca=...,
    pca_kwargs=..., 
    umap_kwargs=..., 
    hdbscan_kwags=...
    )

#Uses GPT (via Chat Completions) for cluster summaries and Hugging Face for sentiment analysis.
summarize_clusters(clustered_df, max_sample_size=..., verbose=...)

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
    def __init__(self, df, online_group_name: str, verbose=False):
        self.verbose               # Verbose for all parts of the pipeline
        self.file_df               # DataFrame of csv file
        self.online_group_name     # Name of the online community or data source
        self.embeddings_df         # DataFrame after embedding
        self.cluster_df            # DataFrame after clustering
        self.summary_df            # DataFrame after summarization

```

**Methods:**
```python
load_embeddings()
cluster(
    use_pca=...,
    pca_kwargs=..., 
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
<summary>Click to expand</summary>

- **verbose:** Print/show detailed parameter scaling info and progress bars.

- **use_pca:** Toggle whether or not you want to use PCA before UMAP (default is True since it helps reduce RAM usage from UMAP).

- **umap_kwargs:** Allows for input of UMAP parameters.

- **hdbscan_kwags:** Allows for input of HDBSCAN parameters.

- **pca_kwargs:** Allows for input of PCA parameters.

</details>


## Estimated Cost (OpenAI Pricing)

Estimated cost: **$0.02 to $0.17 per 1 million tokens**.

Example: A CSV containing 1,000, all greater than one sentence long, Reddit comments costs approximately **$0.01** to process.

<details>
<summary>Click for pricing details</summary>

The OpenAI text-embedding-3-small model costs approximately $0.02 per 1 million input tokens. Determined by the total tokens of your input textual messages.

The Chat Completions model used for summarization (gpt-4o-mini) is $0.15 per 1 million input tokens. The max_sample_size parameter (referenced later) helps reduce costs by limiting how many comments are passed into gpt-4o-mini for each cluster. This can significantly reduce the Chat Completions token usage.

The gpt-4o-mini input prompt (excluding the text) and output summary (for both stages) are very short (<1000 tokens), so their cost contribution is negligible.

</details>