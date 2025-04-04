# NarrativeMapper

**Overview:**

The NarrativeMapper package is a discourse analysis pipeline that uncovers the dominant narratives and emotional tones within online communities.

This project processes textual messages from .csv files, then applies OpenAI’s embedding API (text-embedding-3-large) to convert each message into semantic vectors. These embeddings are clustered using UMAP for dimensionality reduction and HDBSCAN for density-based clustering.

For each discovered cluster, the tool:

- Extracts the main talking points OpenAI Chat Completions

- Analyzes the emotional tone using a Hugging Face sentiment classifier

- Outputs structured summaries of the narrative + emotion pairs

**Example Output:**

This example is based off of 1800 r/antiwork comments from the top 300 posts within the last year (Date of Writing: 2025-04-03). 

Output using format_to_dict() function. Useful for JSON export.

<details>
<summary>click to view output example</summary>

```python
{
    'online_group_name': 'r/antiwork',
    'clusters': [
        {
            'label': 'The core theme of this cluster revolves around frustrations and criticisms of modern job application processes, including exploitative practices, ineffective interviews, and the use of AI and personality tests that often discriminate against neurodiverse individuals.',
            'tone': 'NEGATIVE',
            'comment_count': 74
        },
        {
            'label': 'The core themes of this cluster revolve around the challenges of low wages in the fast food and service industries, the rising cost of living, and the perceived disconnect between corporate profits and employee compensation.',
            'tone': 'NEGATIVE',
            'comment_count': 109
        },
        {
            'label': 'The core theme of this cluster revolves around employee dissatisfaction with workplace policies, management practices, and the struggle for work-life balance, often highlighting issues of wage theft, lack of respect for personal time, and the negative impact of corporate culture on mental health.',
            'tone': 'NEGATIVE',
            'comment_count': 500
        },
        {
            'label': "The core theme of this cluster revolves around the dissatisfaction with traditional work schedules, advocating for shorter workweeks and better work-life balance, while highlighting the negative impact of long hours and inadequate parental leave on individuals' well-being.",
            'tone': 'NEGATIVE',
            'comment_count': 83
        },
        {
            'label': "The core theme of this cluster revolves around workers' struggles for fair wages, unionization, and collective action against corporate exploitation, particularly in the context of Boeing.",
            'tone': 'NEGATIVE',
            'comment_count': 56
        },
        {
            'label': 'The comments primarily express strong criticism of Elon Musk and the corporate culture surrounding wealth accumulation, highlighting issues of exploitation, inequality, and the disconnect between CEOs and their employees.',
            'tone': 'NEGATIVE',
            'comment_count': 50
        },
        {
            'label': 'The core theme of this cluster revolves around the critique of wealth inequality and capitalism, highlighting the exploitation of workers, the concentration of wealth among the elite, and the systemic issues that perpetuate economic disparity and social injustice.',
            'tone': 'NEGATIVE',
            'comment_count': 157
        },
        {
            'label': 'The comments reflect widespread frustration and despair among younger generations regarding financial instability, lack of affordable housing, inadequate retirement planning, and the perception of being exploited in the workforce, often contrasting their struggles with the experiences of older generations.',
            'tone': 'NEGATIVE',
            'comment_count': 92
        }
    ]
}

```

</details>

Other formatting functions are available. Both format_by_text() and format_by_cluster() return pandas DataFrames that are well-suited for CSV export.

format_by_cluster() example to showcase output format:

[See full output CSV →](./example_outputs/test_2.csv)

format_by_text() example output to showcase output format:

[See full output CSV →](./example_outputs/test_1.csv)

**Pipeline Architecture:**

----------------------------------------------------------------------------------------------------------------------------

CSV Text Data --> Embeddings (embeddings.py) --> Cluster (clustering.py) --> Summarize (summarize.py)  --> Formatting (formatters.py)

----------------------------------------------------------------------------------------------------------------------------

*embeddings.py:*
Converts textual messages into 3072 dimensional vectors (OPEN AI's text-embedding-3-large).

*clustering.py:*
Clusters embedding vectors using UMAP for reduction and HDBSCAN for clustering.

*summarize.py:*
Determines summaries/label-names (4o-gpt-mini Chat Completion) and sentiment (distilbert-base-uncased-finetuned-sst-2-english) for each cluster. 

*formatters.py:*
Formats summarized clusters into useful forms for data analysis.