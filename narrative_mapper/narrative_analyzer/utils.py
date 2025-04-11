from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from contextlib import nullcontext
import os
import tiktoken

def progress_bars(verbose, bars=True):
    '''
    Used to create 'rich' progress bars used within pipeline
    '''
    if not verbose:
        return nullcontext()
    if bars:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TimeElapsedColumn()
        )
    else:
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            TimeElapsedColumn()
        )

    
def get_openai_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it in your shell or load it in your script using dotenv."
        )
    return key

def batch_list(big_list, model="gpt-4o-mini", max_tokens=2000):
    """
    Splits a list of text strings into batches, ensuring each batch stays under the token limit.

    Args:
        big_list (list): List of text strings to be batched.
        model (str): Model name for tiktoken encoding.
        max_tokens (int): Max tokens allowed per batch (including some buffer).

    Returns:
        List[List[str]]: A list of batches.
    """
    encoding = tiktoken.encoding_for_model(model)
    batches = []
    current_batch = []
    current_tokens = 0

    for text in big_list:
        text_tokens = len(encoding.encode(text))

        #if adding this text exceeds the limit, start a new batch
        if current_tokens + text_tokens > max_tokens:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(text)
        current_tokens += text_tokens

    #add any leftover batch
    if current_batch:
        batches.append(current_batch)

    return batches
