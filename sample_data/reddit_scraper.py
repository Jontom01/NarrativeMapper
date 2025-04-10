from dotenv import load_dotenv
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
import praw
import requests
import pandas as pd
import os

load_dotenv()

#TO USE:
#1. Create reddit application to use praw (this is free you just have to set it up).
#2. Put the following three variables in an .env file (with the values you received from creating a reddit application).
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

def scrape_subreddit(
    subreddit_name_list: list[str], 
    submission_limit: int=10, 
    comment_limit: int=10, 
    time_filter: str="year", 
    file_name: str="reddit_data.csv"
    ) -> str:

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,        
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    comment_list = []
    time_created_list = []
    comment_id_list = []
    subreddit_list = []
    subredditVar = reddit.subreddit(subreddit_name)
    progress_context_sentiment = (
    Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
        ))
    for subreddit_name in subreddit_names:
        print(subreddit_name)
        subredditVar = reddit.subreddit(subreddit_name)
        with progress_context_sentiment as progress:
            task = progress.add_task(f"[cyan]Scraping {subreddit_name}...", total=submission_limit)
            for submission in subredditVar.top(time_filter=time_filter, limit=submission_limit): #get submission objects from subreddit object
                #submission.comment_sort = 'new'
                comments_in_post = submission.comments #get comment objects from submission object
                num_of_comments_to_grab = min(comment_limit, len(comments_in_post))
                for i in range(1,num_of_comments_to_grab):
                    text = comments_in_post[i].body.strip()
                    if text.lower() in ["[deleted]", "[removed]"] or len(text.split()) < 15: #skip short or deleted/removed comments
                        continue
                    else:
                        comment_list.append(comments_in_post[i].body)
                        time_created_list.append(comments_in_post[i].created_utc)
                        comment_id_list.append(comments_in_post[i].id)
                subreddit_list += [subreddit_name] * num_of_comments_to_grab
                progress.update(task, advance=1)

    df = pd.DataFrame({
        'subreddit': subreddit_list, 
        'text': comment_list, 
        'created_utc': time_created_list, 
        'comment_id': comment_id_list
        })

    df.to_csv(file_name, index=False)

if __name__ == "__main__":

    subreddit_name = ["politics", "antiwork"]
    
    scrape_subreddit(subreddit_name, submission_limit=500, comment_limit=10, file_name="HUGE_comment_data_politics_.csv")