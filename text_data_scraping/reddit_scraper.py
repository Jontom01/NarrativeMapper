# data/scrape_reddit.py
import praw
import requests
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")

def scrape_subreddit(subreddit_names: list[str], limit: int = 10) -> str:

    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,        
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )
    comment_list = []
    for subreddit_name in subreddit_names:
        print(subreddit_name)
        subredditVar = reddit.subreddit(subreddit_name)
        
        for submission in subredditVar.top(time_filter="year", limit=limit): #get submission objects from subreddit object
            comments_in_post = submission.comments #get comment objects from submission object

            num_of_comments_to_grab = min(10,len(comments_in_post))
            for i in range(1,num_of_comments_to_grab):
                text = comments_in_post[i].body.strip()
                #skip short or unhelpful comments
                if text.lower() in ["[deleted]", "[removed]"] or len(text.split()) < 15:
                    continue
                else:
                    comment_list.append({
                        'text': comments_in_post[i].body, 
                        'subreddit': subreddit_name, 
                        'utc_time': comments_in_post[i].created_utc,
                        'id': comments_in_post[i].id
                        })


    flattened_data = []
    for item in comment_list:
        row = {
            'text': item['text'],
            'id': item['id'],
            'subreddit': item['subreddit'],
            'utc_time': item['utc_time']
        }
        flattened_data.append(row)

    df = pd.DataFrame(flattened_data)
    file_name = "comment_data.csv"
    df.to_csv(file_name, index=False)

    return file_name


if __name__ == "__main__":
    subreddit_names = [
        "conservative"
    ]
    scrape_subreddit(subreddit_names, limit=300)