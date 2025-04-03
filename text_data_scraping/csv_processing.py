import pandas as pd
import numpy as np
import csv
"""
This file is simply used to take a list of dictionaries and convert them to CSV file
"""
def dict_to_csv(features_dict_list: list[dict]):
    flattened_data = []
    for item in features_dict_list:
        row = {
            'text': item['text'],
            'id': item['id'],
            'subreddit': item['subreddit'],
            'utc_time': item['utc_time'],
            'cluster': item['cluster']
        }
        #row.update(item['features'])  #merges all features into the row
        flattened_data.append(row)

    #Convert to a DataFrame
    df = pd.DataFrame(flattened_data)

    #Save to CSV
    df.to_csv("ideology_data_EBV.csv", index=False)

def get_specific_averages(csv_file: str,filter_val: str, val_name: str) -> dict:
    fin = {}
    df = pd.read_csv(csv_file)
    filter_val_instances = df[filter_val].tolist()
    filter_val_instances = list(set(filter_val_instances)) #removes and copys in list

    for filter in filter_val_instances:
        filtered_df = df[df[filter_val] == filter] #filters by subreddit
        numeric_only = filtered_df[val_name] #df with only extremity_score
        
        np_2d = numeric_only.values #turn into numpy 2d array
        mean_value = np_2d.mean(axis=0) #return mean value of numpy 2d array
        fin[filter] = float(mean_value)

    return fin


def csv_to_dict(file_name: str) -> list[dict]:
    with open(file_name, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    return data