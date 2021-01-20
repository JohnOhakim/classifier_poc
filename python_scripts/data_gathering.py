import requests
import time
import pandas as pd
import numpy as np

url_1 = 'https://www.reddit.com/r/movies.json'
url_2 = 'https://www.reddit.com/r/television.json'

headers = {'User-agent': 'fbi surveillance'}
res_1 = requests.get(url_1, headers=headers)
res_2 = requests.get(url_2, headers=headers)

def fetch_data(url, res, headers):

    posts = []
    after = None

    for i in range(10):
        print(i)
        if not after:
            params = {'limit': 100}
        else:
            params = {'after': after, 'limit': 100}

        res = requests.get(url, params = params, headers=headers)
        if res.status_code == 200:
            json_file = res.json()
            posts.extend(json_file['data']['children'])
            after = json_file['data']['after']
        else:
            print(f'The status code is: {res.status_code}')
            break
        time.sleep(1)
    return posts

   
def to_dataframe(posts, limit=812):

    user_posts = []

    for i in range(limit):
        list_of_posts = {}
        list_of_posts['post'] = posts[i]['data']['selftext']
        list_of_posts['num_of_comments'] = posts[i]['data']['num_comments']
        list_of_posts['title'] = posts[i]['data']['title']
        list_of_posts['subreddit'] = posts[i]['data']['subreddit']
        user_posts.append(list_of_posts)
        
    df = pd.DataFrame(user_posts)
    return df


mov_posts = fetch_data(url_1, res_1, headers)
df_movies = to_dataframe(mov_posts, limit=812)

tv_posts = fetch_data(url_2, res_2, headers)
df_tv = to_dataframe(tv_posts, limit=812)


## Feature Enginnering

def combine_posts(df):

    if type(df['post']) ==str:
        df['posts_combined']= df['post'] +'\n'+ df['title']
        
    else:
        df['posts_combined']= df['title']
    
    return df


df_m = combine_posts(df_movies)
df_t = combine_posts(df_tv)

df_mt = [df_m, df_t]

df = pd.concat(df_mt, sort=False).reset_index(drop=True)

reddit_3 = df[['posts_combined', 'subreddit']]

reddit_3['is_tv'] = reddit_3['subreddit'].map({'movies': 0, 'television': 1})

reddit_3.to_csv('reddit_post_3.csv', index=False)

posts_1 = pd.read_csv('reddit_post_3.csv')
posts_2 = pd.read_csv('reddit_post_train.csv')
posts_3 = pd.read_csv('reddit_posts.csv')

posts_4 = posts_3[['subreddit', 'posts_combined']]

posts_4['is_tv'] = posts_4['subreddit'].map({'movies': 0, 'television': 1})

posts_1.drop_duplicates(inplace=True)
posts_2.drop_duplicates(inplace=True)
posts_4.drop_duplicates(inplace=True)

posts = [posts_1, posts_2, posts_4]

df = pd.concat(posts, sort=False).reset_index(drop=True)

mask = np.random.rand(len(df)) < 0.6 


train = df[mask]
test = df[~mask]

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)