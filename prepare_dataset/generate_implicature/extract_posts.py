'''
This code snipet reads Politosphere (Hofmann et al., 2022) and save 2k relevant posts
with unmodified adj expressions for each scalar adjective pair in the dataset used in
Gotner et al. (2018).
'''
import argparse
import pandas as pd
import json
from read_scale.read_scale import NgScale


def filter_short_comments(url):
    '''
    Filter comments shorter than 60 words.

        Parameters:
            url (str): url path to monthly reddit comment

        Returns:
            comments (dict): a dictionary of short comments
    '''
    file = pd.read_json(url, compression='bz2', lines=True, dtype=False, chunksize=10000)
    comments = dict()
    for i in file:
        for dp in i.iterrows():
            comment = dp[1]['body_cleaned']
            if comment and len(comment.split()) < 60:
                comments[dp[1]['id']] = dict(dp[1])
    return comments


def filter_unmodified_comments(url, posts, copula):
    '''
    Select posts containing constructions of [be] [weak_adj] (e.g. is impossible).
    Only keep 2000 distinct posts for each entry.

        Parameters:
            url (str): url path to monthly reddit comment
            posts (dict): dictionary for collected posts
            copula (set): set of copula words

        Returns:
            posts (dict): dictionary for further collected posts
    '''
    comments = filter_short_comments(url)
    for comment in comments.values():
        for w in posts.keys():
            if len(posts[w]) >= 2000:
                continue
            if w not in set(comment['body_cleaned'].split()):
                continue
            left = comment['body_cleaned'].split(w)[0].strip().split()
            # Only select distinct posts in which trigger words are in right-most positions
            # and they are in constructions [be] [trigger] (e.g. is possible)
            if len(comment['body_cleaned'].split(w)) == 2:
                if len(left) > 0 and left[-1] in copula and not comment['body_cleaned'].split(w)[1]\
                 and json.dumps(comment) not in posts[w]:
                    posts[w].append(json.dumps(comment))
                continue
            right = comment['body_cleaned'].split(w)[1]
            # Exclude constuctions which are not in declarative sentences
            if len(left) > 0 and left[-1] in copula and right[0] in '.!'\
                    and json.dumps(comment) not in posts[w]:
                posts[w].append(json.dumps(comment))
    return posts


def collect_posts(weak, out_path):
    '''
    Select and save posts containing constructions of [be] [weak_adj] (e.g. is impossible).
    Only keep 2000 distinct posts for each entry.

        Parameters:
            weak (list): a list of weak adjectives
            out_path (str): path for output file
    '''
    copula = set(['be', 'been', 'was', 'were', 'is', 'am', 'are'])
    posts = dict()
    for year in range(11):
        for i in range(11):
            flag = True
            for w in weak:
                if len(posts[w]) < 2000:
                    flag = False
                    break
            if not flag:
                if i > 2:
                    month = f'{2019-year}-0{12-i}'
                else:
                    month = f'{2019-year}-{12-i}'
                print(f'Filtering unmodified short comments_{month} ...')
                url = f'https://zenodo.org/record/5851729/files/comments_{month}.bz2?download=1'
                posts = filter_unmodified_comments(url, posts, copula)
                # Save file after each reading, in case internet connection breaks
                with open(f'{out_path}/posts.json', 'w') as f:
                    json.dump(posts, f)
            else:
                break
        if flag:
            break


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path for output file')

    args = parser.parse_args()
    ng_scale = NgScale()

    weak = list(set(ng_scale.get_scalar_items()[0]))

    collect_posts(weak, args.out_path)


if __name__ == '__main__':
    main()
