'''
This file selects posts for mlm further training (i.e. generate the SI-100, SI-500 datasets).
'''

from read_scale.read_scale import NgScale
from generate_implicature.generate_implicature import customised_clean_body
from generate_implicature.extract_posts import filter_short_comments
import json
import argparse
import random


class SetEncoder(json.JSONEncoder):
    '''
    Modify class to serialize set object.
    '''
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


def filter_posts(comments, posts):
    '''
    Select dictinct posts containing weak or strong adjectives.

        Parameters:
            comments (dict): a dictionary of short comments

        Returns:
            posts (dict): a dictionary of filtered posts
    '''
    for comment in comments.values():
        for w in posts.keys():
            if len(posts[w]) >= 10000:
                continue
            if set(w.split()).intersection(set(comment['body_cleaned'].split()))\
                    and w in comment['body_cleaned']:
                post = customised_clean_body(comment['body'])
                posts[w].add(post)
    print('posts filtered')
    return posts


def generate_si_10k(out_path, posts):
    '''
    Generate SI-10k as a superset for training file.

        Parameters:
            out_path (str): path for output files
            posts (dict): a dictionary of posts

        Returns:
            posts (dict): a dictionary of selected posts
    '''
    for year in range(11):
        flag = False
        for i in range(11):
            flag = True
            for w in posts.keys():
                if len(posts[w]) < 10000:
                    flag = False
                    break
            if not flag:
                if i > 2:
                    month = f'{2019-year}-0{12-i}'
                else:
                    month = f'{2019-year}-{12-i}'
                print(f'Filtering unmodified short comments_{month} ...')
                url = f'https://zenodo.org/record/5851729/files/comments_{month}.bz2?download=1'
                comments = filter_short_comments(url)
                posts = filter_posts(comments, posts)
                with open(f'{out_path}/si_10k.json', 'w') as f:
                    json.dump(posts, f, cls=SetEncoder)
                print('file saved')
            else:
                break
        if flag:
            break

    return posts


def generate_si_subset(si_10k, size):
    '''
    Generate dataset for further training.

        Parameters:
            si_10k (dict): a dictionary of selected posts
            size (int): data size for per entry

        Returns:
            dataset (str): a string of dataset for further training
    '''
    random.seed(0)
    si_subset = list()
    for val in si_10k.values():
        si_subset += random.sample(val, min(size, len(val)))
    random.shuffle(si_subset)
    dataset = '\n'.join(si_subset)

    return dataset


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path for output file')

    args = parser.parse_args()

    # Initiate a dictionary for posts
    ng_scale = NgScale()
    w, s = ng_scale.get_scalar_items()
    words = set(w+s)
    posts = dict()
    for w in words:
        posts[w] = set()

    si_10k = generate_si_10k(args.out_path, posts)

    for size in [100, 500]:
        dataset = generate_si_subset(si_10k, size)
        with open(f'{args.out_path}/si_{str(size)}.txt', w) as f:
            f.write(dataset)
