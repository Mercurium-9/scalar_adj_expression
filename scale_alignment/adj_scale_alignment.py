'''
This code aims to investigate whether models can distinguish in-scale adjectives from
random adjectives.

Each adjective's is represented by its embedding in one of its context sentence (randomly
selected). This is done so that in-scale adjectives are not pulled together because of
identical contexts.

For each scale, we compute a scale vector by adding the representations of the weakest
and strongest adjective on a scale. We then compute the cosine similarity between each
adjective (including the strongest and weakest adjective) on the scale with the scale
vector, and average the similarity as the in-scale similarity for the scale. Random
similarity is computed by calculating the cosine similarity between one random adjective
in the dataset which does not belong to the target scale and the scale vector. The experiment
is repeated 10 times with different random seeds each time to get different random adjectives
and context sentences. Random baseline is drawn by averaging results in 10 experiments.

Experiment is done with respect to different layers (12 for base models and 24 for large
models). For each layer, the in-scale similarity and random similarity is computed by averaging
the similarity for each scale over the number of scales.
'''

import pickle
import os
import random
from scipy.spatial.distance import cosine
import pandas as pd
import argparse


# Read scales from three datasets
# Modified from Gari Soler & Apidianki (2020)
def read_scales(dirname):
    adj_terms = dict()
    for f in os.listdir(dirname):
        # Skip system file
        if f == '.DS_Store':
            continue
        folder = os.path.join(dirname, f)
        adj_terms[f] = list()
        termsfiles = [os.path.join(folder+'/terms', termfile)
                      for termfile in os.listdir(folder + '/terms') if termfile != '.DS_Store']
        for tf in termsfiles:
            terms = []
            with open(tf, 'r') as fin:
                for line in fin:
                    __, w = line.strip().split('\t')
                    terms.append(w)

            adj_terms[f].append(tuple(terms))
        adj_terms[f] = set(adj_terms[f])
    return adj_terms


# Read all adj embeddings generated from context sentences
def read_all_adj_embeddings(path):
    adjs = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                adjs.append(pickle.load(openfile))
            except EOFError:
                break
        openfile.close()
    return adjs[0]


# Randomly choose different sentences for each adjective to compute representations
# for adjs in three sets
def generate_selected_embedding(adj_terms,
                                adj_embeddings,
                                model_size='base'):
    embeddings = {'demelo': dict(),
                  'crowd': dict(),
                  'wilkinson': dict()}
    if model_size == 'base':
        layer_num = 12
    else:
        layer_num = 24
    for group in adj_embeddings.keys():
        group_embedding = dict()
        random_sample = random.sample(range(10), len(group))
        for i in range(len(group)):
            sent_id = random_sample[i]
            adj = group[i]
            group_embedding[adj] = dict()
            for layer in range(layer_num+1):
                layer_rep = adj_embeddings[group][sent_id]['representations'][adj][layer]
                group_embedding[adj][layer] = layer_rep
        for key in embeddings.keys():
            if group in adj_terms[key]:
                embeddings[key][group] = group_embedding

    return embeddings


# Calculate cosine similarity for adjectives in scale
def in_scale_sim(embeddings,
                 layer,
                 alignment_dic):

    sim_dic = dict()
    for data_name in embeddings.keys():
        data = embeddings[data_name]
        for group in data.keys():
            sim = 0
            scale = (data[group][group[-1]][layer]+data[group][group[0]][layer]).tolist()
            for word in group:
                sim += 1-cosine(data[group][word][layer], scale)
            sim_dic[group] = sim/(len(group))
        in_scale_sim = sum(sim_dic.values())/len(sim_dic.values())
        alignment_dic[data_name]['in-scale'][layer].append(in_scale_sim)


# Calculate cosine similarity between random adjectives and scale vector
def random_sim(embeddings,
               layer,
               alignment_dic):

    for data_name in embeddings.keys():
        rand_sim = 0
        data = embeddings[data_name]
        for group in data.keys():
            scale = (data[group][group[-1]][layer]+data[group][group[0]][layer]).tolist()
            # Randomly select an adjective not in the scale
            rand_group = group
            while rand_group == group:
                rand_group = random.choice(list(data.keys()))
            rand_adj = random.choice(list(data[rand_group]))

            rand_embedding = data[rand_group][rand_adj][layer]
            rand_sim += 1-cosine(rand_embedding, scale)
        average_rand_sim = rand_sim/len(data)
        alignment_dic[data_name]['random'][layer].append(average_rand_sim)


# Compute random and in-scale similarity
def compute_alignment(adj_embeddings,
                      adj_terms,
                      model_size='base'):

    if model_size == 'base':
        layer_num = 12
    else:
        layer_num = 24

    alignment_dic = dict()
    # Generate dictionary to hold alignment results
    for data_name in ['demelo', 'crowd', 'wilkinson']:
        alignment_dic[data_name] = {'in-scale': dict(),
                                    'random': dict()}
        for key in alignment_dic[data_name].keys():
            for layer in range(1, layer_num+1):
                alignment_dic[data_name][key][layer] = []

    # Repeat the same experiment 10 times
    for i in range(10):
        # Set different random seeds for each time
        random.seed(i)
        selected_embeddings = generate_selected_embedding(adj_terms, adj_embeddings, model_size)
        # Compute embedding for each layer
        for layer in range(1, layer_num+1):
            in_scale_sim(selected_embeddings, layer, alignment_dic)
            random_sim(selected_embeddings, layer, alignment_dic)

    return alignment_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--embedding_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of context sentences')

    parser.add_argument('--term_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of scalar terms')

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path for output file')

    parser.add_argument('--model_name',
                        default=None,
                        type=str,
                        required=False,
                        help='BERT or RoBERTa')

    parser.add_argument('--model_size',
                        default='base',
                        type=str,
                        required=False,
                        help='Specify if large model is used')

    args = parser.parse_args()

    adj_embeddings = read_all_adj_embeddings(args.embedding_path)
    adj_terms = read_scales(args.term_path)

    alignment_dic = compute_alignment(adj_embeddings, adj_terms, args.model_size)

    for data_name, data in alignment_dic.items():
        df = pd.DataFrame(data)
        df.to_csv(f'{args.out_path}/{args.model_name.lower()}_{args.model_size}/\
                    {args.model_name.lower()}_{args.model_size}_{data_name}.csv')
    print('file is saved!')


if __name__ == '__main__':
    main()
