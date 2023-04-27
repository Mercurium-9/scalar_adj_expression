'''
This code investigates whether models can align adjectives on their scales.

Each adjective's is represented by its embedding in one of its context sentence (randomly
selected). This is done so that in-scale adjectives are not pulled together because of
identical contexts.

For each scale, we compute a scale vector by adding the representations of the weakest
and strongest adjective on a scale. We then compute the cosine similarity between every
adjective with every scale vector, and rank the similarity score to predict what is the
most likely scale it belongs to (e.g. scale with similarity of 0.8 get ranked 1, scale with
similarity of 0.6 get ranked 2 etc.) The experiment is repeated 10 times with different random
seeds each time to get different context sentences.

Experiment is done with respect to different layers (12 for base models and 24 for large
models). MRR is used as the evaluation metric. For each layer, MRR is averaged over the number
of the number of all adjectives in a dataset.
'''

import pandas as pd
import argparse
from scipy.spatial.distance import cosine
import random
from adj_scale_alignment import read_scales, read_all_adj_embeddings, generate_selected_embedding


# Generate a scale vector for each scale
def compute_scale_vec(selected_embeddings, model_layer):
    scale_vec = dict()
    for data_name, data in selected_embeddings.items():
        scale_vec[data_name] = dict()
        for scale in data.keys():
            scale_vec[data_name][scale] = dict()
            # Weakest and strongest adjectives on a scale
            a_w, a_s = scale[0], scale[-1]
            for layer in range(1, model_layer+1):
                # Add representations of a_w and a_s in every layer
                scale_vec[data_name][scale][layer] = data[scale][a_w][layer]+data[scale][a_s][layer]
    return scale_vec


# Classify the scale that target adjectives belong to
def classify_scale(selected_embeddings, scale_vec, model_layer):
    adjs = dict()
    for layer in range(1, model_layer+1):
        adjs[layer] = dict()
        for data_name, data in selected_embeddings.items():
            curr_scale_vec = scale_vec[data_name]
            adjs[layer][data_name] = dict()
            for scale, scale_adjs in data.items():
                for adj in scale:
                    cos_a_s = list()
                    for vec_name in curr_scale_vec.keys():
                        cos = cosine(curr_scale_vec[vec_name][layer], scale_adjs[adj][layer])
                        cos_a_s.append((cos, vec_name))
                    cos_a_s.sort()
                    for idx, item in enumerate(cos_a_s):
                        if item[-1] == scale:
                            scale_rank = idx
                            break
                    adjs[layer][data_name][(scale, adj)] = {'cos_rank': cos_a_s,
                                                            'alignment_rank': scale_rank}
    return adjs


def calculate_mrr(term_dir, embedding_path, model_size, model_layer):
    adj_terms = read_scales(term_dir)
    adj_embeddings = read_all_adj_embeddings(embedding_path)
    selected_embeddings = generate_selected_embedding(adj_terms, adj_embeddings, model_size)
    scale_vec = compute_scale_vec(selected_embeddings, model_layer)
    adjs = classify_scale(selected_embeddings, scale_vec, model_layer)
    mrr_res = {'demelo': [], 'crowd': [], 'wilkinson': []}
    for layer in range(1, model_layer+1):
        curr = adjs[layer]
        for data_name in curr.keys():
            mrr = 0
            for ranking in curr[data_name].values():
                mrr += 1/(ranking['alignment_rank']+1)
            mrr /= len(curr[data_name])
            mrr_res[data_name].append(mrr)
    return mrr_res


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--term_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to adj terms')

    parser.add_argument('--bert_base_data_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to BERT base embeddings')

    parser.add_argument('--bert_large_data_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to BERT large embeddings')

    parser.add_argument('--roberta_base_data_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to RoBERTa base embeddings')

    parser.add_argument('--roberta_large_data_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path to RoBERTa large embeddings')

    args = parser.parse_args()

    averaged_res = {'demelo': {'bert_base': pd.Series(dtype='float64'),
                               'roberta_base': pd.Series(dtype='float64'),
                               'bert_large': pd.Series(dtype='float64'),
                               'roberta_large': pd.Series(dtype='float64')},
                    'wilkinson': {'bert_base': pd.Series(dtype='float64'),
                                  'roberta_base': pd.Series(dtype='float64'),
                                  'bert_large': pd.Series(dtype='float64'),
                                  'roberta_large': pd.Series(dtype='float64')},
                    'crowd': {'bert_base': pd.Series(dtype='float64'),
                              'roberta_base': pd.Series(dtype='float64'),
                              'bert_large': pd.Series(dtype='float64'),
                              'roberta_large': pd.Series(dtype='float64')}}
    for i in range(10):
        random.seed(i)
        bert_base = calculate_mrr(args.term_path,
                                  args.bert_base_data_path,
                                  'base',
                                  12)

        bert_large = calculate_mrr(args.term_path,
                                   args.bert_large_data_path,
                                   'large',
                                   24)
        roberta_large = calculate_mrr(args.term_path,
                                      args.roberta_large_data_path,
                                      'large',
                                      24)
        roberta_base = calculate_mrr(args.term_path,
                                     args.roberta_base_data_path,
                                     12)
        model_dic = {'bert_base': bert_base, 'bert_large': bert_large,
                     'roberta_base': roberta_base, 'roberta_large': roberta_large}
        for data_name in averaged_res.keys():
            for model_name, data in model_dic.items():
                averaged_res[data_name][model_name] = averaged_res[data_name][model_name].\
                                                    add(pd.Series(data[data_name]), fill_value=0)


if __name__ == '__main__':
    main()
