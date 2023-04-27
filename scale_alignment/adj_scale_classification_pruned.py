'''
This code prunes CROWD and DEMELO to investigate what factors contribute to the
performance divergence of models in these datasets and WILKINSON.
'''

from adj_scale_classification import compute_scale_vec, classify_scale, read_scales
from adj_scale_alignment import read_all_adj_embeddings
from collections import Counter
import random
from wordfreq import word_frequency
import argparse
import json


def prune_dataset(adj_terms,
                  prune_freq,
                  prune_overlap):
    '''
    Prune CROWD and DEMELO to mimic data distribution of WILKINSON.
        Parameters:
            adj_terms (dict): a dictionary containing adj terms in three datasets
            prune_freq (bool): whether to prune words with frequency lower than WILKINSON
            prune_overlap (bool): whether to prune words that appear in more than one scale

        Returns:
            pruned_dic (dict): a dictionary containing pruned adj terms
    '''
    pruned_dic = dict()
    for data_name, data in adj_terms.items():
        words = [adj for i in data for adj in i]
        overlap = [k for k, v in Counter(words).items() if v > 1]
        pruned = list()
        for scale in data:
            curr_scale = list()
            for adj in scale:
                if prune_freq:
                    freq = word_frequency(adj, 'en')
                    if freq < 4.07e-07:
                        continue
                if prune_overlap:
                    if adj in overlap:
                        continue

                curr_scale.append(adj)
            if len(curr_scale) > 1:
                pruned.append(tuple(curr_scale))
        pruned_dic[data_name] = set(pruned)

    return pruned_dic


def generate_selected_embedding_pruned(pruned_dic,
                                       adj_embeddings,
                                       model_size):
    '''
    Randomly choose different sentences for each adjective to compute representations for adjs.
    This function is different from a similar one in adj_scale_classification.py in that
    it takes input of pruned datasets.
    '''
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
            for pruned_key in pruned_dic[key]:
                if set(list(pruned_key)).issubset(set(list(group))):
                    embeddings[key][pruned_key] = group_embedding

    return embeddings


def calculate_pruned_mrr(adj_terms, embedding_path, model_size, model_layer):
    '''
    This function computes mean reciprocal rank for pruned datasets.
    It is different from the similar one in adj_scale_classification.py in that
    it uses generate_selected_embedding_pruned.
    '''
    adj_embeddings = read_all_adj_embeddings(embedding_path)
    selected_embeddings = generate_selected_embedding_pruned(adj_terms, adj_embeddings, model_size)
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

    parser.add_argument('--embedding_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of generated embeddings')

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

    parser.add_argument('--prune_freq',
                        default=True,
                        type=bool,
                        required=False,
                        help='Whether to prune words with low frequencies')

    parser.add_argument('--prune_overlap',
                        default=True,
                        type=bool,
                        required=False,
                        help='Whether to prune words appearing in more than one scale')

    args = parser.parse_args()

    adj_terms = read_scales(args.term_path)

    averaged_res = {'demelo': dict(),
                    'wilkinson': dict(),
                    'crowd': dict()}

    pruned_dic = prune_dataset(adj_terms, args.prune_freq, args.prune_overlap)

    for i in range(0, 10):
        random.seed(i)
        bert_base = calculate_pruned_mrr(pruned_dic,
                                         args.embedding_path+"/bert_base.pkl",
                                         'base',
                                         12)

        bert_large = calculate_pruned_mrr(pruned_dic,
                                          args.embedding_path+"/bert_large.pkl",
                                          'large',
                                          24)

        roberta_large = calculate_pruned_mrr(pruned_dic,
                                             args.embedding_path+"/roberta_large.pkl",
                                             'large',
                                             24)

        roberta_base = calculate_pruned_mrr(pruned_dic,
                                            args.embedding_path+"/roberta_base.pkl",
                                            'base',
                                            12)

        model_dic = {'bert_base': bert_base, 'bert_large': bert_large,
                     'roberta_base': roberta_base, 'roberta_large': roberta_large}

        for data_name in averaged_res.keys():
            for model_name, data in model_dic.items():
                if model_name not in averaged_res[data_name]:
                    averaged_res[data_name][model_name] = [data[data_name]]
                else:
                    averaged_res[data_name][model_name].append(data[data_name])

    with open(args.out_path+'/pruned_dic.json', 'w') as f:
        json.dumps(averaged_res, f)

    print('file is saved!')


if __name__ == '__main__':
    main()
