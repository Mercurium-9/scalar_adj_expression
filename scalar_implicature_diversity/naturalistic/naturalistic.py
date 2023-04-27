'''
This code snipet assesses whether SIs triggered by scalar adjectives display general
context insensitivity in naturalistic data.
'''

import torch
import numpy as np
import argparse
from read_scale.read_scale import NgScale
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from controlled.mnli import compute_correlation


def compute_implicature(ws,
                        model_dic,
                        posts,
                        device):
    '''
    This function computes whether SIs triggered by scalar adjectives display general
    context insensitivity in naturalistic data.

        Parameters:
            ws (list): a list of scalar items
            model_dic (dict): a dictionary of paired models and tokenizers
            posts (dict): a dictionary of reddit posts paired with SIs
            device (str): device for processing

        Returns:
            general_implicature_dic (dict): a dictionary of naturalistic implicature results
    '''
    implicature_dic = dict()
    general_implicature_dic = dict()
    for model_name, model_tokenizer in model_dic.items():
        implicature_dic[model_name] = dict()
        for neg in ['non_neg', 'neg']:
            implicature_dic[model_name][neg] = dict()
            general_implicature_dic[f'{model_name}_{neg}'] = list()
            tokenizer = model_tokenizer['tokenizer']
            model = model_tokenizer['model']
            for adj_pair in ws:
                val = posts[adj_pair]
                implicature_dic[model_name][neg][adj_pair] = list()
                for i in val:
                    with torch.no_grad():
                        x = tokenizer(i['post'], i['implicature'].replace(' not ', ' '),
                                      return_tensors='pt')
                        if model_name == 'bert':
                            # label_dic = {0: 'entailment', 1: 'neutral', 2:'contradiction'}
                            logits = model(x['input_ids'].to(device),
                                           x['attention_mask'].to(device),
                                           x['token_type_ids'].to(device))[0]
                            probs = torch.softmax(logits, -1)
                            confidence = float(probs[0][2])
                        else:
                            logits = model(x['input_ids'].to(device),
                                           x['attention_mask'].to(device))[0]
                            probs = torch.softmax(logits, -1)
                            confidence = float(probs[0][0])
                        implicature_dic[model_name][neg][adj_pair].append(confidence)

                general_implicature_dic[f'{model_name}_{neg}']\
                    .append(np.array(implicature_dic[model_name][neg][adj_pair]).mean())

    return general_implicature_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--post_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for reddit posts and implicatures')
    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Output path')
    parser.add_argument('--device',
                        default=None,
                        type=str,
                        required=True,
                        help='Device for processing')
    args = parser.parse_args()

    device = torch.device(args.device)

    bert_tokenizer = AutoTokenizer.from_pretrained("madlag/bert-large-uncased-mnli")
    bert_model = AutoModelForSequenceClassification.\
        from_pretrained("madlag/bert-large-uncased-mnli")
    bert_model.to(device)
    roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
    roberta_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
    roberta_model.to(device)

    model_dic = {'bert': {'model': bert_model, 'tokenizer': bert_tokenizer},
                 'roberta': {'model': roberta_model, 'tokenizer': roberta_tokenizer}}

    with open(args.post_path) as f:
        file = f.read()
        posts = json.loads(file)

    ng_scale = NgScale()
    w, s = ng_scale.get_scalar_items()
    temp_ws = ['/'.join([w[i], s[i]]) for i in range(len(s))]
    ws = list()
    factor_dic = ng_scale.get_factors_and_data()
    si = factor_dic['SI']
    si_dic = {'SI': list()}
    for i in range(len(temp_ws)):
        if temp_ws[i] in posts.keys():
            si_dic['SI'].append(si[i])
            ws.append(temp_ws[i])

    implicature_dic = compute_implicature(ws, model_dic, posts, args.device)
    compute_correlation(implicature_dic, args.out_path, si_dic)
    print('Analysis finished!')


if __name__ == '__main__':
    main()
