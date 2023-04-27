'''
This code snipet assesses to what extent BERT-large and RoBERTa-large fine-tuned on MNLI
can predict scalar diversity corresponding to human cognitive patterns.
Dataset used is controlled data in Gotzner et al. (2018).
'''

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import argparse
from scipy import stats
from read_scale.read_scale import NgScale


def compute_implicature(model_dic,
                        ws,
                        pronoun_dic,
                        device):
    '''
    This function computes how likely finetuned models are to reason pragmatically about
    scalar diversity.

        Parameters:
            model_dic (dict): a dictionary of paired models and tokenizers
            ws (list): a list of scalar items
            pronoun_dic (dict): a dictionary of scalar adj pairs and corresponding pronouns
            device_name (torch.device): device for processing

        Returns:
            implicature_dic (dict): a dictionary of implicature results

    '''
    implicature_dic = dict()
    for model_name, model_tokenizer in model_dic.items():
        # Get results for non-negated and negated settings
        for neg in ['non_neg', 'neg']:
            implicature = list()
            tokenizer = model_tokenizer['tokenizer']
            model = model_tokenizer['model']
            for w, s in ws:
                pronoun = pronoun_dic[(w, s)]
                with torch.no_grad():
                    if neg == 'non_neg':
                        x = tokenizer(f'{pronoun} is {w}.', f'{pronoun} is {s}.',
                                      return_tensors='pt')
                    else:
                        x = tokenizer(f'{pronoun} is {w}.', f'{pronoun} is not {s}.',
                                      return_tensors='pt')
                    # BERT and RoBERTa have different label dict
                    if model_name == 'bert':
                        # label_dic = {0: 'entailment', 1: 'neutral', 2:'contradiction'}
                        logits = model(x['input_ids'].to(device),
                                       x['attention_mask'].to(device),
                                       x['token_type_ids'].to(device))[0]
                        probs = torch.softmax(logits, -1)
                        if neg == 'non_neg':
                            confidence = float(probs[0][2])
                        else:
                            confidence = float(probs[0][0])
                    else:
                        logits = model(x['input_ids'].to(device),
                                       x['attention_mask'].to(device))[0]
                        probs = torch.softmax(logits, -1)
                        if neg == 'non_neg':
                            confidence = float(probs[0][0])
                        else:
                            confidence = float(probs[0][2])
                    implicature.append(confidence)
            implicature_dic[f'{model_name}_{neg}'] = implicature

    return implicature_dic


def compute_correlation(implicature_dic, out_path, factor_dic):
    '''
    This function computes the correlation between models' SI prediction confidence and
    scalar diversity factors in dataset offered by Gotnzer et al. (2018).

        Parameters:
            implicature_dic (dict): a dictionary of implicature results
            out_path (str): path for output correlation file
            factor_dic (dict): scalar diversity factors by Gotnzer et al. (2018)

    '''
    index = list()
    for i in list(factor_dic.keys()):
        index += [f'{i}_r', f'{i}_p']

    spearmanr_df = pd.DataFrame(columns=list(implicature_dic.keys()),
                                index=index)

    for model in list(spearmanr_df.columns):
        for feature in list(factor_dic.keys()):
            spearmanr = stats.spearmanr(list(factor_dic[feature]), implicature_dic[model])
            spearmanr_df[model][f'{feature}_r'] = '%.3f' % spearmanr.correlation
            spearmanr_df[model][f'{feature}_p'] = '%.3f' % spearmanr.pvalue

    spearmanr_df.to_csv(out_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Output path for correlation data frame')
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

    ng_scale = NgScale()
    w, s = ng_scale.get_scalar_items()
    ws = zip(w, s)
    pronoun_dic = ng_scale.get_pronouns()
    factor_dic = ng_scale.get_factors_and_data()

    implicature_dic = compute_implicature(model_dic, ws, pronoun_dic, args.device)
    compute_correlation(implicature_dic, f'{args.out_path}/controlled_mnli.csv', factor_dic)
    print('Analysis finished!')


if __name__ == '__main__':
    main()
