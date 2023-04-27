'''
This code snipet assesses to what extent BERT-large and RoBERTa-large can predict
scalar diversity corresponding to human cognitive patterns in a zero-shot setting.
Dataset used is controlled data in Gotzner et al. (2018).
'''

import torch
from transformers import pipeline
import argparse
from mnli import compute_correlation
from read_scale.read_scale import NgScale


def compute_implicature_mlm(ws,
                            pronoun_dic,
                            bert_pl,
                            roberta_pl):
    '''
    This function computes how likely zero-shot models are to reason pragmatically about
    scalar diversity.

        Parameters:
            ws (list): a list of scalar items
            pronoun_dic (dict): a dictionary of scalar adj pairs and corresponding pronouns
            bert_pl (transformers.pipelines.fill_mask.FillMaskPipeline): BERT MLM pipeline
            RoBERTa_pl (transformers.pipelines.fill_mask.FillMaskPipeline): RoBERTa MLM pipeline

        Returns:
            implicature_dic (dict): a dictionary of implicature results

    '''
    implicature_dic = dict()
    for model_name in ['BERT', 'RoBERTa']:
        implicature = list()
        for w, s in ws:
            pronoun = pronoun_dic[(w, s)]
            with torch.no_grad():
                if model_name == 'BERT':
                    template = f"Is {pronoun} {s}? {bert_pl.tokenizer.mask_token},\
                            {pronoun} is {w}."
                    output = bert_pl(template,
                                     targets=['yes', 'no'])
                    for i in output:
                        if i['token_str'] == 'no':
                            no = i['score']
                        else:
                            yes = i['score']

                    confidence = no/(no+yes)

                else:
                    template = f"Is {pronoun} {s}? {roberta_pl.tokenizer.mask_token},\
                            {pronoun} is {w}."
                    output = roberta_pl(template,
                                        targets=['Yes', 'No'])

                    for i in output:
                        if i['token_str'] == 'No':
                            no = i['score']
                        else:
                            yes = i['score']

                    confidence = no/(no+yes)

                implicature.append(confidence)
            implicature_dic[f'{model_name}'] = implicature

    return implicature_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

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
    bert_pl = pipeline("fill-mask", model="bert-large-uncased", device=device)
    roberta_pl = pipeline("fill-mask", model="roberta-large", device=device)

    ng_scale = NgScale()
    w, s = ng_scale.get_scalar_items()
    ws = zip(w, s)
    pronoun_dic = ng_scale.get_pronouns()
    factor_dic = ng_scale.get_factors_and_data()

    implicature_dic = compute_implicature_mlm(ws, pronoun_dic, bert_pl, roberta_pl)
    compute_correlation(implicature_dic, f'{args.out_path}/controlled_zero.csv', factor_dic)
    print('Analysis finished!')


if __name__ == '__main__':
    main()
