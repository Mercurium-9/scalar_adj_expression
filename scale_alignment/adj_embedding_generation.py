'''
This script generate contextualised word embeddings for base and large models
of BERT and RoBERTa.

The code is mainly based on implementations in Gari Soler & Apidianaki (2020).
There is a sentence set which provides 10 identical context sentences for each scale of
adjectives (e.g. It is good/ excellent work!). Every context sentence contains one adjective
on a scale, and the code substitutes the target adjective with its scalemates to get identical
contexts for all adjectives on the same scale (e.g. replacing 'good' with 'excellent' in 'It is
good'). If adjectives are word-pieced, the embeddings will be the averaged representations of their
sub-tokens.
'''

import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
import pickle
from copy import deepcopy
import argparse


def aggregate_reps(reps_list,
                   hidden_size):
    # This function averages representations of
    # a word that has been split into wordpieces.
    reps = torch.zeros([len(reps_list), hidden_size])
    for i, wrep in enumerate(reps_list):
        w, rep = wrep
        reps[i] = rep

    if len(reps) > 1:
        reps = torch.mean(reps, axis=0)
    reps = reps.view(hidden_size)

    return reps.cpu()


def special_tokenization(sentence, tokenizer):
    map_ori_to_bert = []
    tok_sent = ['[CLS]']

    for orig_token in sentence.split():
        current_tokens_bert_idx = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token)  # tokenize
        tok_sent.extend(bert_token)  # add to my new tokens
        if len(bert_token) > 1:  # if the new token has been 'wordpieced'
            extra = len(bert_token) - 1
            for i in range(extra):
                # list of new positions of the target word
                current_tokens_bert_idx.append(current_tokens_bert_idx[-1]+1)
        map_ori_to_bert.append(tuple(current_tokens_bert_idx))

    tok_sent.append('[SEP]')

    return tok_sent, map_ori_to_bert


def extract_representations(infos,
                            tokenizer,
                            model,
                            device):
    reps = []

    model.eval()
    with torch.no_grad():
        for info in infos:
            tok_sent = info['bert_tokenized_sentence']
            ids = [tokenizer.convert_tokens_to_ids(tok_sent)]
            input_ids = torch.tensor(ids).to(device)
            outputs = model(input_ids)
            hidden_states = outputs[2]
            bpositions = info["bert_position"]

            reps_for_this_instance = dict()
            for i, w in enumerate(info["bert_tokenized_sentence"]):
                if i in bpositions:
                    # all layers
                    for layer in range(len(hidden_states)):
                        if layer not in reps_for_this_instance:
                            reps_for_this_instance[layer] = []
                        embedding = hidden_states[layer][0][i].cpu()
                        reps_for_this_instance[layer].append((w, embedding))
            reps.append(reps_for_this_instance)

    return reps, model


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--sentence_path',
                        default=None,
                        type=str,
                        required=True,
                        help='File of context sentences')

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path for output file')

    parser.add_argument('--model',
                        default=None,
                        type=str,
                        required=False,
                        help='BERT or RoBERTa')

    parser.add_argument('--size',
                        default=None,
                        type='base',
                        required=False,
                        help='Specify if large model is used')

    args = parser.parse_args()

    torch.manual_seed(0)

    # Code for using mps in Mac
    # Change this to coda if using other chips
    device = torch.device('mps') if torch.backends.mps.is_available()\
        else torch.device('cpu')

    print('Mps is available.') if device == torch.device('mps')\
        else print('Mps not available. Using CPU.')

    data = pickle.load(open(args.sentence_path, "rb"))
    infos = []

    if args.model == 'BERT':
        tokenizer = AutoTokenizer.from_pretrained(f"bert-{args.size}-uncased")
        config = AutoConfig.from_pretrained(f'bert-{args.size}-uncased',
                                            output_hidden_states=True)
        model_class = AutoModel.from_pretrained(f'bert-{args.size}-uncased',
                                                config=config)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"roberta-{args.size}")
        config_class = AutoConfig.from_pretrained(f'roberta-{args.size}',
                                                  output_hidden_states=True)
        model_class = AutoModel.from_pretrained(f'roberta-{args.size}',
                                                config=config)

    for scale in data:
        for instance in data[scale]:
            if '' in instance['sentence_words']:
                print(instance['sentence_words'])
            for scaleword in scale:
                cinstance = deepcopy(instance)
                sentence_words = list(cinstance["sentence_words"][:])

                # Replace a by an and viceversa if necessary
                quantifier = sentence_words[cinstance["position"]-1]
                if quantifier == "a" and scaleword[0] in 'aeiou':
                    sentence_words[cinstance["position"]-1] = "an"
                elif quantifier == "an" and scaleword[0] not in 'aeiou':
                    sentence_words[cinstance["position"]-1] = "a"

                # Replace original adjective by current adjective
                sentence_words[cinstance["position"]] = scaleword
                cinstance["position"] = [cinstance["position"]]

                sentence = " ".join(sentence_words)
                tokenization = special_tokenization(sentence, tokenizer)
                bert_tokenized_sentence, mapp = tokenization
                current_positions = cinstance['position']

                if len(current_positions) == 1:
                    # list of positions (might have been split into wordpieces)
                    bert_position = mapp[cinstance['position'][0]]
                elif len(current_positions) > 1:
                    bert_position = []
                    for p in current_positions:
                        bert_position.extend(mapp[p])

                cinstance["bert_tokenized_sentence"] = bert_tokenized_sentence
                cinstance["bert_position"] = bert_position
                cinstance["scale"] = scale
                cinstance["lemma"] = scaleword
                infos.append(cinstance)

    # EXTRACTING REPRESENTATIONS
    reps, model = extract_representations(infos,
                                          tokenizer,
                                          config_class,
                                          model_class, device)

    for rep, instance in zip(reps, infos):
        scale = instance["scale"]
        lemma = instance["lemma"]
        for ins2 in data[scale]:
            if ins2["sentence_words"] == instance["sentence_words"]:
                if "representations" not in ins2:
                    ins2["representations"] = dict()
                if lemma not in ins2["representations"]:
                    ins2["representations"][lemma] = dict()
                for layer in rep:
                    reps = aggregate_reps(rep[layer], model.config.hidden_size)
                    ins2['representations'][lemma][layer] = reps

    pickle.dump(data, open(args.out_path, "wb"))


if __name__ == '__main__':
    main()
