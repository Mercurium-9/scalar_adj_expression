import json
from scipy.spatial.distance import cosine
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import argparse
from read_scale.read_scale import NgScale


def calculate_anchor_adj(w, s, pronoun, model, tokenizer, device_name):
    '''
    Calculate the embeddings for anchor adjs (weak and strong adjs).

        Parameters:
            w (str): the weak adj
            s (str): the strong adj
            pronoun (str): pronoun used
            model (transformers.models.bert.modeling_bert.BertForMaskedLM): bert model
            tokenizer (transformers.models.bert.tokenization_bert_fast.BertTokenizerFast):
            bert tokenizer
            device_name (str): device for processing

        Returns:
            adj_rep (torch.tensor): word embeddings for four anchor adjs
    '''
    patterns = [f'{pronoun} is {w}, and even {s}.'.capitalize(),
                f'{pronoun} is {w}, and almost {s}.'.capitalize()]
    adj_rep = list()
    for pattern in patterns:
        encodings = tokenizer(pattern, return_tensors="pt")
        ids = encodings.input_ids.to(torch.device(device_name))
        # Get the start and end positions for weak adj segments
        ws = int((ids == 677).nonzero(as_tuple=True)[1])+1
        we = int((ids == 117).nonzero(as_tuple=True)[1])
        # Get the start and end positions for strong adj segments
        if 'even ' in pattern:
            ss = int((ids == 965).nonzero(as_tuple=True)[1])+1
        else:
            ss = int((ids == 1593).nonzero(as_tuple=True)[1])+1
        se = int((ids == 119).nonzero(as_tuple=True)[1])
        with torch.no_grad():
            output = model(ids)
            hidden_states = output.hidden_states
            # Only take the first-layer embedding
            # Average segment representations
            adj_rep.append(hidden_states[-1][0][ws:we].mean(axis=0))
            adj_rep.append(hidden_states[-1][0][ss:se].mean(axis=0))
    return adj_rep


def locate_target_word(tokens, target):
    '''
    Get the positions of target adj embeddings in reddit comments.

        Parameters:
            tokens (list): list of tokenized input
            target (str): the target adj

        Returns:
            target_id (list): list of embedding positions
    '''
    # when target word is not segmented
    if target in tokens:
        target_id = [tokens.index(target)]
    else:
        # when target word is segmented
        for idx in range(len(tokens)):
            if tokens[idx] == target[:len(tokens[idx])]:
                temp_token = tokens[idx]
                temp_idx = idx+1
                while tokens[temp_idx][:2] == '##':
                    temp_token += tokens[temp_idx]
                    temp_idx += 1
                if temp_token.replace('##', '') == target:
                    target_id = [j for j in range(idx, temp_idx)]
                    break
    return target_id


def compute_target_embedding(hidden_states, target_id):
    '''
    Get the embedding for target word in reddit posts.

        Parameters:
            hidden_states (list): list of output hidden states
            target_id (list): list of embedding positions

        Returns:
            embedding (torch.tensor): target word embedding in reddit posts
    '''
    s, e = target_id[0], target_id[-1]+1
    embedding = hidden_states[1][0][s:e].mean(axis=0)
    return embedding


def rank_acceptability(pronoun_dic, posts, model, tokenizer, device_name):
    '''
    Rank acceptability by comparing similarity or target adj and anchor adjs.

        Parameters:
            pronoun_dic (dict): dict of adj pairs and associated pronouns
            posts (dict): dictionary of posts
            model (transformers.models.bert.modeling_bert.BertForMaskedLM): bert model
            tokenizer (transformers.models.bert.tokenization_bert_fast.BertTokenizerFast):
            bert tokenizer
            device_name (str): device for processing

        Returns:
            similar_dic (dict): dictionary for post ids with acceptability ranking
    '''
    device = torch.device(device_name)
    similar_dic = dict()
    embedding_dic = dict()
    for (w, s), pronoun in pronoun_dic:
        anchors = calculate_anchor_adj(w, s, pronoun, tokenizer, device)
        similar_dic[(w, s)] = list()
        embedding_dic[w] = list()
        for i in range(len(posts[w])):
            curr = i['post']
            encodings = tokenizer(curr, return_tensors="pt")
            ids = encodings.input_ids.to(device)
            tokens = tokenizer.convert_ids_to_tokens(ids[0])
            try:
                target_id = locate_target_word(tokens, w)
            # If target word is not in the post, skip
            except Exception:
                continue
            with torch.no_grad():
                output = model(ids)
                hidden_states = output.hidden_states
            target_embedding = compute_target_embedding(hidden_states, target_id)
            embedding_dic[w].append(target_embedding.cpu().numpy())
            cos = 0
            for anchor in anchors:
                cos += cosine(target_embedding.cpu(), anchor.cpu())
            similar_dic[(w, s)].append((cos, i))
    return similar_dic


negation_conditional = set(['not', 'never', 'none', 'no', 'if', 'unless', 'when', 'while'])


def get_final_posts(negation_conditional, similar_dic, posts):
    '''
    Get final posts which rank top 30 for each entry without negation and conditionals.

        Parameters:
            negation_conditional (set): set of negation and conditional markers
            similar_dic (dict): dictionary for post ids with acceptability ranking
            posts (dict): dictionary of posts

        Returns:
            similar_dic (dict): dictionary for post ids with acceptability ranking
    '''

    final_post_dic = dict()
    for key, val in similar_dic.items():
        val.sort(reverse=True)
        final_post_dic[key] = list()
        val.sort()
        temp = set()

        for cos, idx in val:
            post = posts['/'.join(list(key))][idx]['post']
            if set(post.lower().split()).intersection(negation_conditional):
                continue
            implicature = posts['/'.join(list(key))][idx]['implicature']
            if len(final_post_dic[key]) < 30:
                if post not in temp:
                    final_post_dic[key].append({'post': post, 'implicature': implicature})
                    temp.add(post)
            else:
                break

    return final_post_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--post_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for reddit posts')

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=True,
                        help='Path for final posts')

    parser.add_argument('--device',
                        default='mps',
                        type=str,
                        required=False,
                        help='Device for processing')

    args = parser.parse_args()

    torch.manual_seed(0)

    ng_scale = NgScale()
    pronoun_dic = ng_scale.get_pronouns()

    with open(args.post_path) as f:
        file = f.read()
        posts = json.loads(file)

    config = BertConfig.from_pretrained("DeepPavlov/bert-base-cased-conversational",
                                        output_attentions=True,
                                        output_hidden_states=True)

    tokenizer = BertTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational",
                                              config=config)
    model = BertForMaskedLM.from_pretrained("DeepPavlov/bert-base-cased-conversational",
                                            config=config)
    device = torch.device(args.device_name)
    model.to(device)

    similarity_dic = rank_acceptability(pronoun_dic, posts, model, tokenizer, args.device_name)
    final_posts = get_final_posts(negation_conditional, similarity_dic, posts)

    with open(f'{args.out_path}/final_posts.json', 'w') as f:
        json.dump(final_posts, f)


if __name__ == '__main__':
    main()
