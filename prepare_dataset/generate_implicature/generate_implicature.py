'''
This code snipet generates scalar implicatures for reddit posts.
Generated file will be saved to output path.
'''
import re
import spacy
import json
import argparse


def customised_clean_body(text, url_pattern=re.compile(r"((http|https)\:\/\/)?\
        [a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*")):
    '''
    Customise a text cleaning function.

        Parameters:
            url_pattern (re.Pattern): a pattern for recognising urls
            text (str): a string of post

        Returns:
            cleaned_text (str): a string of cleaned post
    '''
    cleaned_text = text.replace(' \\u2019t ', "'")
    cleaned_text = cleaned_text.replace("\'", "'")
    cleaned_text = cleaned_text.replace("\\s", " ")
    cleaned_text = cleaned_text.replace("&amp;", "&")
    cleaned_text = cleaned_text.replace('&gt;', '')
    cleaned_text = re.sub(url_pattern, '[URL]', cleaned_text)
    return cleaned_text


def get_phrase(dep_parser, post, w):
    '''
    Get the phrase containing target weak adj.

        Parameters:
            dep_parser (spacy.lang.en.English): spacy dependency parser
            post (str): a string of post
            w (str): the weak adj in post

        Returns:
            phrase (str): a string of phrase containing the weak adj
    '''
    res = list()
    docs = dep_parser(post)
    found_adj = False
    found_copula = False
    for i in range(len(docs)-1, -1, -1):
        if found_adj:
            res.append(docs[i].text)
        if docs[i].text == w:
            found_adj = True
            res.append(docs[i].text)
            continue
        if found_adj and docs[i].lemma_ == 'be':
            found_copula = True
        if docs[i].pos_ not in 'AUX' and found_adj:
            if not found_copula:
                return ''
            phrase = ' '.join(res[:-1][::-1])
            return phrase
    return ''


def get_clause(dep_parser, post, sent, w, s):
    '''
    Get the clause containing target phrase.

        Parameters:
            dep_parser (spacy.lang.en.English): spacy dependency parser
            post (str): a string of post
            sent (spacy.tokens.span.Span): sentence containing target phrase
            w (str): the weak adj in post

        Returns:
            clause (str): a string of clause containing target phrase.
    '''
    phrase = get_phrase(dep_parser, post, w)
    temp = sent
    flag = True
    is_s = True
    while flag:
        flag = False
        for i in list(temp._.children):
            if phrase in i.text and i.text != phrase:
                temp = i
                if 'S' in i._.labels:
                    is_s = True
                else:
                    is_s = False
                flag = True
                break
    if is_s:
        clause = f'not {s}'.join(temp.text.split(w))
        return clause
    return ''


def generate_implicature(dep_parser, con_parser, dic):
    '''
    Generate scalar implicatures for reddit posts

        Parameters:
            dep_parser (spacy.lang.en.English): spacy dependency parser
            con_parser (spacy.lang.en.English): spacy constituency parser (benepar)
            dic (dict): dictionary for reddit posts

        Returns:
            implicature_dic (dict): dictionary for reddit posts paired with implicatures
    '''
    implicature_dic = dict()
    for key, val in dic.items():
        w, s = key.split('/')
        implicature_dic[key] = list()
        for ele in val:
            post = customised_clean_body(json.loads(ele)['body'])
            try:
                sents = con_parser(post).sents
            # Skip if there is no complete sentence.
            except Exception:
                continue
            for sent in list(sents):
                if w in sent.text:
                    break
            implicature = get_clause(dep_parser, post, sent, w, s)
            if not implicature:
                continue
            implicature_dic[key].append({'implicature': implicature,
                                        'post': post})

    return implicature_dic


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument('--post_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path for filtered reddit posts')

    parser.add_argument('--out_path',
                        default=None,
                        type=str,
                        required=False,
                        help='Path for output file')

    args = parser.parse_args()

    # Load benepar
    dep_parser, con_parser = spacy.load('en_core_web_sm'), spacy.load('en_core_web_sm')
    con_parser.add_pipe('benepar', config={'model': 'benepar_en3'})

    # Load reddit posts
    with open(args.post_path) as f:
        file = f.read()
        dic = json.loads(file)

    implicature_dic = generate_implicature(dep_parser, con_parser, dic)

    with open(f'{args.out_path}/post_implicature.json', 'w') as f:
        json.dump(implicature_dic, f)


if __name__ == '__main__':
    main()
