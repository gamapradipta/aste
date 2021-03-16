import numpy as np
import csv
import argparse
import os
import json
import matplotlib.pyplot as plt

def load_data(filename):
    data , labels = [], []
    with open(filename) as file:
        read_tsv = csv.reader(file, delimiter="\t",  quoting=csv.QUOTE_NONE)
        tokens, tags = [], []
        for row in read_tsv:
            if len(row) > 1:
                tokens.append(row[2])
                tags.append(row[3])
            else:
                if len(tokens) > 0:
                    data.append(tokens)
                    labels.append(tags)
                    tokens,tags= [],[]
    return data, labels

def load_data_json(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def extract_term_tag(tags):
    for tag in tags:
        if tag.startswith("ASPECT") or tag.startswith("SENTIMENT"):
            return tag
    return None

def get_all_term(tags, term):
    prev = ""
    term_bio = []
    for tag in tags:
        if term in tag:
            if tag != prev:
                term_bio.append("B")
                prev = tag
            else:
                term_bio.append("I")
        else:
            term_bio.append("O")
    return term_bio

def handle_relation(relation):
    start_bracket = relation.index('[')
    end_bracket = relation.index(']')
    polarity = relation[:start_bracket]
    pair = relation[start_bracket+1:end_bracket]
    sent, aspect = pair.split('_')
    return polarity, sent, aspect

def generate_bio_token(tokens, term_tags, term):
    return [token + "//" + term_tag for token, term_tag in zip(
        tokens, get_all_term(term_tags, term)
    )]

def check_sentence_pack(term_dict):
    return all(term_dict.values())

def convert_json(tokens, tags):
    sentence = " ".join(tokens)
    triple = []
    term_tags = []
    relation_tags = []
    for tag in tags:
        tag = tag.split("|")
        term_tag = tag[0] if len(tag) == 1 else extract_term_tag(tag)
        # If term tag exist
        if term_tag != None:
            term_tags.append(term_tag)
            tag.pop(tag.index(term_tag))
            relation_tags += tag

    # All Terms
    all_aspect_tags = " ".join(generate_bio_token(tokens, term_tags, "ASPECT"))
    all_sent_tags = " ".join(generate_bio_token(tokens, term_tags, "SENTIMENT"))

    # Handle Triple
    term_dict = {term_tag : False for term_tag in term_tags if term_tag != "_"}

    def handle_triple(tokens, term_tags, relation_tag):
        polarity, sent_num, aspect_num = handle_relation(relation_tag)
        sent_term = f"SENTIMENT[{sent_num}]"
        aspect_term = f"ASPECT[{aspect_num}]"
        term_dict[aspect_term] = True
        term_dict[sent_term] = True
        aspect_tags = " ".join(generate_bio_token(tokens, term_tags, aspect_term))
        sent_tags = " ".join(generate_bio_token(tokens, term_tags, sent_term))
        return {
            "aspect_tags" : aspect_tags,
            "sent_tags" : sent_tags,
            "polarity" : polarity
        }
    
    for relation_tag in relation_tags:
        triple.append(handle_triple(tokens, term_tags, relation_tag))
    
    valid = check_sentence_pack(term_dict) and len(triple) != 0

    return {
        "sentence" : sentence,
        "aspect_tags" : all_aspect_tags,
        "sent_tags" : all_sent_tags,
        "triples" : triple,
        "valid" : valid
    }
    
def check_output_dir(args):
    path = args.prefix + args.output + args.dataset + "/"
    if not os.path.exists(path):
        os.mkdir(path)

def write_file(json_data, filename):
    with open(filename, 'w') as outfile:
        json.dump(json_data, outfile)

def parse_all_file(args, check_validity = False, show_statistic = True):
    input_path_prefix = args.prefix + args.input + args.dataset + "/"

    output_path_prefix = args.prefix + args.output + args.dataset + "/"

    files = os.listdir(input_path_prefix)
    for file in files:
        in_filename = input_path_prefix + file
        out_filename = output_path_prefix + file.replace(".tsv",".json")

        data, labels = load_data(in_filename)
        json_data = []
        for tokens, tags in zip(data, labels):
            data = convert_json(tokens, tags)
            if check_validity and data['valid'] == False:
                continue
            json_data.append(data)
        
        if show_statistic : statistic(file, json_data, args)
        write_file(json_data, out_filename)


def remove_unvalid_data(args, show_statistic=True):
    input_path_prefix = args.prefix + args.input + args.dataset + "/"
    output_path_prefix = args.prefix + args.output + args.dataset + "/"

    files = os.listdir(input_path_prefix)
    for file in files:
        in_filename = input_path_prefix + file
        out_filename = output_path_prefix + file.replace(".tsv",".json")
        
        data = [data for data in load_data_json(in_filename) if data["valid"]]
        if show_statistic : statistic(file, data,args)
        write_file(data, out_filename)


def statistic(filename, data,args):
    '''
    Displaying Data Statistic
    '''
    n_sentence_pack = len(data)

    def calculate_n_term(tags):
        return sum(1 for tag in tags.split(' ') if tag.endswith('b'))
    
    sentence_length = []

    n_aspect_term, n_sent_term, n_triple = 0, 0, 0
    triple_polarity = {}

    for sentence_pack in data:
        n_aspect_term += calculate_n_term(sentence_pack['aspect_tags'])
        n_sent_term += calculate_n_term(sentence_pack['sent_tags'])
        n_triple += len(sentence_pack['triples'])
        sentence_length.append(len(sentence_pack['sentence'].split(' ')))
        for triple in sentence_pack['triples']:
            if triple['polarity'] not in triple_polarity:
                triple_polarity[triple['polarity']] = 0
            triple_polarity[triple['polarity']]+=1
    
    sentence_length_max = max(sentence_length)
    sentence_length_min = min(sentence_length)
    sentence_length_mean = np.mean(sentence_length)
    
    print(f"Number sentence pack \t: {n_sentence_pack}")
    print(f"Number Aspect Term \t: {n_aspect_term}")
    print(f"Number Sentiment Term \t: {n_sent_term}")
    print(f"Number Triple \t\t: {n_triple}")
    print(f"Number Triple Polarity \t: {triple_polarity}")
    print(f"Sentence Length max \t: {sentence_length_max}")
    print(f"Sentence Length min \t: {sentence_length_min}")
    print(f"Sentence Length mean \t: {sentence_length_mean}")

    figure_out = args.prefix + args.figures + "sentence_length.png"
    plt.hist(sentence_length)
    plt.savefig(figure_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default="../../data/", help='data location')

    parser.add_argument('--input', type=str, default="raw/", help="input data location")

    parser.add_argument('--output', type=str, default="interim/", help="output data locaiton")

    parser.add_argument('--dataset', type=str, default="dummy", help="dataset")

    parser.add_argument('--mode', type=str, default="parse_all", choices=['parse_all', 'parse_valid_only', 'remove_unvalid_data_json'])

    parser.add_argument('--figures', type=str, default="../figures/", help="figures location from prefix")

    args = parser.parse_args()
    if args.mode == "parse_all":
        check_output_dir(args)
        parse_all_file(args)
    elif args.mode == "parse_valid_only":
        check_output_dir(args)
        parse_all_file(args, True)
    elif args.mode == "remove_unvalid_data_json":
        check_output_dir(args)
        remove_unvalid_data(args)

