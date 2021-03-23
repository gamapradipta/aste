import tensorflow as tf
from constant import *
import numpy as np
import json

class BaseSentence(object):
    def __init__(self, sentence_pack:dict, tokenizer, args):
        '''
            BaseSentence Class
        '''
        self.args = args
        self.sentence = sentence_pack["sentence"].lower()
        self.tokens = self.sentence.strip().split()
        self.sentence_length = len(self.tokens)
        self.token_ranges = []

        self.input_ids, _, self.attention_mask = tokenizer(
            self.sentence,
            padding='max_length',
            max_length=self.args.max_len
        ).values()
        # self.input_ids = tf.Variable(self.input_ids)
        # self.attention_mask = tf.Variable(self.attention_mask)
        self.length = len(tf.boolean_mask(self.input_ids, self.attention_mask))

        token_begin = 1
        for idx, token in enumerate(self.tokens):
            token_end = token_begin + len(tokenizer.encode(token, add_special_tokens=False))
            self.token_ranges.append([token_begin, token_end-1])
            token_begin = token_end
        # print(self.length, self.token_ranges[-1][-1]+2)
        # print(self.__dict__)
        assert self.length == self.token_ranges[-1][-1]+2
    # TEMPORARY FIX
    def get_X(self):
        return [np.array([self.input_ids]),np.array([self.attention_mask])]

class SentenceExample(BaseSentence):
    def __init__(self, sentence_pack, tokenizer, args):
        super().__init__(sentence_pack, tokenizer, args)
        
        self.gen_label(sentence_pack)

    def get_spans(self, tags, tags_dict=BIO_TAGS):
        tags = tags.strip().split()
        spans = []
        begin = -1
        length = len(tags)
        for i in range(length):
            if tags[i].endswith(tags_dict['begin']):
                if begin != -1:
                    spans.append([begin, i-1])
                begin = i 
            elif tags[i].endswith(tags_dict['other']) and begin != -1:
                spans.append([begin, i-1])
                begin = -1
        if begin != -1:
            spans.append([begin, length-1])
        return spans
    
    def set_label(self, spans, val='aspect'):
        assert spans != [] and spans != ([],[])
        assert self.tags[0][0] == IGNORE_INDEX
        # assert self.tags[1][2] != -1
        if val == 'aspect' or val == 'sentiment':
            for l, r in spans:
                begin = self.token_ranges[l][0]
                end = self.token_ranges[r][-1]
                for i in range(begin, end+1):
                    for j in range(i, end+1):
                        self.tags[i][j] = LABELS[val]
                for i in range(l, r+1):
                    tl, tr = self.token_ranges[i]
                    '''handle sub words'''
                    self.tags[tl+1:tr+1, :] = IGNORE_INDEX
                    self.tags[:, tl+1:tr+1] = IGNORE_INDEX
        elif val in POLARITY:
            assert len(spans) == 2
            a_span, s_span = spans
            for a_begin, a_end in a_span:
                for s_begin, s_end in s_span:
                    for i in range(a_begin, a_end+1):
                        for j in range(s_begin, s_end+1):
                            al, ar = self.token_ranges[i]
                            sl, sr = self.token_ranges[j]
                            self.tags[al:ar+1, sl:sr+1] = IGNORE_INDEX
                            if i > j:
                                self.tags[sl][al] = LABELS[val]
                            else:
                                self.tags[al][sl] = LABELS[val]
        else:
            raise ValueError('Unknown val parameter', val)
        # print(spans,"\n", self.tags)
    
    def handle_triple(self, triple):
        aspect = triple['aspect_tags']
        sentiment = triple['sent_tags']
        polarity = triple['polarity']

        aspect_span = self.get_spans(aspect)
        sentiment_span = self.get_spans(sentiment)

        # set tag for aspect
        self.set_label(aspect_span, "aspect")

        # set tag for sentiment
        self.set_label(sentiment_span,"sentiment")

        self.set_label((aspect_span, sentiment_span), polarity)
    
    def gen_label(self, sentence_pack):
        self.tags = np.zeros((self.args.max_len, self.args.max_len))
        if self.args.exclude_padding:
            self.tags[:,:] = IGNORE_INDEX
            for i in range(1, self.length-1):
                self.tags[i,i:self.length-1] = 0
        
        aspect_span = self.get_spans(sentence_pack["aspect_tags"])
        sentiment_span = self.get_spans(sentence_pack["sent_tags"])

        self.set_label(aspect_span, "aspect")
        self.set_label(sentiment_span, "sentiment")

        for triple in (sentence_pack['triples']):
            self.handle_triple(triple)
        # self.tags = tf.Variable(self.tags)

def create_sentence_example(filename, tokenizer, args):
    sentence_examples = []
    with open(filename) as f:
        raw_data = json.load(f)
        for item in raw_data:
            sentence_examples.append(SentenceExample(item, tokenizer, args))
    return sentence_examples

def create_inputs_targets(sentence_examples):
    dataset_dict = {
        "input_ids": [],
        "attention_mask" : [],
        "tags" : [],
    }

    for item in sentence_examples:
        for key in dataset_dict:
            dataset_dict[key].append(getattr(item, key))

    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    y = [dataset_dict["tags"]]
    return x,y