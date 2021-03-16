import argparse

class Config(object):
    def __init__(self):
        self.max_len = 18
        self.exclude_padding = True
        self.fine_tuned = True
        self.bert_version = 'indobenchmark/indobert-base-p1'
        self.bert_feature_dim = 768
        self.nhop = 1
        self.class_num = 6
        self.learning_rate = 5e-5