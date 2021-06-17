POLARITY = ('PO', 'NT', 'NG')
LABELS = {'aspect':1, 'sentiment': 2, 'PO': 3, 'NT':4, 'NG' : 5, 'other' : 0}
BIO_TAGS = {'begin': 'B', 'inside': 'I', 'other': 'O'}
BIO_ASPECT_TAGS = {'start': 'B-ASPECT', 'inside': 'I-ASPECT', 'other': 'O'}
BIO_SENTIMENT_TAGS = {'start': 'B-SENTIMENT', 'inside': 'I-SENTIMENT', 'other': 'O'}
IGNORE_INDEX = -1