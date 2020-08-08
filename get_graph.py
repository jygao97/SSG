import pickle as pkl
import os
import re
from collections import defaultdict, Counter
import itertools
import pandas as pd
dataset_category = "instrument"
dataset_filename = "Musical_Instruments_5.json"
TPS_DIR = '../dataset/{}'.format(dataset_category)
para = pkl.load(
    open(os.path.join(TPS_DIR, '{}.para'.format(dataset_category)), 'rb'))
user_num = para['user_num']
review_len = para['review_len_u']
edge_reviews = []
edge_ratings = []
edge_id1 = []
edge_id2 = []
neigh_list = defaultdict(list)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def process_line(l, review_len):
    line = l.strip().split(',')
    uid = int(line[0])
    iid = int(line[1])
    rating = int(float(line[2]))
    raw_review = ''.join(line[3:-1])
    review = clean_str(raw_review).split(' ')
    cur_len = len(review)
    if cur_len > review_len:
        review = review[:review_len]
    else:
        review.extend(['<PAD/>'] * (review_len - cur_len))
    return uid, iid, rating, review


def process_text(raw_review, review_len):
    review = clean_str(raw_review).split(' ')
    cur_len = len(review)
    if cur_len > review_len:
        review = review[:review_len]
    else:
        review.extend(['<PAD/>'] * (review_len - cur_len))
    return review


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary


edge_file = open(
    os.path.join(TPS_DIR, '{}_edge.csv'.format(dataset_category)), 'r')
edge_idx = 0
for l in edge_file:
    uid, iid, rating, review = process_line(l, review_len)
    edge_id1.append(uid)
    edge_id2.append(iid)
    edge_ratings.append(rating)
    edge_reviews.append(review)
    neigh_list[uid].append((iid + user_num, edge_idx))
    neigh_list[iid + user_num].append((uid, edge_idx))
    edge_idx += 1
edge_file.close()

vocabulary = build_vocab(edge_reviews)
edge_reviews = [[vocabulary[word] for word in review]
                for review in edge_reviews]

pkl.dump(
    (neigh_list, edge_id1, edge_id2, edge_ratings, edge_reviews, vocabulary),
    open(os.path.join(TPS_DIR, 'graph_info'), 'wb'))
