import numpy as np
import re
import itertools
import time
from collections import Counter
import datetime
import tensorflow as tf
import csv
import pickle
import os


# Given a date str "01 01, 2020", return days from 1970.01.1
def str_to_days(s):
    st_date = datetime.date(1970, 1, 1)
    cur_date = datetime.date(
        int(s.split(', ')[1]), int(s.split(', ')[0].split(' ')[0]),
        int(s.split(', ')[0].split(' ')[1]))
    return (cur_date - st_date).days


dataset_category = "instrument"
tf.flags.DEFINE_string("valid_data", "../dataset/{}/{}_valid.csv".format(
    dataset_category, dataset_category), " Data for validation")
tf.flags.DEFINE_string("test_data", "../dataset/{}/{}_test.csv".format(
    dataset_category, dataset_category), "Data for testing")
tf.flags.DEFINE_string("train_data", "../dataset/{}/{}_train.csv".format(
    dataset_category, dataset_category), "Data for training")
tf.flags.DEFINE_string("user_review",
                       "../dataset/{}/user_review".format(dataset_category),
                       "User's reviews")
tf.flags.DEFINE_string("item_review",
                       "../dataset/{}/item_review".format(dataset_category),
                       "Item's reviews")
tf.flags.DEFINE_string("user_review_id",
                       "../dataset/{}/user_rid".format(dataset_category),
                       "user_review_id")
tf.flags.DEFINE_string("item_review_id",
                       "../dataset/{}/item_rid".format(dataset_category),
                       "item_review_id")
tf.flags.DEFINE_string("stopwords", "../dataset/stopwords", "stopwords")


def clean_str(string):
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


def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len
    review_len = u2_len

    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri][0]
                time_str = u_reviews[ri][1]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append((new_sentence, time_str))
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append((new_sentence, time_str))
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append((new_sentence,
                                       str_to_days('01 01, 2020')))
        full_empty = ([padding_word] * review_len, 0)
        padded_u_train.insert(0, full_empty)
        u_text2[i] = padded_u_train

    return u_text2


def pad_reviewid(u_train, u_valid, u_test, u_len, num):
    pad_u_train = []

    for i in range(len(u_train)):
        x = u_train[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        x.insert(0, num)
        pad_u_train.append(x)
    pad_u_valid = []

    for i in range(len(u_valid)):
        x = u_valid[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        x.insert(0, num)
        pad_u_valid.append(x)
    pad_u_test = []

    for i in range(len(u_test)):
        x = u_test[i]
        while u_len > len(x):
            x.append(num)
        if u_len < len(x):
            x = x[:u_len]
        x.insert(0, num)
        pad_u_test.append(x)
    return pad_u_train, pad_u_valid, pad_u_test


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    u_text2 = {}
    u_time = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = np.array(
            [[vocabulary_u[word] for word in words[0]] for words in u_reviews])
        t = np.array([[words[1]] for words in u_reviews])
        u_text2[i] = u
        u_time[i] = t
    i_text2 = {}
    i_time = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = np.array(
            [[vocabulary_i[word] for word in words[0]] for words in i_reviews])
        t = np.array([[words[1]] for words in i_reviews])
        i_text2[j] = i
        i_time[j] = t
    return u_text2, i_text2, u_time, i_time


def load_data(train_data, valid_data, test_data, user_review, item_review,
              user_rid, item_rid, stopwords):
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    u_text, i_text, y_train, y_valid, y_test, time_train, time_valid, time_test, u_len, i_len, u2_len, i2_len, uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num, item_num \
        , reid_user_train, reid_item_train, reid_user_valid, reid_item_valid, reid_user_test, reid_item_test = \
        load_data_and_labels(train_data, valid_data, test_data, user_review, item_review, user_rid, item_rid, stopwords)
    print("load data done")
    u_text = pad_sentences(u_text, u_len, u2_len)
    reid_user_train, reid_user_valid, reid_user_test = pad_reviewid(
        reid_user_train, reid_user_valid, reid_user_test, u_len, item_num + 1)

    print("pad user done")
    i_text = pad_sentences(i_text, i_len, i2_len)
    reid_item_train, reid_item_valid, reid_item_test = pad_reviewid(
        reid_item_train, reid_item_valid, reid_item_test, i_len, user_num + 1)

    print("pad item done")

    user_voc = [xx[0] for x in u_text.values() for xx in x]
    item_voc = [xx[0] for x in i_text.values() for xx in x]

    vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(
        user_voc, item_voc)
    print(len(vocabulary_user))
    print(len(vocabulary_item))
    u_text, i_text, u_time, i_time = build_input_data(
        u_text, i_text, vocabulary_user, vocabulary_item)
    y_train = np.array(y_train)
    y_valid = np.array(y_valid)
    y_test = np.array(y_test)
    time_train = np.array(time_train)
    time_valid = np.array(time_valid)
    time_test = np.array(time_test)
    uid_train = np.array(uid_train)
    uid_valid = np.array(uid_valid)
    uid_test = np.array(uid_test)
    iid_train = np.array(iid_train)
    iid_valid = np.array(iid_valid)
    iid_test = np.array(iid_test)
    reid_user_train = np.array(reid_user_train)
    reid_user_valid = np.array(reid_user_valid)
    reid_user_test = np.array(reid_user_test)
    reid_item_train = np.array(reid_item_train)
    reid_item_valid = np.array(reid_item_valid)
    reid_item_test = np.array(reid_item_test)

    return [
        u_text, i_text, u_time, i_time, y_train, y_valid, y_test, time_train,
        time_valid, time_test, vocabulary_user, vocabulary_inv_user,
        vocabulary_item, vocabulary_inv_item, uid_train, iid_train, uid_valid,
        iid_valid, uid_test, iid_test, user_num, item_num, reid_user_train,
        reid_item_train, reid_user_valid, reid_item_valid, reid_user_test,
        reid_item_test
    ]


def load_data_and_labels(train_data, valid_data, test_data, user_review,
                         item_review, user_rid, item_rid, stopwords):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files

    f_train = open(train_data, "r")
    f1 = open(user_review, 'rb')
    f2 = open(item_review, 'rb')
    f3 = open(user_rid, 'rb')
    f4 = open(item_rid, 'rb')

    user_reviews = pickle.load(f1)
    item_reviews = pickle.load(f2)
    user_rids = pickle.load(f3)
    item_rids = pickle.load(f4)
    user_num = len(user_reviews)
    item_num = len(item_reviews)
    print("user_num:", user_num)
    print("item_num:", item_num)

    reid_user_train = []
    reid_item_train = []
    uid_train = []
    iid_train = []
    y_train = []
    time_train = []
    u_text = {}
    u_rid = {}
    i_text = {}
    i_rid = {}
    i = 0
    for line in f_train:
        i = i + 1
        line = line.split(',')
        uid_train.append(int(line[0]))
        iid_train.append(int(line[1]))
        if int(line[0]) in u_text:
            reid_user_train.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = []
            for (s, time) in user_reviews[int(line[0])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                u_text[int(line[0])].append((s1, time))
            u_rid[int(line[0])] = []
            for s in user_rids[int(line[0])]:
                u_rid[int(line[0])].append(int(s))
            reid_user_train.append(u_rid[int(line[0])])

        if int(line[1]) in i_text:
            reid_item_train.append(i_rid[int(line[1])])  #####write here
        else:
            i_text[int(line[1])] = []
            for (s, time) in item_reviews[int(line[1])]:
                s1 = clean_str(s)
                s1 = s1.split(" ")
                i_text[int(line[1])].append((s1, time))
            i_rid[int(line[1])] = []
            for s in item_rids[int(line[1])]:
                i_rid[int(line[1])].append(int(s))
            reid_item_train.append(i_rid[int(line[1])])
        y_train.append(float(line[2]))
        time_train.append(int(line[3]))
    print("valid")
    reid_user_valid = []
    reid_item_valid = []

    uid_valid = []
    iid_valid = []
    y_valid = []
    time_valid = []
    f_valid = open(valid_data)
    for line in f_valid:
        line = line.split(',')
        uid_valid.append(int(line[0]))
        iid_valid.append(int(line[1]))
        if int(line[0]) in u_text:
            reid_user_valid.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [(['<PAD/>'], str_to_days('01 01, 2020'))]
            u_rid[int(line[0])] = [item_num + 1]
            reid_user_valid.append(u_rid[int(line[0])])

        if int(line[1]) in i_text:
            reid_item_valid.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = [(['<PAD/>'], str_to_days('01 01, 2020'))]
            i_rid[int(line[1])] = [user_num + 1]
            reid_item_valid.append(i_rid[int(line[1])])

        y_valid.append(float(line[2]))
        time_valid.append(int(line[3]))
    print("test")
    reid_user_test = []
    reid_item_test = []

    uid_test = []
    iid_test = []
    y_test = []
    time_test = []
    f_test = open(test_data)
    for line in f_test:
        line = line.split(',')
        uid_test.append(int(line[0]))
        iid_test.append(int(line[1]))
        if int(line[0]) in u_text:
            reid_user_test.append(u_rid[int(line[0])])
        else:
            u_text[int(line[0])] = [(['<PAD/>'], '01 01, 2020')]
            u_rid[int(line[0])] = [item_num + 1]
            reid_user_test.append(u_rid[int(line[0])])

        if int(line[1]) in i_text:
            reid_item_test.append(i_rid[int(line[1])])
        else:
            i_text[int(line[1])] = [(['<PAD/>'], '01 01, 2020')]
            i_rid[int(line[1])] = [user_num + 1]
            reid_item_test.append(i_rid[int(line[1])])

        y_test.append(float(line[2]))
        time_test.append(int(line[3]))

    review_num_u = np.array([len(x) for x in u_text.values()])
    x = np.sort(review_num_u)
    u_len = x[int(0.9 * len(review_num_u)) - 1]
    review_len_u = np.array([len(j[0]) for i in u_text.values() for j in i])
    x2 = np.sort(review_len_u)
    u2_len = x2[int(0.9 * len(review_len_u)) - 1]

    review_num_i = np.array([len(x) for x in i_text.values()])
    y = np.sort(review_num_i)
    i_len = y[int(0.9 * len(review_num_i)) - 1]
    review_len_i = np.array([len(j[0]) for i in i_text.values() for j in i])
    y2 = np.sort(review_len_i)
    i2_len = y2[int(0.9 * len(review_len_i)) - 1]
    print("u_len:", u_len)
    print("i_len:", i_len)
    print("u2_len:", u2_len)
    print("i2_len:", i2_len)
    return [
        u_text, i_text, y_train, y_valid, y_test, time_train, time_valid,
        time_test, u_len, i_len, u2_len, i2_len, uid_train, iid_train,
        uid_valid, iid_valid, uid_test, iid_test, user_num, item_num,
        reid_user_train, reid_item_train, reid_user_valid, reid_item_valid,
        reid_user_test, reid_item_test
    ]


if __name__ == '__main__':
    st_time = time.time()
    TPS_DIR = '../dataset/{}'.format(dataset_category)
    FLAGS = tf.flags.FLAGS

    u_text, i_text, u_time, i_time, y_train, y_valid, y_test, time_train, time_valid, time_test, vocabulary_user, vocabulary_inv_user, vocabulary_item, \
    vocabulary_inv_item, uid_train, iid_train, uid_valid, iid_valid, uid_test, iid_test, user_num, item_num, reid_user_train, reid_item_train, reid_user_valid, reid_item_valid, reid_user_test, reid_item_test = \
        load_data(FLAGS.train_data, FLAGS.valid_data, FLAGS.test_data, FLAGS.user_review, FLAGS.item_review, FLAGS.user_review_id,
                  FLAGS.item_review_id, FLAGS.stopwords)

    np.random.seed(2017)

    shuffle_indices = np.random.permutation(np.arange(len(y_train)))

    userid_train = uid_train[shuffle_indices]
    itemid_train = iid_train[shuffle_indices]
    y_train = y_train[shuffle_indices]
    time_train = time_train[shuffle_indices]
    reid_user_train = reid_user_train[shuffle_indices]
    reid_item_train = reid_item_train[shuffle_indices]

    y_train = y_train[:, np.newaxis]
    y_valid = y_valid[:, np.newaxis]
    y_test = y_test[:, np.newaxis]
    time_train = time_train[:, np.newaxis]
    time_valid = time_valid[:, np.newaxis]
    time_test = time_test[:, np.newaxis]

    userid_train = userid_train[:, np.newaxis]
    itemid_train = itemid_train[:, np.newaxis]

    userid_valid = uid_valid[:, np.newaxis]
    itemid_valid = iid_valid[:, np.newaxis]

    userid_test = uid_test[:, np.newaxis]
    itemid_test = iid_test[:, np.newaxis]

    batches_train = list(
        zip(userid_train, itemid_train, reid_user_train, reid_item_train,
            y_train, time_train))
    batches_valid = list(
        zip(userid_valid, itemid_valid, reid_user_valid, reid_item_valid,
            y_valid, time_valid))
    batches_test = list(
        zip(userid_test, itemid_test, reid_user_test, reid_item_test, y_test,
            time_test))
    print('write begin')
    output = open(
        os.path.join(TPS_DIR, '{}.train'.format(dataset_category)), 'wb')
    pickle.dump(batches_train, output, protocol=2)
    output = open(
        os.path.join(TPS_DIR, '{}.valid'.format(dataset_category)), 'wb')
    pickle.dump(batches_valid, output, protocol=2)
    output = open(
        os.path.join(TPS_DIR, '{}.test'.format(dataset_category)), 'wb')
    pickle.dump(batches_test, output, protocol=2)

    para = {}
    para['user_num'] = user_num
    para['item_num'] = item_num
    print(u_text[0].shape)
    para['review_num_u'] = u_text[0].shape[0]
    para['review_num_i'] = i_text[0].shape[0]
    para['review_len_u'] = u_text[0].shape[1]
    para['review_len_i'] = i_text[0].shape[1]
    para['user_vocab'] = vocabulary_user
    para['item_vocab'] = vocabulary_item
    para['train_length'] = len(y_train)
    para['valid_length'] = len(y_valid)
    para['test_length'] = len(y_test)
    para['u_text'] = u_text
    para['i_text'] = i_text
    para['u_time'] = u_time
    para['i_time'] = i_time
    output = open(
        os.path.join(TPS_DIR, '{}.para'.format(dataset_category)), 'wb')

    # Pickle dictionary using protocol 0.
    pickle.dump(para, output, protocol=2)
    print("Whole process takes {} s".format(time.time() - st_time))
