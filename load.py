import os
import json
import pandas as pd
import pickle
import numpy as np
import time
import datetime

st_time = time.time()
dataset_category = "instrument"
dataset_filename = "Musical_Instruments_5.json"
TPS_DIR = '../dataset/{}'.format(dataset_category)
TAR_DIR = '../dataset/{}'.format(dataset_category)
TP_file = os.path.join(TPS_DIR, dataset_filename)

f = open(TP_file)
users_id = []
items_id = []
ratings = []
reviews = []
times = []
np.random.seed(2017)


# Given a date str "01 01, 2020", return days from 1970.01.1
def str_to_days(s):
    st_date = datetime.date(1970, 1, 1)
    cur_date = datetime.date(
        int(s.split(', ')[1]), int(s.split(', ')[0].split(' ')[0]),
        int(s.split(', ')[0].split(' ')[1]))
    return (cur_date - st_date).days


for line in f:
    js = json.loads(line)
    if str(js['reviewerID']) == 'unknown':
        print("unknown")
        continue
    if str(js['asin']) == 'unknown':
        print("unknown2")
        continue
    reviews.append(js['reviewText'])
    users_id.append(str(js['reviewerID']) + ',')
    items_id.append(str(js['asin']) + ',')
    ratings.append(str(js['overall']))
    times.append(str_to_days(js['reviewTime']))
data = pd.DataFrame({
    'user_id': pd.Series(users_id),
    'item_id': pd.Series(items_id),
    'ratings': pd.Series(ratings),
    'reviews': pd.Series(reviews),
    'times': pd.Series(times)
})[['user_id', 'item_id', 'ratings', 'reviews', 'times']]


def get_count(tp, id):
    playcount_groupbyid = tp[[id, 'ratings']].groupby(id, as_index=False)
    count = playcount_groupbyid.size()
    return count


usercount, itemcount = get_count(data, 'user_id'), get_count(data, 'item_id')

unique_uid = usercount.index
unique_sid = itemcount.index
item2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
user2id = dict((uid, i) for (i, uid) in enumerate(unique_uid))


def numerize(tp):
    uid = list(map(lambda x: user2id[x], tp['user_id']))
    sid = list(map(lambda x: item2id[x], tp['item_id']))
    tp['user_id'] = uid
    tp['item_id'] = sid
    return tp


data = numerize(data)
tp_rating = data[['user_id', 'item_id', 'ratings', 'times']]

n_ratings = tp_rating.shape[0]
test = np.random.choice(n_ratings, size=int(0.20 * n_ratings), replace=False)
test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_1 = tp_rating[test_idx]
tp_train = tp_rating[~test_idx]
edge = data[~test_idx]

data2 = data[test_idx]
data = data[~test_idx]

n_ratings = tp_1.shape[0]
test = np.random.choice(n_ratings, size=int(0.50 * n_ratings), replace=False)

test_idx = np.zeros(n_ratings, dtype=bool)
test_idx[test] = True

tp_test = tp_1[test_idx]
tp_valid = tp_1[~test_idx]
edge.to_csv(
    os.path.join(TAR_DIR, '{}_edge.csv'.format(dataset_category)),
    index=False,
    header=None)
tp_train.to_csv(
    os.path.join(TAR_DIR, '{}_train.csv'.format(dataset_category)),
    index=False,
    header=None)
tp_valid.to_csv(
    os.path.join(TAR_DIR, '{}_valid.csv'.format(dataset_category)),
    index=False,
    header=None)
tp_test.to_csv(
    os.path.join(TAR_DIR, '{}_test.csv'.format(dataset_category)),
    index=False,
    header=None)

user_meta = {}
item_meta = {}

for i in data.values:
    if i[0] in user_meta:
        user_meta[i[0]].append((i[1], i[3], i[4]))
    else:
        user_meta[i[0]] = [(i[1], i[3], i[4])]
    if i[1] in item_meta:
        item_meta[i[1]].append((i[0], i[3], i[4]))
    else:
        item_meta[i[1]] = [(i[0], i[3], i[4])]

user_reviews = {}
item_reviews = {}
user_rid = {}
item_rid = {}

for u in user_meta:
    user_meta[u] = sorted(user_meta[u], key=lambda x: x[2])
    user_reviews[u] = [(i[1], i[2]) for i in user_meta[u]]
    user_rid[u] = [i[0] for i in user_meta[u]]
for i in item_meta:
    item_meta[i] = sorted(item_meta[i], key=lambda x: x[2])
    item_reviews[i] = [(x[1], x[2]) for x in item_meta[i]]
    item_rid[i] = [x[0] for x in item_meta[i]]

for i in data2.values:
    if i[0] in user_reviews:
        l = 1
    else:
        print("one new user")
        user_rid[i[0]] = [-1]
        user_reviews[i[0]] = [('<PAD/>', '01 01, 2020')]
        # no corresponding training data for this user
    if i[1] in item_reviews:
        l = 1
    else:
        print("one new item")
        item_rid[i[1]] = [-1]
        # no corresponding training data for this item
        item_reviews[i[1]] = [('<PAD/>', '01 01, 2020')]

pickle.dump(user_reviews, open(os.path.join(TAR_DIR, 'user_review'), 'wb'))
pickle.dump(item_reviews, open(os.path.join(TAR_DIR, 'item_review'), 'wb'))
pickle.dump(user_rid, open(os.path.join(TAR_DIR, 'user_rid'), 'wb'))
pickle.dump(item_rid, open(os.path.join(TAR_DIR, 'item_rid'), 'wb'))
pickle.dump(user2id, open(os.path.join(TAR_DIR, 'user2id'), 'wb'))
pickle.dump(item2id, open(os.path.join(TAR_DIR, 'item2id'), 'wb'))

ed_time = time.time()
print("Whole process takes {} s".format(ed_time - st_time))
