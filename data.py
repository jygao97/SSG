import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import BatchSampler

import os, sys, copy
import random
import pickle as pkl
import numpy as np


def getSubGraph(target_uid,
                target_iid,
                neigh_list,
                user_num,
                n_hops=2,
                sample_num=10):
    target_ids = set()
    for uid in target_uid:
        target_ids.add(uid)
    for iid in target_iid:
        target_ids.add(iid + user_num)
    target_edges = set()
    for k in range(n_hops):
        cur_list = list(target_ids)
        for target_id in cur_list:

            n_list = copy.deepcopy(neigh_list[target_id])
            random.shuffle(n_list)
            if len(n_list) > sample_num:
                n_list = n_list[:sample_num]

            for next_id, next_edge in n_list:
                target_ids.add(next_id)
                target_edges.add(next_edge)
    sub_neigh_list = {}
    for target_id in target_ids:
        sub_neigh_list[target_id] = []
        for next_id, edge_idx in neigh_list[target_id]:
            if next_id in target_ids and edge_idx in target_edges:
                sub_neigh_list[target_id].append((next_id, edge_idx))
    return sub_neigh_list, target_ids


class PartitionDataset(data.Dataset):
    def __init__(self,
                 partition,
                 data_path,
                 u_text,
                 i_text,
                 u_time,
                 i_time,
                 percentile=10,
                 max_rel=100):
        with open(data_path, 'rb') as f:
            self.data = pkl.load(f)

        self.u_text = u_text
        self.i_text = i_text
        self.u_time = u_time
        self.i_time = i_time

        self.percentile = percentile
        self.max_rel = max_rel

    def time_handler(self, times, cur_time):
        times = list(times.reshape(-1))
        renum = 0
        pos_ind = []
        rel_dt = []
        abs_dt = []
        for t in times:
            if t < cur_time:
                dt = cur_time - t
                rel_dt.append(dt)
                abs_dt.append(dt)
                pos_ind.append(renum)
                renum += 1
            else:
                rel_dt.append(0)
                abs_dt.append(0)
                pos_ind.append(0)

        for i in range(len(pos_ind)):
            if i < renum:
                pos_ind[i] = renum - pos_ind[i]

        dt = []
        for i in range(len(rel_dt)):
            if i == 0:
                continue
            dt.append(rel_dt[i - 1] - rel_dt[i])
            if rel_dt[i] == 0:
                break
        dt = np.array(dt)

        pos_ind = np.array(pos_ind)
        rel_dt = np.array(rel_dt)
        abs_dt = np.array(abs_dt)

        dt_nonzero = np.delete(dt, np.where(dt == 0))
        m = max(np.percentile(dt_nonzero, self.percentile), 1.0)
        for i in range(len(rel_dt)):
            rel_dt[i] = min((rel_dt[i] // m), self.max_rel)
        return renum, pos_ind, rel_dt, abs_dt

    def __getitem__(self, index):
        uid, iid, reuid, reiid, y, time = self.data[index]
        uid = int(uid)
        iid = int(iid)
        y = float(y)

        input_u = self.u_text[uid]
        input_i = self.i_text[iid]
        time = int(time)

        u_times = self.u_time[uid]

        u_s_renum, u_pos_ind, u_rel_dt, u_abs_dt = self.time_handler(
            u_times, time)

        i_times = self.i_time[iid]
        i_s_renum, i_pos_ind, i_rel_dt, i_abs_dt = self.time_handler(
            i_times, time)

        return uid, iid, input_u, input_i, reuid, reiid, u_s_renum, i_s_renum, u_pos_ind, i_pos_ind, u_rel_dt, i_rel_dt, u_abs_dt, i_abs_dt, y

    def __len__(self):
        return len(self.data)


class MyDataset(object):
    def __init__(self,
                 para_path,
                 graph_path,
                 train_path,
                 valid_path,
                 test_path,
                 n_hops=2,
                 max_rating=5,
                 sample_num=10,
                 percentile=10,
                 max_rel=100):
        self.para = pkl.load(open(para_path, 'rb'))
        self.user_num = self.para['user_num']
        self.item_num = self.para['item_num']
        self.node_num = self.user_num + self.item_num

        self.review_num_u = self.para['review_num_u']
        self.review_num_i = self.para['review_num_i']
        self.review_len_u = self.para['review_len_u']
        self.review_len_i = self.para['review_len_i']
        self.review_len_g = self.review_len_u
        self.vocabulary_user = self.para['user_vocab']
        self.vocabulary_item = self.para['item_vocab']

        self.u_text = self.para['u_text']
        self.i_text = self.para['i_text']
        self.u_time = self.para['u_time']
        self.i_time = self.para['i_time']

        self.n_hops = n_hops
        self.max_rating = max_rating
        self.sample_num = sample_num
        self.percentile = percentile
        self.max_rel = max_rel
        self.neigh_list, self.edge_id1, self.edge_id2, self.edge_ratings, self.edge_reviews, self.vocabulary = pkl.load(
            open(graph_path, 'rb'))

        self.train_dataset = PartitionDataset('train', train_path, self.u_text,
                                              self.i_text, self.u_time,
                                              self.i_time, percentile, max_rel)
        self.valid_dataset = PartitionDataset('valid', valid_path, self.u_text,
                                              self.i_text, self.u_time,
                                              self.i_time, percentile, max_rel)
        self.test_dataset = PartitionDataset('test', test_path, self.u_text,
                                             self.i_text, self.u_time,
                                             self.i_time, percentile, max_rel)

    def collate_fn(self, data_in):
        uids, iids, input_u, input_i, reuid, reiid, u_s_renum, i_s_renum, u_pos_ind, i_pos_ind, u_rel_dt, i_rel_dt, u_abs_dt, i_abs_dt, ys = zip(
            *data_in)
        sub_neigh_list, target_ids = getSubGraph(
            uids,
            iids,
            self.neigh_list,
            self.user_num,
            n_hops=self.n_hops,
            sample_num=self.sample_num)
        target_ids = list(target_ids)
        nodes = torch.Tensor(target_ids).long()
        adj = torch.zeros(len(target_ids), len(target_ids))
        tmp_reviews = []
        tmp_ratings = []
        nodes_map = {x: i for i, x in enumerate(target_ids)}
        for node in target_ids:
            neighs = sub_neigh_list[node]
            tmp_neighs = []
            for (neigh, edge) in neighs:
                i = nodes_map[node]
                j = nodes_map[neigh]
                tmp_neighs.append((j, edge))
            tmp_neighs = sorted(tmp_neighs, key=lambda x: x[0])
            for (j, edge) in tmp_neighs:
                adj[i, j] = 1.0
                tmp_reviews.append(self.edge_reviews[edge])
                tmp_ratings.append(self.edge_ratings[edge] - 1)

        reviews = torch.Tensor(tmp_reviews).long()
        # one hot
        ratings = np.zeros((len(tmp_ratings), self.max_rating))
        ratings[np.arange(len(tmp_ratings)), tmp_ratings] = 1.0
        ratings = torch.Tensor(ratings).float()

        pairs = torch.Tensor([(nodes_map[uids[i]],
                               nodes_map[iids[i] + self.user_num])
                              for i in range(len(uids))]).long()

        uids = torch.Tensor(uids).long()
        iids = torch.Tensor([i + self.user_num for i in iids]).long()
        ys = torch.Tensor(ys).float()

        input_u = torch.Tensor(input_u).long()
        input_i = torch.Tensor(input_i).long()
        reuid = torch.Tensor(reuid).long()
        reiid = torch.Tensor(reiid).long()
        u_s_renum = torch.Tensor(u_s_renum).long()
        i_s_renum = torch.Tensor(i_s_renum).long()

        u_pos_ind = torch.Tensor(u_pos_ind).long()
        i_pos_ind = torch.Tensor(i_pos_ind).long()
        u_rel_dt = torch.Tensor(u_rel_dt).long()
        i_rel_dt = torch.Tensor(i_rel_dt).long()
        u_abs_dt = torch.Tensor(u_abs_dt).float()
        i_abs_dt = torch.Tensor(i_abs_dt).float()

        return nodes, reviews, ratings, adj, pairs, uids, iids, input_u, input_i, reuid, reiid, u_s_renum, i_s_renum,\
                    u_pos_ind, i_pos_ind, u_rel_dt, i_rel_dt, u_abs_dt, i_abs_dt, ys

    def get_loaders(self, batch_size, workers):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=True,
            collate_fn=self.collate_fn)

        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=self.collate_fn)

        test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            collate_fn=self.collate_fn)

        return train_loader, valid_loader, test_loader
