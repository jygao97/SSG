import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.autograd import Variable
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.models as models

import os, sys, time
import random
import numpy as np

from args import get_parser
import FullModel
from data import MyDataset

parser = get_parser()
opts = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.device)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_vocab(vocab_path, vocabulary, embedding_dim):
    initW = np.random.uniform(-1.0, 1.0, (len(vocabulary), embedding_dim))
    start = time.time()
    with open(vocab_path, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)

            if word in vocabulary:
                idx = vocabulary[word]
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return initW


def main():
    dataset = MyDataset(
        para_path=os.path.join(opts.dir % opts.category,
                               opts.para_data % opts.category),
        graph_path=os.path.join(opts.dir % opts.category, opts.graph_data),
        train_path=os.path.join(opts.dir % opts.category,
                                opts.train_data % opts.category),
        valid_path=os.path.join(opts.dir % opts.category,
                                opts.valid_data % opts.category),
        test_path=os.path.join(opts.dir % opts.category,
                               opts.test_data % opts.category),
        n_hops=opts.n_hops,
        max_rating=opts.max_rating,
        sample_num=opts.sample_num,
        percentile=opts.percentile,
        max_rel=opts.max_rel)

    model = FullModel.MultiView(
        review_num_u=dataset.review_num_u,
        review_num_i=dataset.review_num_i,
        review_len_u=dataset.review_len_u,
        review_len_i=dataset.review_len_i,
        graph_seq_len=dataset.review_len_g,
        user_vocab_size=len(dataset.vocabulary_user),
        item_vocab_size=len(dataset.vocabulary_item),
        graph_vocab_size=len(dataset.vocabulary),
        emb_size=opts.emb_dim,
        filter_sizes=list(map(int, opts.filter_sizes.split(","))),
        num_filters=opts.num_filters,
        user_num=dataset.user_num,
        item_num=dataset.item_num,
        hid_dim=opts.hid_dim,
        id_emb=opts.id_dim,
        att_dim=opts.att_dim,
        graph_att_dim=opts.graph_att_dim,
        gru_size=opts.gru_dim,
        time_dim=opts.time_dim,
        n_latent=opts.n_latent,
        node_num=dataset.node_num,
        node_emb=opts.node_dim,
        n_hops=opts.n_hops,
        max_rating=opts.max_rating,
        n_heads=opts.n_heads,
        alpha=opts.alpha,
        beta=opts.beta,
        decov_lambda=opts.decov_lambda,
        keep_prob=opts.keep_prob,
        l2_lambda=opts.l2_lamb)
    model.cuda()

    if opts.word2vec:
        u_initW = read_vocab(opts.word2vec, dataset.vocabulary_user,
                             opts.emb_dim)
        i_initW = read_vocab(opts.word2vec, dataset.vocabulary_item,
                             opts.emb_dim)
        initW = read_vocab(opts.word2vec, dataset.vocabulary, opts.emb_dim)
        model.user_remb.weight.data.copy_(torch.tensor(u_initW))
        model.item_remb.weight.data.copy_(torch.tensor(i_initW))
        model.graph_cnn.rembedding.weight.data.copy_(torch.tensor(initW))
        print("load dict done")

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=opts.lr,
        eps=opts.eps,
        weight_decay=opts.weight_decay)

    train_loader, valid_loader, test_loader = dataset.get_loaders(
        opts.batchsize, opts.workers)

    data_time = 0.0
    batch_time = 0.0
    valid_time = 0.0
    start_time = time.time()
    end = time.time()

    best_rmse = float('inf')
    best_mae = float('inf')
    rmse_valid_list = []
    mae_valid_list = []
    rmse_test_list = []
    mae_test_list = []

    start_epoch = 0
    for cur_epoch in range(start_epoch, opts.epochs):
        data_t, batch_t, train_rmse, train_mae = train_epoch(
            model, train_loader, optimizer, cur_epoch)
        data_time += data_t
        batch_time += batch_t

        if (cur_epoch + 1) % opts.valfreq == 0:
            end = time.time()
            valid_rmse, valid_mae = validate(model, valid_loader)
            test_rmse, test_mae = validate(model, test_loader)
            print('train: rmse %f mae %f' % (train_rmse, train_mae))
            print('valid: rmse %f mae %f' % (valid_rmse, valid_mae))
            print('test: rmse %f mae %f' % (test_rmse, test_mae))
            rmse_valid_list.append(valid_rmse)
            mae_valid_list.append(valid_mae)
            rmse_test_list.append(test_rmse)
            mae_test_list.append(test_mae)
            valid_time += time.time() - end

        print('data_time:', data_time, 'batch_time:', batch_time,
              'valid_time:', valid_time)
    print(rmse_valid_list)
    print(mae_valid_list)
    print(rmse_test_list)
    print(mae_test_list)


def train_epoch(model, loader, optimizer, epoch):
    print('epoch:', epoch)

    data_time = 0.0
    batch_time = 0.0
    end = time.time()

    model.train()
    rmse_acc = 0
    mae_acc = 0
    train_num_acc = 0

    for i, (nodes, reviews, ratings, adj, pairs, uids, iids, input_u, input_i,
            reuid, reiid, u_s_renum, i_s_renum, u_pos_ind, i_pos_ind, u_rel_dt,
            i_rel_dt, u_abs_dt, i_abs_dt, ys) in enumerate(loader):
        data_time += time.time() - end
        end = time.time()
        train_num = input_u.size(0)

        input_u_var = Variable(input_u).cuda()
        input_i_var = Variable(input_i).cuda()
        reuid_var = Variable(reuid).cuda()
        reiid_var = Variable(reiid).cuda()
        u_s_renum_var = Variable(u_s_renum).cuda()
        i_s_renum_var = Variable(i_s_renum).cuda()

        u_pos_ind_var = Variable(u_pos_ind).cuda()
        i_pos_ind_var = Variable(i_pos_ind).cuda()
        u_rel_dt_var = Variable(u_rel_dt).cuda()
        i_rel_dt_var = Variable(i_rel_dt).cuda()
        u_abs_dt_var = Variable(u_abs_dt).cuda()
        i_abs_dt_var = Variable(i_abs_dt).cuda()

        nodes_var = Variable(nodes).cuda()
        reviews_var = Variable(reviews).cuda()
        ratings_var = Variable(ratings).cuda()
        adj_var = Variable(adj).cuda()
        pairs_var = Variable(pairs).cuda()
        uids_var = Variable(uids).cuda()
        iids_var = Variable(iids).cuda()
        ys_var = Variable(ys).cuda()

        optimizer.zero_grad()
        loss, mae, acc, _ = model(
            input_u_var, input_i_var, reuid_var, reiid_var, u_s_renum_var,
            i_s_renum_var, u_pos_ind_var, i_pos_ind_var, u_rel_dt_var,
            i_rel_dt_var, u_abs_dt_var, i_abs_dt_var, nodes_var, reviews_var,
            ratings_var, adj_var, pairs_var, uids_var, iids_var, ys_var,
            opts.train_clip)
        loss.backward()

        mae_acc += train_num * float(mae.data.cpu().numpy())
        rmse_acc += train_num * (float(acc.data.cpu().numpy())**2)
        train_num_acc += train_num

        if opts.gradclip > 0:
            params = []
            for group in optimizer.param_groups:
                for p in group['params']:
                    params.append(p)
            torch.nn.utils.clip_grad_norm_(params, opts.gradclip)

        optimizer.step()

        if (i % 100) == 0:
            print(loss.item(), 'epoch', epoch, 'batch', i, 'finish')

        batch_time += time.time() - end
        end = time.time()

    rmse = np.sqrt(rmse_acc / train_num_acc)
    mae = mae_acc / train_num_acc

    return data_time, batch_time, rmse, mae


def validate(model, loader):
    model.eval()

    rmse_acc = 0
    mae_acc = 0
    test_num_acc = 0

    with torch.no_grad():
        for i, (nodes, reviews, ratings, adj, pairs, uids, iids, input_u,
                input_i, reuid, reiid, u_s_renum, i_s_renum, u_pos_ind,
                i_pos_ind, u_rel_dt, i_rel_dt, u_abs_dt, i_abs_dt,
                ys) in enumerate(loader):
            test_num = uids.size(0)

            input_u_var = Variable(input_u).cuda()
            input_i_var = Variable(input_i).cuda()
            reuid_var = Variable(reuid).cuda()
            reiid_var = Variable(reiid).cuda()
            u_s_renum_var = Variable(u_s_renum).cuda()
            i_s_renum_var = Variable(i_s_renum).cuda()

            u_pos_ind_var = Variable(u_pos_ind).cuda()
            i_pos_ind_var = Variable(i_pos_ind).cuda()
            u_rel_dt_var = Variable(u_rel_dt).cuda()
            i_rel_dt_var = Variable(i_rel_dt).cuda()
            u_abs_dt_var = Variable(u_abs_dt).cuda()
            i_abs_dt_var = Variable(i_abs_dt).cuda()

            nodes_var = Variable(nodes).cuda()
            reviews_var = Variable(reviews).cuda()
            ratings_var = Variable(ratings).cuda()
            adj_var = Variable(adj).cuda()
            pairs_var = Variable(pairs).cuda()
            uids_var = Variable(uids).cuda()
            iids_var = Variable(iids).cuda()
            ys_var = Variable(ys).cuda()

            loss, mae, acc, _ = model(
                input_u_var, input_i_var, reuid_var, reiid_var, u_s_renum_var,
                i_s_renum_var, u_pos_ind_var, i_pos_ind_var, u_rel_dt_var,
                i_rel_dt_var, u_abs_dt_var, i_abs_dt_var, nodes_var,
                reviews_var, ratings_var, adj_var, pairs_var, uids_var,
                iids_var, ys_var, opts.test_clip)
            mae_acc += test_num * float(mae.data.cpu().numpy())
            rmse_acc += test_num * (float(acc.data.cpu().numpy())**2)
            test_num_acc += test_num

    rmse = np.sqrt(rmse_acc / test_num_acc)
    mae = mae_acc / test_num_acc
    return rmse, mae


if __name__ == '__main__':
    setup_seed(opts.seed)
    print(opts)
    main()
