import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

import os, sys, time
import math
import random
import numpy as np


def decov(x, y, diag=False):
    b = x.size(0)
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    x = x - x_mean[None, :]
    y = y - y_mean[None, :]
    mat = (x.t()).mm(y) / b

    decov_loss = 0.5 * torch.norm(mat, p='fro')**2
    if diag:
        decov_loss = decov_loss - 0.5 * torch.norm(torch.diag(mat))**2

    return decov_loss


class TextCNN(nn.Module):
    def __init__(self, seq_len, vocab_size, emb_size, filter_sizes,
                 num_filters):
        super(TextCNN, self).__init__()

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.filter_sizes = filter_sizes
        self.num_filter_sizes = len(filter_sizes)
        self.num_filters = num_filters

        self.rembedding = nn.Embedding(vocab_size, emb_size)

        self.cnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        for s in filter_sizes:
            self.cnns.append(
                nn.Conv2d(1, num_filters, kernel_size=(s, emb_size)))
            self.pools.append(
                nn.MaxPool2d(kernel_size=(seq_len - s + 1, 1), stride=(1, 1)))

        self.out_dim = self.num_filters * self.num_filter_sizes

        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.rembedding.weight, -0.1, 0.1)
        for i in range(self.num_filter_sizes):
            torch.nn.init.uniform_(self.cnns[i].weight, -0.1, 0.1)
            torch.nn.init.constant_(self.cnns[i].bias, 0.1)

    def forward(self, inputs):
        # inputs: (-1, seq_len)
        inputs = self.rembedding(inputs)
        pooled_out = []
        for i in range(self.num_filter_sizes):
            h = F.relu(self.cnns[i](inputs.view(-1, 1, self.seq_len,
                                                self.emb_size).contiguous()))
            pooled = self.pools[i](h)
            pooled_out.append(pooled)
        outputs = torch.cat(pooled_out, 3).view(
            -1, self.num_filters * self.num_filter_sizes)
        return outputs


class TimeAttn(nn.Module):
    def __init__(self, dim, time_dim, beta):
        super(TimeAttn, self).__init__()
        self.beta = beta
        self.temperature = np.sqrt(dim * 1.0)

        self.pos_emb = nn.Embedding(150, time_dim)
        self.rel_emb = nn.Embedding(150, time_dim)

        self.fc_k = nn.Linear(dim, dim, bias=False)
        self.fc_q = nn.Linear(dim, dim, bias=False)
        self.fc_rp = nn.Linear(2 * time_dim, 1, bias=False)

        self.rate = nn.Parameter(torch.ones(1))
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.pos_emb.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.rel_emb.weight, -0.1, 0.1)

        torch.nn.init.uniform_(self.fc_k.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.fc_q.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.fc_rp.weight, -0.1, 0.1)

    def forward(self, out, hn, pos_ind, rel_dt, abs_dt):
        pad_mask = pos_ind == 0

        pos_ind = self.pos_emb(pos_ind)
        rel_dt = self.rel_emb(rel_dt)

        attn_k = self.fc_k(out)  #(b, s, d)
        attn_q = self.fc_q(hn)  #(b,d)

        attn_0 = (torch.bmm(attn_k, attn_q.unsqueeze(-1)).squeeze(-1)
                  ) / self.temperature  #(b, s)
        attn_1 = self.fc_rp(torch.cat([rel_dt, pos_ind],
                                      -1)).squeeze(-1)  #(b, s)
        attn = attn_0 + self.beta * attn_1

        attn = attn.masked_fill(pad_mask, -np.inf)
        attn = F.softmax(attn, 1)

        outputs = torch.bmm(attn.unsqueeze(1), out).squeeze(1)
        return outputs


class gru_module(nn.Module):
    def __init__(self, input_dim, gru_dim, time_dim, beta):
        super(gru_module, self).__init__()
        self.gru = nn.GRU(input_dim, gru_dim, batch_first=True)
        self.attention = TimeAttn(gru_dim, time_dim, beta)

    def forward(self, inputs, length, pos_ind, rel_dt, abs_dt):
        sorted_len, sorted_idx = length.sort(0, descending=True)
        index_sorted_idx = sorted_idx.view(-1, 1, 1).expand_as(inputs)
        sorted_inputs = inputs.gather(0, index_sorted_idx.long())

        inputs_pack = pack_padded_sequence(
            sorted_inputs, sorted_len, batch_first=True)
        out, hn = self.gru(inputs_pack)
        hn = torch.squeeze(hn, 0)
        out, lens_unpacked = pad_packed_sequence(
            out, batch_first=True, total_length=inputs.size(1))

        _, ori_idx = sorted_idx.sort(0, descending=False)
        unsorted_idx = ori_idx.view(-1, 1).expand_as(hn)
        hn = hn.gather(0, unsorted_idx.long())
        unsorted_idx = ori_idx.view(-1, 1, 1).expand_as(out)
        out = out.gather(0, unsorted_idx.long())

        out = self.attention(out, hn, pos_ind, rel_dt, abs_dt)
        return out


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 emb_size,
                 max_rating,
                 att_dim,
                 dropout,
                 alpha,
                 concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        self.a = nn.Parameter(
            torch.zeros(size=(1, 2 * (out_features + att_dim))))
        self.re_W = nn.Parameter(torch.zeros(size=(emb_size, att_dim)))
        self.ra_W = nn.Parameter(torch.zeros(size=(max_rating, att_dim)))

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.W, -0.1, 0.1)
        torch.nn.init.uniform_(self.a, -0.1, 0.1)
        torch.nn.init.uniform_(self.re_W, -0.1, 0.1)
        torch.nn.init.uniform_(self.ra_W, -0.1, 0.1)

    def forward(self, inputs, adj, review, rating):
        dv = 'cuda' if inputs.is_cuda else 'cpu'

        N = inputs.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(inputs, self.W)
        re_h = torch.mm(review, self.re_W)
        ra_h = torch.mm(rating, self.ra_W)
        assert not torch.isnan(h).any()

        edge_h = torch.cat(
            (h[edge[0, :], :], h[edge[1, :], :], re_h, ra_h), dim=1).t()

        edge_e = torch.exp(self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]),
                                     torch.ones(size=(N, 1), device=dv))
        e_rowsum += 1e-10

        edge_e = self.dropout(edge_e)

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime.div(e_rowsum)
        assert not torch.isnan(h_prime).any()

        h_prime = h_prime + h

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class GraphModel(nn.Module):
    def __init__(self, input_size, node_num, node_emb, hid_dim, n_hops,
                 max_rating, att_dim, n_heads, alpha, keep_prob):
        super(GraphModel, self).__init__()

        self.node_embedding = nn.Embedding(node_num, node_emb)

        self.attentions = nn.ModuleList()
        self.n_hops = n_hops
        for i in range(n_hops - 1):
            tmp_att = nn.ModuleList()
            if i == 0:
                in_feat = node_emb
            else:
                in_feat = hid_dim * n_heads
            for _ in range(n_heads):
                tmp_att.append(
                    SpGraphAttentionLayer(
                        in_feat,
                        hid_dim,
                        emb_size=input_size,
                        max_rating=max_rating,
                        att_dim=att_dim,
                        dropout=1.0 - keep_prob,
                        alpha=alpha,
                        concat=True))
            self.attentions.append(tmp_att)

        self.out_att = SpGraphAttentionLayer(
            hid_dim * n_heads,
            hid_dim,
            emb_size=input_size,
            max_rating=max_rating,
            att_dim=att_dim,
            dropout=1.0 - keep_prob,
            alpha=alpha,
            concat=False)

        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.node_embedding.weight, -0.1, 0.1)

    def forward(self, nodes, edge_emb, ratings, adj, pairs):
        node_emb = self.node_embedding(nodes)
        h = node_emb
        for i in range(self.n_hops - 1):
            h = torch.cat(
                [att(h, adj, edge_emb, ratings) for att in self.attentions[i]],
                dim=1)

        out = self.out_att(h, adj, edge_emb, ratings)

        u_feas = out[pairs[:, 0]]
        i_feas = out[pairs[:, 1]]

        return u_feas, i_feas


class squential(nn.Module):
    def __init__(self, input_size, gru_size):
        super(squential, self).__init__()

        self.u_gru = gru_module(input_size, gru_size)
        self.i_gru = gru_module(input_size, gru_size)

    def forward(self, reviews_u, reviews_i, u_s_renum, i_s_renum):
        u_hn = self.u_gru(reviews_u, u_s_renum)
        i_hn = self.i_gru(reviews_i, i_s_renum)
        return u_hn, i_hn


class MultiView(nn.Module):
    def __init__(self,
                 review_num_u,
                 review_num_i,
                 review_len_u,
                 review_len_i,
                 graph_seq_len,
                 user_vocab_size,
                 item_vocab_size,
                 graph_vocab_size,
                 emb_size,
                 filter_sizes,
                 num_filters,
                 user_num,
                 item_num,
                 hid_dim,
                 id_emb,
                 att_dim,
                 graph_att_dim,
                 gru_size,
                 time_dim,
                 n_latent,
                 node_num,
                 node_emb,
                 n_hops,
                 max_rating,
                 n_heads,
                 alpha,
                 beta,
                 decov_lambda,
                 keep_prob=1.0,
                 l2_lambda=0.0):
        super(MultiView, self).__init__()
        self.review_num_u = review_num_u
        self.review_num_i = review_num_i
        self.review_len_u = review_len_u
        self.review_len_i = review_len_i
        self.user_num = user_num
        self.item_num = item_num
        self.user_vocab_size = user_vocab_size
        self.item_vocab_size = item_vocab_size
        self.n_latent = n_latent
        self.id_emb = id_emb
        self.att_dim = att_dim
        self.emb_size = emb_size
        self.filter_sizes = filter_sizes
        self.num_filter_sizes = len(filter_sizes)
        self.num_filters = num_filters
        # self.gru_size = gru_size
        self.decov_lambda = decov_lambda
        self.keep_prob = keep_prob
        self.l2_reg_lambda = l2_lambda

        # user embedding
        self.user_remb = nn.Embedding(user_vocab_size, emb_size)
        self.user_idemb_att = nn.Embedding(user_num + 2, id_emb)

        # item embedding
        self.item_remb = nn.Embedding(item_vocab_size, emb_size)
        self.item_idemb_att = nn.Embedding(item_num + 2, id_emb)
        self.idemb = nn.Embedding(node_num, n_latent)

        # user convolutions
        self.user_cnns = nn.ModuleList()
        self.user_pools = nn.ModuleList()
        for s in filter_sizes:
            self.user_cnns.append(
                nn.Conv2d(1, num_filters, kernel_size=(s, emb_size)))
            self.user_pools.append(
                nn.MaxPool2d(
                    kernel_size=(review_len_u - s + 1, 1), stride=(1, 1)))

        # item convolutions
        self.item_cnns = nn.ModuleList()
        self.item_pools = nn.ModuleList()
        for s in filter_sizes:
            self.item_cnns.append(
                nn.Conv2d(1, num_filters, kernel_size=(s, emb_size)))
            self.item_pools.append(
                nn.MaxPool2d(
                    kernel_size=(review_len_i - s + 1, 1), stride=(1, 1)))

        self.Wau = Parameter(
            torch.Tensor(num_filters * len(filter_sizes), att_dim))
        self.Wru = Parameter(torch.Tensor(id_emb, att_dim))
        self.Wpu = Parameter(torch.Tensor(att_dim, 1))
        self.bau = Parameter(torch.Tensor(att_dim))
        self.bbu = Parameter(torch.Tensor(1))

        self.Wai = Parameter(
            torch.Tensor(num_filters * len(filter_sizes), att_dim))
        self.Wri = Parameter(torch.Tensor(id_emb, att_dim))
        self.Wpi = Parameter(torch.Tensor(att_dim, 1))
        self.bai = Parameter(torch.Tensor(att_dim))
        self.bbi = Parameter(torch.Tensor(1))

        self.u_dropout = torch.nn.Dropout(1.0 - keep_prob)
        self.i_dropout = torch.nn.Dropout(1.0 - keep_prob)

        self.u_gru = gru_module(num_filters * len(filter_sizes), gru_size,
                                time_dim, beta)
        self.i_gru = gru_module(num_filters * len(filter_sizes), gru_size,
                                time_dim, beta)

        self.u_fc = nn.Linear(
            num_filters * len(filter_sizes) + gru_size + hid_dim, n_latent)
        self.i_fc = nn.Linear(
            num_filters * len(filter_sizes) + gru_size + hid_dim, n_latent)

        self.fm_dropout = torch.nn.Dropout(1.0 - keep_prob)

        self.Wmul = Parameter(torch.Tensor(n_latent, 1))
        # self.ubiases = Parameter(torch.Tensor(user_num+2))
        # self.ibiases = Parameter(torch.Tensor(item_num+2))
        self.biases = Parameter(torch.Tensor(node_num))
        self.gbias = Parameter(torch.Tensor(1))

        self.mse_loss = nn.MSELoss()

        self.graph_cnn = TextCNN(graph_seq_len, graph_vocab_size, emb_size,
                                 filter_sizes, num_filters)
        self.graph_view = GraphModel(num_filters * len(filter_sizes), node_num,
                                     node_emb, hid_dim, n_hops, max_rating,
                                     graph_att_dim, n_heads, alpha, keep_prob)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.uniform_(self.user_remb.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.user_idemb_att.weight, -0.1, 0.1)

        torch.nn.init.uniform_(self.item_remb.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.item_idemb_att.weight, -0.1, 0.1)
        torch.nn.init.uniform_(self.idemb.weight, -0.1, 0.1)

        for i in range(self.num_filter_sizes):
            torch.nn.init.uniform_(self.user_cnns[i].weight, -0.1, 0.1)
            torch.nn.init.constant_(self.user_cnns[i].bias, 0.1)

        for i in range(self.num_filter_sizes):
            torch.nn.init.uniform_(self.item_cnns[i].weight, -0.1, 0.1)
            torch.nn.init.constant_(self.item_cnns[i].bias, 0.1)

        torch.nn.init.uniform_(self.Wau, -0.1, 0.1)
        torch.nn.init.uniform_(self.Wru, -0.1, 0.1)
        torch.nn.init.uniform_(self.Wpu, -0.1, 0.1)
        torch.nn.init.constant_(self.bau, 0.1)
        torch.nn.init.constant_(self.bbu, 0.1)

        torch.nn.init.uniform_(self.Wai, -0.1, 0.1)
        torch.nn.init.uniform_(self.Wri, -0.1, 0.1)
        torch.nn.init.uniform_(self.Wpi, -0.1, 0.1)
        torch.nn.init.constant_(self.bai, 0.1)
        torch.nn.init.constant_(self.bbi, 0.1)

        torch.nn.init.uniform_(self.u_fc.weight, -0.1, 0.1)
        torch.nn.init.constant_(self.u_fc.bias, 0.1)

        torch.nn.init.uniform_(self.i_fc.weight, -0.1, 0.1)
        torch.nn.init.constant_(self.i_fc.bias, 0.1)

        torch.nn.init.uniform_(self.Wmul, -0.1, 0.1)

        # torch.nn.init.constant_(self.ubiases, 0.1)
        # torch.nn.init.constant_(self.ibiases, 0.1)
        torch.nn.init.constant_(self.biases, 0.1)
        torch.nn.init.constant_(self.gbias, 0.1)

    def broad_mm(self, x, y):
        # x: (b, m, n)
        # y: (n, p)
        b, m, n = x.size()
        n1, p = y.size()
        assert n == n1
        # try:
        #   torch.mm(x.view(-1, n), y)
        # except:
        #   pass
        return torch.mm(x.view(-1, n), y).view(b, m, p)

    def forward(self, input_u, input_i, reuid, reiid, u_s_renum, i_s_renum,
                u_pos_ind, i_pos_ind, u_rel_dt, i_rel_dt, u_abs_dt, i_abs_dt,
                nodes, reviews, ratings, adj, pairs, uid, iid, y, clip):

        edge_emb = self.graph_cnn(reviews)
        graph_ufeas, graph_ifeas = self.graph_view(nodes, edge_emb, ratings,
                                                   adj, pairs)
        batch_size = input_u.size(0)

        self.embedding_users = self.user_remb(input_u)
        self.embedding_items = self.item_remb(input_i)

        # TextCNN for set view
        pooled_out_u = []
        for i in range(self.num_filter_sizes):
            h = F.relu(self.user_cnns[i](self.embedding_users.view(
                -1, 1, self.review_len_u, self.emb_size)))
            pooled = self.user_pools[i](h)
            pooled_out_u.append(pooled)
        self.h_pool_u = torch.cat(pooled_out_u, 3).view(
            -1, self.review_num_u, self.num_filters * self.num_filter_sizes)

        pooled_out_i = []
        for i in range(self.num_filter_sizes):
            h = F.relu(self.item_cnns[i](self.embedding_items.view(
                -1, 1, self.review_len_i, self.emb_size)))
            pooled = self.item_pools[i](h)
            pooled_out_i.append(pooled)
        self.h_pool_i = torch.cat(pooled_out_i, 3).view(
            -1, self.review_num_i, self.num_filters * self.num_filter_sizes)

        self.reviews_u = self.h_pool_u
        self.reviews_i = self.h_pool_i

        self.iid_a = F.relu(self.item_idemb_att(reuid))
        self.u_j = self.broad_mm(
            F.relu(
                self.broad_mm(self.reviews_u, self.Wau) + self.broad_mm(
                    self.iid_a, self.Wru) + self.bau), self.Wpu) + self.bbu
        self.u_a = F.softmax(self.u_j, 1)

        self.uid_a = F.relu(self.user_idemb_att(reiid))
        self.i_j = self.broad_mm(
            F.relu(
                self.broad_mm(self.reviews_i, self.Wai) + self.broad_mm(
                    self.uid_a, self.Wri) + self.bai), self.Wpi) + self.bbi
        self.i_a = F.softmax(self.i_j, 1)

        self.u_feas = torch.sum(self.reviews_u * self.u_a, 1)
        self.i_feas = torch.sum(self.reviews_i * self.i_a, 1)

        self.u_hn = self.u_gru(self.reviews_u, u_s_renum, u_pos_ind, u_rel_dt,
                               u_abs_dt)
        self.i_hn = self.i_gru(self.reviews_i, i_s_renum, i_pos_ind, i_rel_dt,
                               i_abs_dt)

        decov_loss = 0.0
        decov_loss += decov(self.u_feas, self.u_hn) + decov(
            self.i_feas, self.i_hn)
        decov_loss += decov(self.u_hn, graph_ufeas) + decov(
            self.i_hn, graph_ifeas)
        decov_loss += decov(self.u_feas, graph_ufeas) + decov(
            self.i_feas, graph_ifeas)

        self.u_feas = torch.cat([self.u_feas, self.u_hn, graph_ufeas], dim=1)
        self.i_feas = torch.cat([self.i_feas, self.i_hn, graph_ifeas], dim=1)

        self.u_feas = self.u_fc(self.u_feas)
        self.i_feas = self.i_fc(self.i_feas)
        # self.u_feas = self.u_dropout(self.u_feas)
        # self.i_feas = self.i_dropout(self.i_feas)

        self.uid_emb = self.idemb(uid).view(-1, self.n_latent)
        self.iid_emb = self.idemb(iid).view(-1, self.n_latent)

        self.u_feas = self.u_feas + self.uid_emb
        self.i_feas = self.i_feas + self.iid_emb

        self.FM = F.relu(self.u_feas * self.i_feas)
        # self.FM = self.fm_dropout(self.FM)
        self.mul = torch.matmul(self.FM, self.Wmul)

        self.score = torch.sum(self.mul, 1, keepdim=True)
        #self.score = self.FM.sum(dim=1,keepdim=True)

        self.u_bias = torch.gather(self.biases, 0, uid).view(-1, 1)
        self.i_bias = torch.gather(self.biases, 0, iid).view(-1, 1)
        self.pred = self.score + self.u_bias + self.i_bias + self.gbias

        y = y.float()
        self.pred = self.pred.view(-1)
        if clip > 0:
            self.pred = torch.clamp(self.pred, 1, 5)
        self.loss = 0.5 * self.mse_loss(self.pred, y)
        self.l2_loss = 0.5 * torch.sum(self.Wau**2)
        self.l2_loss = self.l2_loss + 0.5 * torch.sum(self.Wru**2)
        self.l2_loss = self.l2_loss + 0.5 * torch.sum(self.Wai**2)
        self.l2_loss = self.l2_loss + 0.5 * torch.sum(self.Wri**2)
        self.loss = self.loss + self.l2_reg_lambda * self.l2_loss + self.decov_lambda * decov_loss

        self.mae = torch.mean(torch.abs(self.pred - y))
        self.acc = torch.sqrt(torch.mean((self.pred - y)**2))

        return self.loss, self.mae, self.acc, self.pred
