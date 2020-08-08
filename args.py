import argparse


def get_parser():
    parser = argparse.ArgumentParser(description='model parameters')
    # data
    parser.add_argument(
        '--word2vec', default='../dataset/GoogleNews-vectors-negative300.bin')
    parser.add_argument('--category', default='instrument')
    parser.add_argument('--dir', default='../dataset/%s')
    parser.add_argument('--graph_data', default='graph_info')
    parser.add_argument('--train_data', default='%s.train')
    parser.add_argument('--valid_data', default='%s.valid')
    parser.add_argument('--test_data', default='%s.test')
    parser.add_argument('--para_data', default='%s.para')
    parser.add_argument('--max_rating', default=5, type=int)

    # training
    parser.add_argument('--seed', default=2019, type=int)
    parser.add_argument('--device', default=1, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--checkpoint', default='./ckpt/', type=str)
    parser.add_argument('--maxCkpt', default=3, type=int)
    parser.add_argument('--restore', default='', type=str)

    parser.add_argument('--lr', default=0.002, type=float)
    parser.add_argument('--lr_update', default=20, type=int)

    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--gradclip', default=-1.0, type=float)

    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--valfreq', default=1, type=int)

    # hyper
    parser.add_argument('--l2_lamb', default=1.0, type=float)

    parser.add_argument('--emb_dim', default=300, type=int)
    parser.add_argument('--id_dim', default=32, type=int)
    parser.add_argument('--att_dim', default=32, type=int)
    parser.add_argument('--n_latent', default=8, type=int)
    parser.add_argument('--gru_dim', default=32, type=int)
    parser.add_argument('--time_dim', default=32, type=int)
    parser.add_argument('--filter_sizes', default='3', type=str)
    parser.add_argument('--batchsize', default=100, type=int)
    parser.add_argument('--num_filters', default=100, type=int)
    parser.add_argument('--keep_prob', default=1.0, type=float)
    parser.add_argument('--train_clip', default=0, type=int)
    parser.add_argument('--test_clip', default=1, type=int)

    parser.add_argument('--alpha', default=0.25, type=float)
    parser.add_argument('--n_hops', default=2, type=int)
    parser.add_argument('--n_heads', default=8, type=int)
    parser.add_argument('--node_dim', default=128, type=int)
    parser.add_argument('--hid_dim', default=64, type=int)
    parser.add_argument('--graph_att_dim', default=32, type=int)
    parser.add_argument('--sample_num', default=10, type=int)

    parser.add_argument('--percentile', default=10, type=int)
    parser.add_argument('--max_rel', default=100, type=int)

    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--decov_lambda', default=0.01, type=float)
    return parser
