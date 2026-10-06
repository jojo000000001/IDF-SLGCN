'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=64,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn, lgn_social]')
    # --- Social branch (only used when --model lgn_social) ---
    parser.add_argument('--social_source', type=str, default='none',
                        choices=['none', 'idf', 'trust'],
                        help="user-user social graph source: none | idf (IDF-cosine top-K) | trust (lastfm ground-truth)")
    parser.add_argument('--social_K', type=int, default=20,
                        help="top-K neighbors when social_source=idf (default 20)")
    parser.add_argument('--social_layers', type=int, default=1,
                        help="number of propagation steps on the social graph (default 1)")
    parser.add_argument('--gate_hidden', type=int, default=64,
                        help="gate MLP hidden dim (default = latent_dim)")
    parser.add_argument('--gate_max', type=float, default=1.0,
                        help="upper bound clamp on gate value, 1.0 = no clamp (default 1.0)")
    parser.add_argument('--gate_lambda', type=float, default=1e-4,
                        help="L1 penalty on gate values (0 disables, default 1e-4)")
    parser.add_argument('--social_alpha', type=float, default=0.1,
                        help="initial alpha_u for adaptive fusion; gate starts from this value (default 0.1)")
    return parser.parse_args()
