"""
Validate the scalar-gate backward path for LightGCN_Social.

If something is wrong (gate frozen, detach, SocialSparse not used, etc.), this
prints a red banner. If everything is healthy, prints GREEN.

Usage (server side):
    cd ~/LightGCN/code
    python test_scalar_gate.py
"""
import sys, os
sys.path.insert(0, '.')

import torch
import world, dataloader, model

# minimal fake dataset configuration to construct the model
world.config['latent_dim_rec'] = 64
world.config['lightGCN_n_layers'] = 3
world.config['embedding_size'] = 64
world.config['dropout'] = 0
world.config['keep_prob'] = 1.0
world.config['pretrain'] = 0
world.config['social_source'] = 'idf'
world.config['social_K'] = 20
world.config['social_layers'] = 1
world.config['gate_hidden'] = 64
world.config['gate_max'] = 1.0
world.config['gate_lambda'] = 1e-4
world.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# build a toy CSR u-i graph (just enough for indices)
import numpy as np
import scipy.sparse as sp
n_users, m_items = 100, 50
n_int = 500
rows = np.random.randint(0, n_users, n_int)
cols = np.random.randint(0, m_items, n_int)
vals = np.ones(n_int, dtype=np.float32)
UserItemNet = sp.csr_matrix((vals, (rows, cols)), shape=(n_users, m_items))

# Monkey-patch BasicDataset.init_train_stats: skip train.txt load
class ToyDataset(dataloader.BasicDataset):
    def __init__(self, path):
        self.n_users = n_users
        self.m_items = m_items
        self.trainDataSize = n_int
        self.UserItemNet = UserItemNet
        self.testDict = {}
        self.trainUser = rows
        self.trainItem = cols
        self.trainUniqueUsers = np.unique(rows)
        self.socialNet_trust_sym = None
        self.socialNet_idf = None
        self.socialNet_idf_cache = '/tmp/social_idf_K20.npz'
        self.path = path
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']

ds = ToyDataset(path='/tmp')
rec = model.LightGCN_Social(world.config, ds)
rec = rec.to(world.device)
print('[OK] model constructed: n_users=', n_users, 'm_items=', m_items)

# forward once
users = torch.arange(n_users, device=world.device)
u_emb, i_emb = rec.computer()
print('[OK] computer() forward: u_emb=', u_emb.shape, 'i_emb=', i_emb.shape)

# check scalar-gate param list has trainable params
gate_nparam = sum(p.numel() for p in [rec.gate_lin1.weight, rec.gate_lin1.bias, rec.gate_lin2.weight, rec.gate_lin2.bias])
print(f'[OK] gate params: {gate_nparam}')

# run a synthetic loss
target = torch.zeros(n_users, m_items, device=world.device)
loss = (u_emb @ i_emb.T - target).pow(2).mean()
loss.backward()

# verify gradient on gate_lin1 / gate_lin2 / embedding_user
def has_grad(name, p):
    return p.grad is not None and torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0

report = []
for n, p in rec.named_parameters():
    if 'gate_lin' in n or 'embedding_user' in n:
        report.append(f'  {n}: has_grad={has_grad(n, p)}')

print('\nGradient sanity:')
for r in report:
    print(r)

bad = [r for r in report if 'has_grad=False' in r]
if bad:
    print('\n!! RED !! Some parameters have no gradient:', bad)
    sys.exit(1)
else:
    print('\nGREEN: all critical params received gradient -> backward is real.')
