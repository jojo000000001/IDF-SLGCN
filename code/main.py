import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)

# Trigger social graph build now so the first epoch is not slowed by it
if hasattr(Recmodel, '_get_social_graph') and getattr(Recmodel, 'enable_social', False):
    print("building social graph (one-shot) ...")
    _ = Recmodel._get_social_graph()
    if hasattr(Recmodel, 'last_gate_mean'):
        print(f"initial gate mean (random init -> bias=+1): {Recmodel.last_gate_mean:.4f}")

bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        gate_info = f" gate~{Recmodel.last_gate_mean:.3f}" if hasattr(Recmodel, 'last_gate_mean') and getattr(Recmodel, 'enable_social', False) else ""
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}{gate_info}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()