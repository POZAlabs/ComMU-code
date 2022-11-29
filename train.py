import argparse
import contextlib
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

from logger import logger
from commu.model.config_helper import get_default_cfg_training
from commu.model.dataset import ComMUDataset
from commu.model.exp_utils import logging_config
from commu.model.model import MemTransformerLM

@contextlib.contextmanager
def sync_workers(args):
    """
    Yields distributed rank and synchronizes all workers on exit.
    """
    yield args.local_rank
    dist.barrier()


def save_checkpoint(
        args,
        model,
        optimizer,
        vocab,
        train_step,
        best_val_loss,
        scheduler,
        name="checkpoint.pt",
):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "train_step": train_step,
        "scheduler": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "vocab": vocab,
    }

    checkpoint["amp"] = None

    # with sync_workers(args) as rank:
    path = os.path.join(args.work_dir, name)
    logger.info(f"Saving checkpoint to {path}")
    # if rank == 0:
    torch.save(checkpoint, path)


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Transformer Language Model")
    parser.add_argument(
        "--data_dir", type=str, required=True, help="location of the data corpus"
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Base directory to save the trained model.",
    )
    args = parser.parse_args()
    return args



def evaluate(eval_iter):
    # Turn on evaluation mode def disables dropout.
    model.eval()

    eval_model = model

    eval_model.reset_length(
        tgt_len=cfg.EVALUATE.tgt_length, mem_len=cfg.EVALUATE.mem_length)
    eval_model.same_length = True
    # Evaluation
    total_token_num = 0
    total_nll = 0.0

    with torch.no_grad():
        mems = None

        for i, (data, target, all_reset_mem, batch_token_num) in enumerate(eval_iter()):

            if all_reset_mem:
                mems = None

            ret = model(data, target, None, mems)
            loss, mems = ret
            loss = loss[target != dataset.vocab.pad_id]
            loss = loss.mean()
            total_nll += batch_token_num * loss.float().item()
            total_token_num += batch_token_num

    eval_model.reset_length(cfg.TRAIN.tgt_length, cfg.TRAIN.mem_length)
    eval_model.same_length = cfg.MODEL.same_length

    model.train()

    return total_token_num, total_nll


def train():
    global train_step
    global best_val_nll

    log_train_loss = torch.tensor(0.0).float().to(device)
    log_grad_norm = torch.tensor(0.0).float().to(device)
    log_token_num = torch.tensor(0).to(device)

    log_start_time = time.time()

    mems = [None for _ in range(cfg.TRAIN.batch_chunk)]

    assert batch_size % cfg.TRAIN.batch_chunk == 0
    train_real_iter = train_iter()

    for batch, (data, target, reset_mems, batch_token_num) in enumerate(
            train_real_iter
    ):
        model.temperature = 1.0

        model.zero_grad()


        data_chunks = torch.chunk(data, cfg.TRAIN.batch_chunk, 1)
        target_chunks = torch.chunk(target, cfg.TRAIN.batch_chunk, 1)
        reset_mems_chunks = torch.chunk(reset_mems, cfg.TRAIN.batch_chunk, 0)
        for i in range(cfg.TRAIN.batch_chunk):

            data = data_chunks[i].contiguous()
            target = target_chunks[i].contiguous()
            reset_mems = reset_mems_chunks[i].contiguous()

            ret = model(data, target, reset_mems, mems[i])
            loss, mems[i] = ret

            loss = loss[target != dataset.vocab.pad_id]
            loss = loss.float().mean() / cfg.TRAIN.batch_chunk
            log_train_loss += (
                    loss.item()
                    * (target != dataset.vocab.pad_id).sum()
                    * cfg.TRAIN.batch_chunk
            )
            loss.backward()

        log_token_num += int(batch_token_num)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.TRAIN.clip
        )

        log_grad_norm += grad_norm
        optimizer.step()
        optimizer.zero_grad()

        # step-wise learning rate annealing
        train_step += 1
        scheduler.step()

        if train_step % cfg.TRAIN.log_interval == 0:
            # torch.distributed.all_reduce(log_train_loss)
            # torch.distributed.all_reduce(log_grad_norm)
            # torch.distributed.all_reduce(log_token_num)

            log_train_loss /= log_token_num
            log_grad_norm /= cfg.TRAIN.log_interval * num_gpus
            if args.local_rank == 0:
                elapsed = time.time() - log_start_time
                logger.info(
                    "Train Step {}/{}, lr={:f}, tokens/s={:.1f},"
                    " nll={:.4f}, ppl={:.2f}, grad norm={}, ".format(
                        train_step,
                        cfg.TRAIN.max_step,
                        optimizer.param_groups[0]["lr"],
                        log_token_num.item() / elapsed,
                        log_train_loss.item(),
                        math.exp(log_train_loss.item()),
                        log_grad_norm.item(),
                    )
                )

            log_train_loss[()] = 0
            log_grad_norm[()] = 0
            log_token_num[()] = 0

            log_start_time = time.time()

        if train_step % cfg.TRAIN.eval_interval == 0:
            eval_start_time = time.time()

            val_token_num, val_total_nll = evaluate(
                eval_iter=val_iter
            )

            val_token_num_pt = torch.tensor(val_token_num).to(device)
            val_total_nll_pt = torch.tensor(val_total_nll / 10000.0).to(device)

            # torch.distributed.all_reduce(val_token_num_pt)
            # torch.distributed.all_reduce(val_total_nll_pt)

            val_token_num = val_token_num_pt.item()
            val_total_nll = val_total_nll_pt.item()

            val_nll = val_total_nll / (val_token_num / 10000.0)

            if args.local_rank == 0:
                logger.info(
                    "Eval step {}, time={}s, val nll={}, val ppl={},".format(
                        train_step,
                        time.time() - eval_start_time,
                        val_nll,
                        math.exp(val_nll),
                        val_token_num,
                    )
                )

            name = "checkpoint_last.pt"
            save_checkpoint(
                args,
                model,
                optimizer,
                dataset.vocab,
                train_step,
                val_nll,
                scheduler,
                name,
            )

            if not best_val_nll or val_nll < best_val_nll:
                best_val_nll = val_nll

                name = "checkpoint_best.pt"
                save_checkpoint(
                    args,
                    model,
                    optimizer,
                    dataset.vocab,
                    train_step,
                    best_val_nll,
                    scheduler,
                    name,
                )

                test_start_time = time.time()

                def calculate_test_nll_during_training(test_iter):

                    test_token_num, test_total_nll = evaluate(
                        eval_iter=test_iter
                    )
                    test_token_num_pt = torch.tensor(test_token_num).to(device)
                    test_total_nll_pt = torch.tensor(test_total_nll / 10000.0).to(device)
                    # torch.distributed.all_reduce(test_token_num_pt)
                    # torch.distributed.all_reduce(test_total_nll_pt)

                    test_token_num = test_token_num_pt.item()
                    test_nll = test_total_nll_pt.item() / (test_token_num / 10000.0)

                    return test_token_num, test_nll

                test_token_num, test_nll = calculate_test_nll_during_training(test_iter)

                if args.local_rank == 0:
                    logger.info(
                        "Test step {}, time={}s, test nll={}, test ppl={}, #evaluated tokens={}".format(
                            train_step,
                            time.time() - test_start_time,
                            test_nll,
                            math.exp(test_nll),
                            test_token_num,
                        )
                    )

        if train_step == cfg.TRAIN.max_step:
            logger.info("-" * 100)
            logger.info("End of training")
            break


def init_weight(weight):
    init_std = cfg.INITIALIZER.base_init
    nn.init.normal_(weight, 0.0, init_std)


def init_embed(weight):
    init_std = cfg.INITIALIZER.embed_init
    nn.init.normal_(weight, 0.0, init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("AdaptiveEmbedding") != -1:
        if hasattr(m, "emb_projs"):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    init_embed(m.emb_projs[i])
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, "out_projs"):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    init_embed(m.out_projs[i])
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, cfg.INITIALIZER.base_init)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        if hasattr(m, "r_emb"):
            init_weight(m.r_emb)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias)
        if hasattr(m, "r_bias"):
            init_bias(m.r_bias)


def update_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        if hasattr(m, "p"):
            m.p = cfg.MODEL.dropout


def update_dropatt(m):
    if hasattr(m, "dropatt"):
        m.dropatt.p = cfg.MODEL.attention_dropout


args = parse_args()
# Modify 2022 11 27
cfg = get_default_cfg_training()


torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

#torch.distributed.init_process_group(backend="nccl", init_method="env://")

exp_time = torch.tensor(time.time(), dtype=torch.float64).to(device)
#torch.distributed.broadcast(exp_time, 0)
exp_time = float(exp_time.cpu().numpy())

args.work_dir = os.path.join(
    args.work_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime(exp_time))
)

os.makedirs(args.work_dir, exist_ok=True)

if args.local_rank == 0:
    with open(os.path.join(args.work_dir, "config.yml"), "w") as f:
        f.write(str(cfg))

if args.local_rank == 0:
    logging_config(args.work_dir, "train_rank{}".format(args.local_rank), console=True)
else:
    logging_config(args.work_dir, "train_rank{}".format(args.local_rank), console=False)

seed = cfg.TRAIN.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

###############################################################################
# Load data
###############################################################################
logger.info("Loading data")
dataset = ComMUDataset(args.data_dir, cfg)
vocab = dataset.vocab

local_seed = cfg.TRAIN.seed + args.local_rank * 1000
num_gpus = torch.cuda.device_count()
assert cfg.TRAIN.batch_size % num_gpus == 0
batch_size = cfg.TRAIN.batch_size // num_gpus

train_iter = dataset.get_iterator(
    batch_size, cfg.TRAIN.tgt_length, device, "train", True, seed=local_seed
)
val_iter = dataset.eval_iterator(
    cfg.EVALUATE.batch_size,
    cfg.EVALUATE.tgt_length,
    device,
    "valid",
    local_rank=args.local_rank,
    world_size=num_gpus,
)
test_iter = dataset.eval_iterator(
    cfg.EVALUATE.batch_size,
    cfg.EVALUATE.tgt_length,
    device,
    "test",
    local_rank=args.local_rank,
    world_size=num_gpus,
)


###############################################################################
# Build the model
###############################################################################

logger.info("Build the model")

assert cfg.MODEL.units % cfg.MODEL.num_heads == 0
model = MemTransformerLM(cfg, vocab)
model.apply(weights_init)
model.word_emb.apply(
    weights_init
)  # ensure embedding init is not overridden by out_layer in case of weight sharing

args.n_all_param = sum([p.nelement() for p in model.parameters()])
args.n_nonemb_param_gen = sum(
    [p.nelement() for p in model.layers.parameters()]
)

model = model.to(device)

# MLE optimizer
local_lr = cfg.TRAIN.lr / num_gpus
optimizer = optim.Adam(model.parameters(), lr=local_lr,
                       weight_decay=cfg.TRAIN.weight_decay)

#### scheduler

# originally used for Transformer (in Attention is all you need)
def lr_lambda(step):
    # return a multiplier instead of a learning rate
    if step == 0 and cfg.TRAIN.warmup_step == 0:
        return 1.0
    else:
        return (
            max(
                (cfg.TRAIN.warmup_step ** 0.5) / (step ** 0.5),
                cfg.TRAIN.lr_min / cfg.TRAIN.lr,
            )
            if step > cfg.TRAIN.warmup_step
            else step / cfg.TRAIN.warmup_step
        )
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


train_step = 0
best_val_nll = np.inf
"""
model = DDP(
    model,
    device_ids=[args.local_rank],
    output_device=args.local_rank,
    broadcast_buffers=False,
    find_unused_parameters=False,
)
"""
logger.info("=" * 100)
logger.info(args)
logger.info("=" * 100)
logger.info("#total params = {}".format(args.n_all_param))
logger.info("#non emb params in generator = {}".format(args.n_nonemb_param_gen))

###############################################################################
# Training code
###############################################################################
logger.info("Start training")

if __name__ == "__main__":
    train()
    # Load the best saved model.
    cfg.defrost()
    cfg.MODEL.same_length = True
    cfg.freeze()
    model = MemTransformerLM(cfg, dataset._vocab)
    checkpoint = torch.load(os.path.join(args.work_dir, "checkpoint_best.pt"))

    model.load_state_dict(checkpoint["model"])
    # Do the evaluation of the best model
    model = model.to(device)

    test_token_num, test_total_nll = evaluate(
        eval_iter=test_iter
    )
    test_token_num_pt = torch.tensor(test_token_num).to(device)
    test_total_nll_pt = torch.tensor(test_total_nll / 10000.0).to(device)
    # torch.distributed.all_reduce(test_token_num_pt)
    # torch.distributed.all_reduce(test_total_nll_pt)
    test_token_num = test_token_num_pt.item()
    test_nll = test_total_nll_pt.item() / (test_token_num / 10000.0)
    logger.info("=" * 100)
    logger.info(
        "| End of training | test nll {:5.2f} | test ppl {:9.3f}".format(
            test_nll, math.exp(test_nll)
        )
    )
    logger.info("=" * 100)