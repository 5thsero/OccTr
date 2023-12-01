# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import random
# import warnings

import numpy as np
import torch
from torch import distributed as dist
# from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info
# from mmcv.utils import build_from_cfg

from .mmdet_train import custom_train_detector
from mmseg.apis import train_segmentor
from mmdet.apis import train_detector

# TODO 加入了init_random_seed  set_random_seed
def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, optional): The seed. Default to None.
        device (str, optional): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def custom_train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                eval_model=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        assert False
    else:
        custom_train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            eval_model=eval_model,
            meta=meta)


def train_model(model,
                dataset,
                cfg,
                distributed=False,
                validate=False,
                timestamp=None,
                meta=None):
    """A function wrapper for launching model training according to cfg.

    Because we need different eval_hook in runner. Should be deprecated in the
    future.
    """
    if cfg.model.type in ['EncoderDecoder3D']:
        train_segmentor(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
    else:
        train_detector(
            model,
            dataset,
            cfg,
            distributed=distributed,
            validate=validate,
            timestamp=timestamp,
            meta=meta)
