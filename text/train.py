import os
import random
import argparse
import numpy as np
import torch
from data import build_dataloader
from loss import make_loss
from model.backbone_prompt import make_model
from optimizer import make_optimizer
from optimizer.scheduler_factory import WarmupCosineScheduler,create_scheduler
from processor import do_train
from utils.logger import setup_logger
from config import cfg

torch.set_num_threads(16)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True

# Reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():

    parser = argparse.ArgumentParser(description="CLIP + CCA ReID Training")
    parser.add_argument("--config_file", default="", type=str,
                        help="path to config file")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="Modify config options from command line")
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local rank for DistributedDataParallel")
    args = parser.parse_args()

    # Load config
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)

    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Set Seed
    set_seed(cfg.SOLVER.SEED)

    # Set CUDA env
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.DEVICE_ID)

    # Init distributed mode
    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )

    # Prepare output folder + Logger
    if cfg.OUTPUT_DIR and not os.path.exists(cfg.OUTPUT_DIR):
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    logger = setup_logger("CLIP_CCA_ReID", cfg.OUTPUT_DIR, if_train=True)
    logger.info(f"Saving output to: {cfg.OUTPUT_DIR}")
    logger.info(args)

    if args.config_file != "":
        logger.info(f"Loaded configuration file {args.config_file}")
        with open(args.config_file, 'r') as cf:
            logger.info("\n" + cf.read())

    logger.info("Running with config:\n{}".format(cfg))

    # Build dataloader
    dl_ret = build_dataloader(cfg)

    if cfg.DATA.DATASET == 'prcc':
        (
            trainloader,
            queryloader_same,
            queryloader_diff,
            galleryloader,
            dataset,
            train_sampler,
            val_loader,
            val_loader_same
        ) = dl_ret
    else:
        (
            trainloader,
            queryloader,
            galleryloader,
            dataset,
            train_sampler,
            val_loader
        ) = dl_ret

    num_classes = dataset.num_train_pids
    camera_num = 1
    view_num = 1

    # Build Model
    model = make_model(cfg,num_classes, camera_num, view_num)

    # optionally load pretrained weights
    if cfg.MODEL.PRETRAIN_PATH:
        logger.info(f"Loading pretrained weights from {cfg.MODEL.PRETRAIN_PATH}")
        model.load_pretrained(cfg.MODEL.PRETRAIN_PATH)

    # Build Losses
    loss_func = make_loss(cfg, num_classes=dataset.num_train_pids,device="cuda")
    center_criterion = None

    # Optimizer
    optimizer = make_optimizer(cfg, model)
    optimizer_center = None

    # Scheduler
    scheduler = create_scheduler(cfg, optimizer)

    # Training
    if cfg.DATA.DATASET == 'prcc':
        # PRCC 需要 same/diff 两种 query
        do_train(
            cfg=cfg,
            model=model,
            center_criterion=center_criterion,
            train_loader=trainloader,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            scheduler=scheduler,
            loss_fn=loss_func,
            local_rank=args.local_rank,
            dataset=dataset,
            val_loader=val_loader,
            val_loader_same=val_loader_same
        )
    else:
        do_train(
            cfg=cfg,
            model=model,
            center_criterion=center_criterion,
            train_loader=trainloader,
            optimizer=optimizer,
            optimizer_center=optimizer_center,
            scheduler=scheduler,
            loss_fn=loss_func,
            local_rank=args.local_rank,
            dataset=dataset,
            val_loader=val_loader
        )


if __name__ == '__main__':
    main()