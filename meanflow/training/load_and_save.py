# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path

import torch
from training.distributed_mode import is_main_process

import logging


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def save_model(
    args, epoch, model_without_ddp, optimizer, lr_schedule,
    checkpoint_dir=None, step=None,
):
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir is not None else Path(args.output_dir)
    to_save = {
        "model": model_without_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_schedule": lr_schedule.state_dict(),
        "epoch": epoch,
        "step": step,
        "args": args,
    }
    save_on_master(to_save, ckpt_dir / "checkpoint-last.pth")
    if step is not None:
        ckpt_every = getattr(args, "ckpt_every", 0)
        if ckpt_every > 0 and step % (10 * ckpt_every) == 0:
            save_on_master(to_save, ckpt_dir / f"checkpoint-{step}.pth")
    else:
        if (epoch + 1) % 1000 == 0:
            save_on_master(to_save, ckpt_dir / f"checkpoint-{epoch}.pth")


def load_model(args, model_without_ddp, optimizer, lr_schedule):
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"], strict=False)
        logging.info("Resume checkpoint %s" % args.resume)
        if (
            "optimizer" in checkpoint
            and "epoch" in checkpoint
            and not args.eval_only
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_schedule.load_state_dict(checkpoint["lr_schedule"])
            args.start_epoch = checkpoint["epoch"] + 1
            if "step" in checkpoint and checkpoint["step"] is not None:
                args.start_step = checkpoint["step"]
            logging.info(f"Start epoch set to {args.start_epoch}")
            logging.info("With optim & sched!")
