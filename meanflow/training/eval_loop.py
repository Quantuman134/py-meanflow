# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import gc
import logging
import math
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import PIL.Image

import torch
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from training import distributed_mode
import models.rng as rng

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 10


def eval_model(
    model: DistributedDataParallel,
    net_ema: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    args: Namespace,
    suffix: str = "",
    checkpoint_dir: Path | None = None,
    step: int | None = None,
):
    gc.collect()
    model.train(False)

    if args.distributed:
        data_loader.sampler.set_epoch(0)

    assert args.fid_samples <= len(data_loader.dataset), (
        f"In this interface, dataset size ({len(data_loader.dataset)}) must be larger than FID samples ({args.fid_samples})."
    )
    fid_samples = math.ceil(args.fid_samples / distributed_mode.get_world_size())

    fid_metric = FrechetInceptionDistance(normalize=True).to(device=device, non_blocking=True)

    num_synthetic = 0
    snapshots_saved = False
    all_synthetic_for_is = []
    snapshot_dir = (checkpoint_dir if checkpoint_dir is not None else Path(args.output_dir)) / "snapshots"
    fid_samples_base = checkpoint_dir if checkpoint_dir is not None else Path(args.output_dir)
    if args.output_dir:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    for data_iter_step, (samples, _) in enumerate(data_loader):
        samples = samples.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)  # real is always on the entire dataset

        if num_synthetic < fid_samples:          
            model_without_ddp = model.module if isinstance(model, DistributedDataParallel) else model  

            with torch.random.fork_rng(devices=[device]):
                #per node and per step seed
                torch.manual_seed(rng.fold_in(args.seed, rng.get_rank(), data_iter_step, epoch))
                with torch.amp.autocast('cuda', enabled=False), torch.no_grad():
                    synthetic_samples = model_without_ddp.sample(samples_shape=samples.shape, net=net_ema, device=device)
            torch.cuda.synchronize()

            # Scaling to [0, 1] from [-1, 1]
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)

            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0

            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            num_synthetic += synthetic_samples.shape[0]

            if args.compute_is:
                all_synthetic_for_is.append(synthetic_samples.cpu())

            if not snapshots_saved and args.output_dir:
                label = f"step{step}" if step is not None else str(epoch)
                save_image(
                    synthetic_samples,
                    fp=snapshot_dir / f"{label}_{data_iter_step}{suffix}.png",
                )
                snapshots_saved = True

            if args.save_fid_samples and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = fid_samples_base / f"fid_samples{suffix}"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)

        if not args.compute_fid:
            return {}

        if (data_iter_step + 1) % PRINT_FREQUENCY == 0 or data_iter_step == len(data_loader) - 1:
            # Sync fid metric to ensure that the processes dont deviate much.
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(f"Evaluating: current batch {samples.shape[0]},  [{num_synthetic}/{fid_samples}], running fid {running_fid}")

        if args.test_run:
            break
    
    metrics = {"fid": float(fid_metric.compute().detach().cpu())}

    if args.compute_is and all_synthetic_for_is:
        from training.is_metric import compute_is as compute_inception_score
        all_syn = torch.cat(all_synthetic_for_is, dim=0)
        n_splits = getattr(args, "is_splits", 10)
        if len(all_syn) >= n_splits:
            is_mean, is_std = compute_inception_score(all_syn, device=str(device), splits=n_splits)
            metrics["is_mean"] = is_mean
            metrics["is_std"] = is_std
        else:
            logger.warning(
                f"Not enough synthetic samples ({len(all_syn)}) for IS ({n_splits} splits). Skipping IS."
            )

    return metrics
