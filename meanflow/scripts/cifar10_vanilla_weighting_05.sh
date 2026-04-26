#!/bin/bash
set -e

source /scratch/project/prj-02-visual-ai/hkzhang/miniconda3/etc/profile.d/conda.sh
conda activate meanflow
cd /scratch/project/prj-02-visual-ai/hkzhang/py-meanflow/meanflow

NGPU=8
PORT=$(shuf -i 20000-65535 -n 1)
CONFIG="./configs/cifar10_vanilla_weighting_05.yaml"

TRAIN_ARGS=$(python3 - "$CONFIG" <<'PYEOF'
import sys, yaml

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

args = []
for key, val in cfg.items():
    if val is None or val == "":
        continue  # let argparse use its default
    if isinstance(val, bool):
        if val:
            args.append(f"--{key}")
    elif isinstance(val, list):
        args.append(f"--{key}")
        args.extend(str(v) for v in val)
    else:
        args += [f"--{key}", str(val)]

print(" ".join(args))
PYEOF
)

eval "torchrun --standalone --nproc_per_node=$NGPU --master_port=$PORT train.py $TRAIN_ARGS"
