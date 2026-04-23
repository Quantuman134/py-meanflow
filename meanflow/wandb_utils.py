import os
import wandb


def initialize(args, entity, project, run_name, wandb_key=None):
    if wandb_key:
        wandb.login(key=wandb_key)
    elif "WANDB_KEY" in os.environ:
        wandb.login(key=os.environ["WANDB_KEY"])
    wandb.init(
        entity=entity or os.environ.get("WANDB_ENTITY", None),
        project=project or os.environ.get("WANDB_PROJECT", "meanflow"),
        name=run_name,
        config=vars(args),
        settings=wandb.Settings(init_timeout=300),
    )


def log(stats, step=None):
    wandb.log(stats, step=step)
