from utils.utils import seed_everything
import argparse
import wandb
import shutil
import os
import torch
from omegaconf import OmegaConf
import logging
from accelerate import Accelerator
from task import Task, Trainer
from datasets.EcloudDataset import ProteinLigandPairDataset # used
from models.ECloudDiff.latentdiff import EcloudLatentDiffusionModel # used

logger = logging.getLogger("moleculekit.smallmol.smallmol")
logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def get_abs_path(*name):
    fn = os.path.join(*name)
    if os.path.isabs(fn):
        return fn
    return os.path.abspath(os.path.join(os.getcwd(), fn))

def updata_cfg(cfg, args):
    for k, v in args.__dict__.items():
        cfg[k] = v
    return cfg


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Molecule Generation For Transformer3D")
    parser.add_argument("--config", type=str, default='eclouddiff', choices=['eclouddiff'],
                        help="Selected a config for this task.")
    parser.add_argument("--gpus", type=str,  default='0')
    parser.add_argument('--debug', type=bool, default=True, help='is or not debug')
    parser.add_argument('--project_name', type=str, default='ECloudGen', help='project name')
    parser.add_argument('--task_name', type=str,
                        default='eclouddiff_32', help='task_name name')
    parser.add_argument('--save', type=str, default='eclouddiff_32')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = OmegaConf.load(get_abs_path('configs', f'{args.config}.yml'))
    cfg = updata_cfg(cfg, args)

    # set
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    accelerator = Accelerator(mixed_precision='bf16')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    seed_everything(args.seed)

    if accelerator.is_main_process:
        if args.save is not None:
            os.makedirs(os.path.join('./save', args.save), exist_ok=True)
            shutil.copy(os.path.join('configs', args.config + '.yml'), os.path.join('./save', args.save, 'configs.yml'))
        if not args.debug:
            wandb.login(key=cfg.WANDB.KEY)
            wandb.init(project=args.project_name, entity="Odin", name=args.task_name)
            wandb.config = args

    accelerator.wait_for_everyone()

    # load task
    task = Task.setup_task(cfg)
    task.set(accelerator, logger, wandb)

    # model, datasets, loss
    task.build_dataset(cfg)
    task.build_model(cfg)

    # optim
    task.build_optim(cfg)

    # train
    trainer = Trainer(task, cfg)
    trainer.train()