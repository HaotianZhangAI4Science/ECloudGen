import os
import torch
import math
import logging
import pickle
# import torch._dynamo as dynamo

from models.ECloudDiff.ema import ModelEMA
from torch.utils.data import DataLoader

from utils.utils import save_config, get_parameter_number, get_optimizer, get_scheduler, get_scheduler_dataloader

class Task(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.model = None
        self.ema = None
        self.train_dataloader = None
        self.valid_dataloader = None
        self.test_dataloader = None
        self.test_datasets = None
        self.loss = None
        self.optimizer = None
        self.lr_scheduler = None
        self.max_train_steps = None
        self.accelerator = None
        self.logger = None
        self.wandb = None

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls(cfg, **kwargs)

    def set(self, accelerator, logger, wandb):
        self.accelerator = accelerator
        self.logger = logger
        self.wandb = wandb

    def build_dataset(self, cfg):
        import datasets
        self.train_datasets = datasets.build_datasets(cfg, self, mode='train')
        self.valid_datasets = datasets.build_datasets(cfg, self, mode='valid')

        self.train_dataloader = DataLoader(
            self.train_datasets,
            collate_fn=self.train_datasets.collator,
            batch_size=cfg.DATA.train_batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            drop_last=True,
            persistent_workers=cfg.SOLVER.NUM_WORKERS > 0,
        )

        self.valid_dataloader = DataLoader(
            self.valid_datasets,
            collate_fn=self.valid_datasets.collator,
            batch_size=cfg.DATA.valid_batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=cfg.SOLVER.NUM_WORKERS,
            drop_last=False,
            persistent_workers=cfg.SOLVER.NUM_WORKERS > 0,
        )

    def build_model(self, cfg):
        import models
        self.model = models.build_model(cfg, self)
        if self.cfg.MODEL.FREEZE_ENCODER_DECODER:
            # module_list = [self.model.Encoder, self.model.Decoder]
            module_list = [self.model.PocketEncoder, self.model.LigandEncoder, self.model.EcloudDecoder]
            if self.cfg.MODEL.DIFFUSION.add_vqvae_loss:
                module_list.append(self.model.quantizer)
            for module in module_list:
                for p in module.parameters():
                    p.requires_grad = False

        # self.model = torch.compile(self.model)
        # self.model = dynamo.optimize("inductor")(self.model)

        if self.cfg.MODEL.USE_MODEL_CKPT:
            pretrain_path = os.path.join(self.cfg.MODEL.CHECKPOINT_PATH, self.cfg.MODEL.MODEL_NAME)
            assert os.path.exists(pretrain_path), 'checkpoint no exists! '
            pretrained_dict = torch.load(pretrain_path, map_location='cpu')
            model_dict = self.model.state_dict()
            if 'model' in pretrained_dict:
                pretrained_dict = {k: v for k, v in pretrained_dict['model'].items()}
            else:
                pretrained_dict = {k: v for k, v in pretrained_dict.items()}

            total = len(model_dict.keys())
            rate = sum([1 for k in model_dict.keys() if k in pretrained_dict]) / total
            print('Prarmeter Loading Rate: ', rate)
            print([k for k in pretrained_dict.keys() if k not in model_dict])
            print([k for k in model_dict.keys() if k not in pretrained_dict])

            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict, strict=False)
            self.logger.info(f"  =====================================================================================")
            self.logger.info(f"  Load pretrained model from {pretrain_path}")
            self.logger.info(f"  =====================================================================================")


        # self.logger.info(f"  -*- -*- -*- -*- -*- -*- Parameter Volumn -*- -*- -*- -*- -*- -*- -*- -*- ")
        self.logger.info(get_parameter_number(self.model))
        # self.logger.info(f"   -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- -*- ")

        if cfg.MODEL.USE_EMA:
            self.ema = ModelEMA(self.model)

    def build_optim(self, cfg):
        model = self.model
        optimizer = get_optimizer(cfg, model)

        if self.ema is not None:
            self.ema.ema = self.accelerator.prepare(self.ema.ema)

        self.model, self.optimizer, self.train_dataloader, self.valid_dataloader = self.accelerator.prepare(
            model,
            optimizer,
            self.train_dataloader,
            self.valid_dataloader,
        )

        # Scheduler and math around the number of training steps.
        
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / cfg.SOLVER.GRADIENT_ACC)
        print('num_update_steps_per_epoch: ', num_update_steps_per_epoch)
        max_train_steps = cfg.SOLVER.MAX_EPOCHS * num_update_steps_per_epoch
        num_warmup_steps = cfg.SOLVER.WARMUP_STEP_RATIO * max_train_steps
        self.lr_scheduler = get_scheduler_dataloader(cfg, self.optimizer, self.train_dataloader)

        # self.lr_scheduler = get_scheduler(
        #     name=cfg.SOLVER.LR_SCHEDULER,
        #     optimizer=self.optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=max_train_steps
        # )
        self.max_train_steps = max_train_steps
