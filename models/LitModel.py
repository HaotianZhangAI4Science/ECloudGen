import pytorch_lightning as pl
import torch
from ignite.handlers import create_lr_scheduler_with_warmup


class GPT2LitModel(pl.LightningModule):
    """Lightning module for autoregressive (causal) transformer language modeling.
    Successfully tested on HuggingFace `GPT2LMHeadModel`.
    """

    def __init__(self,
                model,
                batch_size: int,
                learning_rate: float,
                final_learning_rate: float, 
                weight_decay: float, 
                adam_eps: float,
                adam_betas: tuple, scheduler_T_max: int,
                save_model_every: int = 10_000, 
                output_dir: str = ""):
        super().__init__()
        self.save_hyperparameters(ignore=("model", "save_model_every",
                                          "output_dir"))
        self.model = model
        self.save_model_every = save_model_every
        self.output_dir = output_dir or "./gpt2litmodel-logs"

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        return output

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        if self.save_model_every > 0 and batch_idx % self.save_model_every == 0:
            self.model.save_pretrained(self.output_dir)

        self.log("loss", outputs["loss"], on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return {'loss': outputs['loss']}

    # def training_epoch_end(self, outputs):
    #     if self.save_model_every > 0:
    #         self.model.save_pretrained(self.output_dir)

    #     losses = [step_output["loss"] for step_output in outputs]
    #     mean_loss = torch.tensor(losses).mean()
    #     ppl = torch.exp(mean_loss)

    #     self.log("ppl", ppl, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        parameters = self.named_parameters()
        no_decay = ["bias", "LayerNorm.weight"]
        grouped_parameters = [
            {"params": [p for n, p in parameters
                        if not any(nd in n for nd in no_decay)],
             "weight_decay": self.hparams.weight_decay},
            {"params": [p for n, p in parameters
                        if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0}]
        optimizer = torch.optim.AdamW(
            grouped_parameters, lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_eps, betas=self.hparams.adam_betas)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.hparams.scheduler_T_max,
            eta_min=self.hparams.final_learning_rate)
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)


        return {'optimizer': optimizer,
                'lr_scheduler': {'scheduler': lr_scheduler,
                                 'interval': 'step', 'frequency': 1}}
