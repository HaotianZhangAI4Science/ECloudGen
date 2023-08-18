import argparse
from models.GPT2ModelWithPreFixTuning import GPT2LMHeadMoelWithPrefixTuning
from models.LitModel import GPT2LitModel
from vocab.tokenization import SMILESBPETokenizer
from transformers import GPT2Config
from pytorch_lightning import Trainer

from dataset.dataset import PrefixTuningDataModule
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# parse arguments
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_root', type=str, default='./mol_data/ecloud')
    parser.add_argument('--checkpoint', type=str, default='./ckpt')
    parser.add_argument('--output_dir', type=str, default='./output')
    # Training Configuration
    parser.add_argument('--gpus', type=int, default=1) # Specify either a list of GPU devices or an integer (0 for no GPU).
    parser.add_argument('--num_workers', type=int, default=24) # Number of dataloader worker processes.
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=80)
    parser.add_argument('--max_length', type=int, default=64)

    # Optimizer args
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--scheduler_T_max', type=int, default=1_000)
    parser.add_argument('--final_learning_rate', type=float, default=5e-8)

    parser.add_argument('--vocab_size', type=int, default=1072)

    # Model configuration
    parser.add_argument('--n_layer', type=int, default=8)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=768)

    # Prefix Tuning Configuration
    parser.add_argument('--position_length', type=int, default=256)
    parser.add_argument('--prefix_length', type=int, default=128)
    parser.add_argument('--prefix_dropout', type=float, default=0.1) # modified from 0.1
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    tokenizer_filename = "./vocab/tokenizer.json"

    # Parse arguments
    args = parse_args()


    

    tokenizer = SMILESBPETokenizer.get_hf_tokenizer(
        tokenizer_filename, model_max_length=args.max_length)
    
    
    datamodule = PrefixTuningDataModule(
        data_root=args.data_root,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
        )

    config = GPT2Config(
        prefix_length=args.prefix_length,  
        prefix_dropout=args.prefix_dropout,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        n_positions = args.position_length,
        n_ctx = args.max_length,
        )
    
    gpt2_with_prefix_tuning = GPT2LMHeadMoelWithPrefixTuning(config=config)
    gpt2_with_prefix_tuning.parallelize()

    litModel = GPT2LitModel(
        model=gpt2_with_prefix_tuning,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_eps=args.adam_eps,
        adam_betas=args.adam_betas,
        scheduler_T_max=args.scheduler_T_max,
        final_learning_rate=args.final_learning_rate,
        save_model_every=10_000,
        output_dir=args.output_dir
    )

    trainer = Trainer(
        devices=args.gpus,
        precision="bf16",
        accelerator='gpu',
        max_epochs=args.max_epochs,
        # auto_lr_find=False,
        # auto_scale_batch_size=False
        # logger=wandb_logger
    )

    print("-------------------------------Start training----------------------------")
    trainer.fit(litModel, datamodule=datamodule)
