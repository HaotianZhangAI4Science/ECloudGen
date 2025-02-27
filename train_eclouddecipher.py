# Offline training of an end-to-end clip
# Encoder/decoder.
# Try to return to torch_emb=False
from models.ECloudDecipher.training.train_coati import train_autoencoder, do_args


def main():
    args = do_args()
    args.nodes = 1  # total num nodes.
    args.nr = 0  # rank of this node.
    # note args.gpus will default to the # gpus on this node.
    args.data_parallel = True

    args.test_frac = 0.02
    args.valid_frac = 0.0
    args.n_layer_e3gnn = 5
    args.n_hidden_e3nn = 256
    args.msg_cutoff_e3nn = 12.0
    args.n_hidden_xformer = 256
    args.n_embd_common = 256
    args.n_layer_xformer = 16
    args.n_head = 16
    args.max_n_seq = 250  # max the model can forward
    #    args.n_seq = 90 # max allowed in training.
    args.n_seq = 64  # max allowed in training.
    args.biases = True
    args.torch_emb = False
    args.norm_clips = True
    args.norm_embed = False
    args.token_mlp = True

    args.tokenizer_vocab = "mar"
    args.p_dataset = 0.2
    args.p_formula = 0.0
    args.p_fim = 0.0
    args.p_graph = 0.0
    args.p_clip = 0.9
    args.p_clip_emb_smi = 0.5
    args.p_randsmiles = 0.3
    args.batch_size = 160

    args.online = False  # Possible offline training of an end-to-end clip
    args.lr = 5.0e-4
    args.weight_decay = 0.1

    args.dtype = "float"
    args.n_epochs = 100
    args.clip_grad = 10
    args.test_interval = 2
    args.debug = False

    args.resume_optimizer = False
    # resume from checkpoint file
    args.resume_document = None

    args.ngrad_to_save = 6e6

    # output logs
    args.output_dir = "./logs/"
    # where to save model checkpoints
    args.model_dir = "./model_ckpts/"
    # where to save dataset cache
    args.data_dir = "./dataset/data/ecloud_decipher_token.h5"
    args.model_filename = "ecloud_decipher"

    print(f"running on {args.gpus} gpus")
    #########################################################
    args.world_size = args.gpus * args.nodes
    train_autoencoder(args.gpus, args)
    #########################################################


if __name__ == "__main__":
    main()
