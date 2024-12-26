import argparse


def get_args():
    parser = argparse.ArgumentParser(description="OMoE")

    # Reproducibility and evaluation arguments
    # parser.add_argument("--k_eval", type=int, default=10, help="Number of items to recommend")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--model_selection", choices=["best", "last"], type=str, default="best", help="")
    parser.add_argument("--cuda", type=str, default="0", help="Cuda device")
    parser.add_argument("--use_pretrained", action="store_true", help="Use pretrained experts")

    # Data arguments
    parser.add_argument(
        "-d", "--dataset", type=str, default="diginetica", choices=["diginetica", "retailrocket", "cosmetics"]
    )
    # parser.add_argument("--data_path", type=str, default="data/dump/", help="Path to dumped dataset")
    # parser.add_argument("--data_intersection", type=str, default="common", help="common/in/any")

    # Model arguments
    parser.add_argument(
        "--loss",
        choices=["cross-entropy", "bce", "ce", "bpr-max", "top1", "softmax", "bpr"],
        type=str,
        default="bpr-max",
        help="Loss function",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[100], help="Layers")
    parser.add_argument("--n_layers", type=int, default=1, help="num layers")
    parser.add_argument("--no_pos_emb", action="store_true", help="Remove positional embedding")
    parser.add_argument("--embedding_size", type=int, default=64, help="Size of the embedding layer")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for the optimizer")
    parser.add_argument("--momentum", type=float, default=0.0, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for the optimizer")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--mode", choices=["first", "last", "any"], type=str, default="last", help="Evaluation mode")
    parser.add_argument("--n_sample", type=int, default=2048, help="Number of additional negative samples")
    parser.add_argument("--maxlen", type=int, default=200, help="Maxlen for attention-based method")
    parser.add_argument("--sample_alpha", type=float, default=0.5, help="alpha for negative sample")
    parser.add_argument(
        "--gate", choices=["single", "multi", "hierarchical"], type=str, default="single", help="Gate type"
    )
    parser.add_argument(
        "--inference", choices=["dense", "sparse"], type=str, default="sparse", help="Inference expert choices"
    )
    parser.add_argument("--warmup_gate", type=int, default=5, help="")
    parser.add_argument("--warmup_expert", type=int, default=-1, help="")
    parser.add_argument("--num_clone", type=int, default=2, help="")
    parser.add_argument("--num_gate", type=int, default=4, help="")
    parser.add_argument("--num_top_experts", type=int, default=2, help="")
    parser.add_argument("--tau", type=float, default=1.0, help="tau in gumbel_softmax")
    parser.add_argument("--use_user_popularity", action="store_true", help="Use user popularity")
    parser.add_argument("--use_session_popularity", action="store_true", help="Use user popularity")
    parser.add_argument("--beta_vae", type=float, default=1.0, help="")

    # parser.add_argument("--save_dir", type=str, default="out_model", help="Save directory")

    args = parser.parse_args()

    return args
