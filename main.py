import os
import pickle

import numpy as np
import torch

from args import get_args
from cornac.data import Reader
from cornac.experiment import Experiment
from cornac.metrics import AUC, MRR, NDCG, Recall
from models import BPR, FPMC, GRU4Rec, MoSE, MoVE, PopBPR, SASRec, SPop
from models.eval import NextSessionEvaluation

if __name__ == "__main__":
    args = get_args()
    num_cuda_devices = torch.cuda.device_count()
    if num_cuda_devices == 0 or "cpu" in args.cuda:
        device = "cpu"
    else:
        avoid_cuda = -1
        available_cuda = [i for i in range(num_cuda_devices) if i != avoid_cuda]
        if avoid_cuda >= 0:
            num_cuda_devices -= 1
        c_device = int(args.cuda) % num_cuda_devices
        c_device = available_cuda[c_device]
        device = torch.device(f"cuda:{c_device}")

    warmup_expert = 0 if args.use_pretrained else args.warmup_expert

    folder_path = os.path.join("processed_data", args.dataset)
    train_path = os.path.join(folder_path, "train.csv")
    val_path = os.path.join(folder_path, "val.csv")
    test_path = os.path.join(folder_path, "test.csv")

    # "diginetica", "retailrocket", "cosmetics"
    if args.use_pretrained:
        import json

        def load_pretrained_model(model_path, device="cpu"):
            if model_path.endswith(".pkl"):
                from utils import CPU_Unpickler

                pretrained_model = CPU_Unpickler(open(model_path, "rb")).load()
                pretrained_model = pretrained_model.model
            else:  # pth model
                pretrained_model = torch.load(model_path, map_location=device)
            return pretrained_model

        with open("pretrained_paths.json", "r") as file:
            pretrained_paths = json.load(file)

        dataset_paths = pretrained_paths.get(args.dataset, {})
        bpr_pretrained_path = dataset_paths.get("bpr_pretrained_path")
        fpmc_pretrained_path = dataset_paths.get("fpmc_pretrained_path")
        gru_pretrained_path = dataset_paths.get("gru_pretrained_path")
        sas_pretrained_path = dataset_paths.get("sas_pretrained_path")

        if os.path.exists(bpr_pretrained_path):
            pretrained_bpr = load_pretrained_model(bpr_pretrained_path, device=device)

        if os.path.exists(gru_pretrained_path):
            pretrained_gru = load_pretrained_model(gru_pretrained_path, device=device)

        if os.path.exists(fpmc_pretrained_path):
            pretrained_fpmc = load_pretrained_model(fpmc_pretrained_path, device=device)

        if os.path.exists(sas_pretrained_path):
            pretrained_sas = load_pretrained_model(sas_pretrained_path, device=device)

    reader = Reader()
    train_data = reader.read(train_path, fmt="USIT", sep=",")
    val_data = reader.read(val_path, fmt="USIT", sep=",")
    test_data = reader.read(test_path, fmt="USIT", sep=",")

    eval_method = NextSessionEvaluation.from_splits(
        train_data=train_data,
        test_data=test_data,
        val_data=val_data,
        fmt="USIT",
        seed=args.seed,
        verbose=True,
        mode=args.mode,
        exclude_unknowns=True,
    )
    expert_list = []
    for idx in range(args.num_clone):
        bpr = BPR(
            name=f"BPR-NextItem_emb{args.layers[0]}_ne{args.n_epochs}_bs{args.batch_size}_lr{args.lr}",
            embedding_dim=args.layers[0],
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            verbose=True,
            model_selection=args.model_selection,
            device=device,
            seed=args.seed,
        )
        gru4rec = GRU4Rec(
            name=f"GRU4Rec_emb{args.layers[0]}_l{args.loss}_ne{args.n_epochs}_bs{args.batch_size}_lr{args.lr}",
            batch_size=args.batch_size,
            n_sample=args.n_sample,
            sample_alpha=args.sample_alpha,
            layers=args.layers,
            device=device,
            verbose=True,
            n_epochs=args.n_epochs,
            mode=args.mode,
            model_selection=args.model_selection,
            learning_rate=args.lr,
            loss=args.loss,
            seed=args.seed,
        )
        fpmc = FPMC(
            name=f"FPMC-NextItem_emb{args.layers[0]}_ne{args.n_epochs}_bs{args.batch_size}_lr{args.lr}",
            embedding_dim=args.layers[0],
            n_epochs=args.n_epochs,
            learning_rate=args.lr,
            verbose=True,
            model_selection=args.model_selection,
            device=device,
            seed=args.seed,
            personalized_gate=True,
        )
        sasrec = SASRec(
            name=f"SASRec_emb{args.layers[0]}_ne{args.n_epochs}_bs{args.batch_size}_lr{args.lr}",
            embedding_dim=args.layers[0],
            batch_size=args.batch_size,
            learning_rate=args.lr,
            momentum=args.momentum,
            n_sample=args.n_sample,
            sample_alpha=args.sample_alpha,
            n_epochs=args.n_epochs,
            max_len=args.maxlen,
            loss=args.loss,
            num_blocks=args.n_layers,
            verbose=True,
            model_selection=args.model_selection,
            device=device,
            seed=args.seed,
            use_pos_emb=not args.no_pos_emb,
        )

        if args.use_pretrained:
            bpr._init_model(eval_method.train_set, pretrained_model=pretrained_bpr)
            gru4rec._init_model(eval_method.train_set, pretrained_model=pretrained_gru)
            fpmc._init_model(eval_method.train_set, pretrained_model=pretrained_fpmc)
            sasrec._init_model(eval_method.train_set, pretrained_model=pretrained_sas)

        expert_list += [bpr, gru4rec, fpmc, sasrec]
    # if args.num_gate == 4:
    #     expert_list = sorted(expert_list, key=lambda x: x.name)
    move = MoVE(
        name=f"MoVE_emb{args.layers[0]}_ne{args.n_epochs}_bs{args.batch_size}_lr{args.lr}_tau{args.tau}",
        batch_size=args.batch_size,
        n_sample=args.n_sample,
        sample_alpha=args.sample_alpha,
        # layers=args.layers,
        embedding_dim=args.layers[-1],
        device=device,
        verbose=True,
        n_epochs=args.n_epochs,
        loss=args.loss,
        mode=args.mode,
        model_selection=args.model_selection,
        learning_rate=args.lr,
        experts=expert_list,
        num_top_experts=args.num_top_experts,
        num_gate=args.num_gate,
        gate_type=args.gate,
        tau=args.tau,
        warmup_gate=args.warmup_gate,
        warmup_expert=warmup_expert,
        inference=args.inference,
    )
    models = [
        # spop,
        # bpr,
        # popbpr,
        # gru4rec,
        # mose,
        move,
    ]
    metrics = [
        AUC(),
        MRR(),
        Recall(k=1),
        Recall(k=3),
        Recall(k=5),
        Recall(k=10),
        Recall(k=20),
        MRR(),
        NDCG(k=1),
        NDCG(k=3),
        NDCG(k=5),
        NDCG(k=10),
        NDCG(k=20),
    ]
    exp = Experiment(
        eval_method=eval_method,
        models=models,
        metrics=metrics,
        user_based=True,
        save_dir=f"out/{args.dataset}",
        verbose=True,
    )

    print("#" * 20)
    print("#" * 20)
    print("Next-item setting (last item)")
    exp.run()

    # export experiment results for ensemble evaluation
    export_result_dir = os.path.join(exp.save_dir, models[0].name)
    os.makedirs(export_result_dir, exist_ok=True)
    with open(os.path.join(export_result_dir, f"{models[0].name}_{args.dataset}_result.pkl"), "wb") as f:
        f.write(pickle.dumps(exp.result[0]))
    with open(os.path.join(export_result_dir, "val_result.pkl"), "wb") as f:
        f.write(pickle.dumps(exp.val_result[0]))

    # define metric to evaluate
    metric_used = "Recall@20"
    test_user_res = np.concatenate(list(exp.result[0].metric_user_results[metric_used].values()))
    val_user_res = np.concatenate(list(exp.val_result[0].metric_user_results[metric_used].values()))

    if hasattr(exp.models[0], "cached_gate_values") and len(exp.models[0].cached_gate_values) > 0:
        gate_values = np.array(models[0].cached_gate_values)
        if gate_values.ndim > 1:
            gate_avg = gate_values.mean(axis=0)
            print("Gate values average:", ", ".join(f"{x:.4f}\t" for x in gate_avg))
            print("Expert choice counts:", [f"{x:.4f}  " for x in (models[0].expert_choice / args.n_epochs)])
# out_model_path = f"out/{models[0].name}"
# if os.path.exists(out_model_path):
#     pretrained_path = max(
#         glob.glob(os.path.join(out_model_path, "*.pkl")),
#         key=os.path.getctime,
#     )  # get latest saved model path
#     with open(pretrained_path, "rb") as f:
#         pretrained_model = pickle.load(f)

# exp.save_dir = None

# print("#" * 20)
# print("#" * 20)
# print("Next first item setting")
# exp.models[0].trainable = False
# eval_method.mode = "first"
# exp.models[0].cached_gate_values = []
# exp.run()
# if hasattr(exp.models[0], "cached_gate_values") and len(exp.models[0].cached_gate_values) > 0:
#     gate_values = np.array(exp.models[0].cached_gate_values)
#     gate_avg = gate_values.mean(axis=0)
#     print("Gate values average:", ", ".join(f"{x:.4f}" for x in gate_avg))

# print("#" * 20)
# print("#" * 20)
# print("Next basket item setting")
# exp.models[0].trainable = False
# eval_method.mode = "any"
# exp.models[0].cached_gate_values = []
# exp.run()

# if hasattr(exp.models[0], "cached_gate_values") and len(exp.models[0].cached_gate_values) > 0:
#     gate_values = np.array(exp.models[0].cached_gate_values)
#     gate_avg = gate_values.mean(axis=0)
#     print("Gate values average:", ", ".join(f"{x:.4f}" for x in gate_avg))
