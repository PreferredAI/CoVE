from copy import deepcopy

from tqdm.auto import trange

from cornac.metrics import Recall
from cornac.utils import get_rng

from . import NextRecommender
from .utils import torch_init_seed


class SASRec(NextRecommender):

    def __init__(
        self,
        name="SASRec",
        embedding_dim=100,
        batch_size=512,
        learning_rate=0.001,
        momentum=0.0,
        n_sample=2048,
        sample_alpha=0.5,
        item_discount=1.0,
        n_epochs=10,
        max_len=50,
        num_blocks=2,
        num_heads=1,
        dropout=0.2,
        l2_reg=0.0,
        bpreg=1.0,
        device="cpu",
        loss="bce",
        use_pos_emb=True,
        trainable=True,
        verbose=False,
        seed=None,
        mode="last",
        model_selection="best",
        init_params=None,
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.discount = item_discount
        self.n_epochs = n_epochs
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.bpreg = bpreg
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.mode = mode
        self.model_selection = model_selection
        self.loss = loss
        self.use_pos_emb = use_pos_emb
        self.init_params = {} if init_params is None else init_params

    def get_gate_input(self, user_idx, in_iids, **kwargs):
        # return self.get_discounted_avg_items(self.model.Wy, in_iids, self.total_items, self.discount, **kwargs)
        return self.get_item_embeddings(self.model.item_emb, in_iids, self.total_items, self.discount, **kwargs)

    def get_hidden(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        return self.model(None, history_iids, out_iids, return_hidden=True)

    def get_item_scores(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        self.model.zero_grad()
        scores = self.model.forward(in_iids, history_iids, out_iids, return_hidden=False)

        return scores

    def get_pred_hidden(self, in_uids, history_iids):
        import torch

        history_iids = history_iids[-self.max_len :].unsqueeze(0)
        out_iids = torch.arange(self.total_items, device=self.device, requires_grad=False)

        return self.model(None, history_iids, out_iids, return_hidden=True)

    def _init_model(self, train_set, pretrained_model=None):
        super().fit(train_set)
        from .nn_models import SASRecModel

        self.model = SASRecModel(
            user_num=self.total_users,
            item_num=self.total_items,
            embedding_dim=self.embedding_dim,
            maxlen=self.max_len,
            n_layers=self.num_blocks,
            n_heads=self.num_heads,
            use_pos_emb=self.use_pos_emb,
            dropout=self.dropout,
            pad_idx=self.total_items,
            device=self.device,
        )

        if pretrained_model is not None:
            self.model.load_state_dict(pretrained_model.state_dict())

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        if not self.trainable:
            return self

        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from .nn_models import SASRecModel, softmax_loss, uio_iter

        writer = SummaryWriter(comment=f"{self.name}_emb_{self.embedding_dim}_lr_{self.learning_rate}")
        torch_init_seed(self.seed)

        self.model = SASRecModel(
            user_num=self.total_users,
            item_num=self.total_items,
            embedding_dim=self.embedding_dim,
            maxlen=self.max_len,
            n_layers=self.num_blocks,
            n_heads=self.num_heads,
            use_pos_emb=self.use_pos_emb,
            dropout=self.dropout,
            pad_idx=self.total_items,
            device=self.device,
        )

        if self.loss == "bce":
            criteria = torch.nn.BCEWithLogitsLoss()
        elif self.loss == "ce":
            criteria = torch.nn.CrossEntropyLoss()
        elif self.loss == "softmax":
            criteria = softmax_loss
        elif self.loss == "bpr":
            criteria = self.bpr_loss
        elif self.loss == "bpr-max":
            criteria = self.bpr_max_loss

        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.98))
        # opt = torch.optim.Adam(self.model.parameters())

        best_epoch_id = 0
        best_recall = 0
        best_val_loss = +np.inf
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            total_loss = 0
            cnt = 0
            for inc, (in_uids, in_iids, out_iids, _, _, hist_iids) in enumerate(
                uio_iter(
                    s_iter=self.train_set.s_iter,
                    uir_tuple=self.train_set.uir_tuple,
                    pad_index=self.total_items,
                    batch_size=self.batch_size,
                    n_sample=self.n_sample,
                    sample_alpha=self.sample_alpha,
                    rng=self.rng,
                    shuffle=True,
                    max_len=self.max_len,
                )
            ):
                out_iids = torch.tensor(out_iids, requires_grad=False, device=self.device)
                hist_iids = torch.tensor(hist_iids, requires_grad=False, device=self.device)

                self.model.zero_grad()
                item_scores = self.model(None, hist_iids, out_iids, return_hidden=False)

                loss = criteria(item_scores)

                if self.l2_reg > 0:
                    for param in self.model.parameters():
                        loss += self.l2_reg * torch.norm(param)

                loss.backward()
                opt.step()

                total_loss += loss.cpu().detach().numpy() * len(in_iids)

                cnt += len(in_iids)
                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))
            writer.add_scalar("Loss/train", total_loss / cnt, epoch_id)
            # Evaluate the model on the validation set
            if self.model_selection == "best" and val_set is not None:
                from .eval import ranking_eval
                val_loss = 0

                with torch.no_grad():
                    for inc, (in_uids, in_iids, out_iids, _, _, hist_iids) in enumerate(
                        uio_iter(
                            s_iter=self.val_set.s_iter,
                            uir_tuple=self.val_set.uir_tuple,
                            pad_index=self.total_items,
                            batch_size=self.batch_size,
                            n_sample=self.n_sample,
                            sample_alpha=self.sample_alpha,
                            rng=self.rng,
                            shuffle=False,
                            max_len=self.max_len,
                        )
                    ):
                        out_iids = torch.tensor(out_iids, requires_grad=False, device=self.device)
                        hist_iids = torch.tensor(hist_iids, requires_grad=False, device=self.device)

                        self.model.zero_grad()
                        item_scores = self.model(None, hist_iids, out_iids, return_hidden=False)

                        loss = criteria(item_scores)

                        val_loss += loss.cpu().detach().numpy() * len(in_iids)

                    if val_loss < best_val_loss:
                        best_epoch_id = epoch_id
                        best_val_loss = val_loss
                        self.best_model = deepcopy(self.model)
                # [current_val_recall], _ = ranking_eval(
                #     model=self,
                #     metrics=[Recall(k=20)],
                #     train_set=train_set,
                #     test_set=val_set,
                #     mode="last",
                # )
                # writer.add_scalar("Recall@20/val", current_val_recall, epoch_id)
                # self.last_model = deepcopy(self.model)
                # if current_val_recall > best_recall:
                #     best_recall = current_val_recall
                #     self.best_model = self.last_model

        if self.model_selection == "best":
            self.model = self.best_model
            print("#" * 10, "\n", f"Best model found at epoch {best_epoch_id}", "\n", "#" * 10)
        return self

    def score(self, user_idx, history_items, **kwargs):
        import numpy as np

        last_seq = history_items[-1] if len(history_items) > 0 and len(history_items[-1]) > 0 else []
        log_seqs = [self.total_items] * (self.max_len - len(last_seq)) + last_seq
        log_seqs = np.array([log_seqs[-self.max_len :]])

        return self.model.predict(None, log_seqs)
