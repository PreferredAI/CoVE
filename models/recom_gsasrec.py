from copy import deepcopy

from tqdm.auto import trange

from cornac.metrics import Recall
from cornac.utils import get_rng

from . import NextRecommender
from .utils import torch_init_seed


class GSASRec(NextRecommender):
    def __init__(
        self,
        name="GSASRec",
        embedding_dim=100,
        batch_size=512,
        learning_rate=0.001,
        momentum=0.0,
        n_sample=256,
        sample_alpha=0.5,
        gbce_t=0.75,
        n_epochs=10,
        max_len=50,
        num_blocks=2,
        num_heads=1,
        dropout=0.5,
        l2_reg=0.0,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        mode="last",
        model_selection="best",
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.gbce_t = gbce_t
        self.n_epochs = n_epochs
        self.max_len = max_len
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.dropout = dropout
        self.l2_reg = l2_reg
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.mode = mode
        self.model_selection = model_selection

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        import numpy as np
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from .nn_models import GSASRecModel, softmax_loss, uio_iter_pad

        writer = SummaryWriter(comment=f"{self.name}_emb_{self.embedding_dim}_lr_{self.learning_rate}")
        torch_init_seed(self.seed)

        self.gsasrec = GSASRecModel(
            num_items=self.total_items,
            embedding_dim=self.embedding_dim,
            sequence_length=self.max_len,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            dropout_rate=self.dropout,
        ).to(self.device)

        if not self.trainable:
            return self

        criteria = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.gsasrec.parameters(), lr=self.learning_rate, betas=(0.9, 0.98))

        best_recall = 0
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            total_loss = 0
            cnt = 0
            for inc, (in_uids, model_input, labels, negative_samples) in enumerate(
                uio_iter_pad(
                    s_iter=self.train_set.s_iter,
                    uir_tuple=self.train_set.uir_tuple,
                    pad_index=self.total_items,
                    batch_size=self.batch_size,
                    n_sample=self.batch_size * self.n_sample * self.max_len,
                    sample_alpha=self.sample_alpha,
                    shuffle=True,
                    rng=self.rng,
                    max_len=self.max_len,
                )
            ):
                self.gsasrec.zero_grad()
                model_input = torch.tensor(model_input, device=self.device)
                labels = torch.tensor(labels, device=self.device)
                negative_samples = torch.tensor(negative_samples, device=self.device)
                last_hidden_state, attentions = self.gsasrec(model_input)
                negatives = negative_samples.view(self.batch_size, self.max_len, self.n_sample)
                pos_neg_concat = torch.cat([labels.unsqueeze(-1), negatives[: len(model_input), :, :]], dim=-1)
                output_embeddings = self.gsasrec.get_output_embeddings()
                pos_neg_embeddings = output_embeddings(pos_neg_concat)
                mask = (model_input != self.total_items + 1).float()
                logits = torch.einsum("bse, bsne -> bsn", last_hidden_state, pos_neg_embeddings)
                # gt = torch.zeros_like(logits)
                # gt[:, :, 0] = 1
                opt.zero_grad()
                loss = softmax_loss(logits)
                total_loss += loss.cpu().detach().numpy()
                loss.backward()
                opt.step()

                cnt += len(model_input)
                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))
            # writer.add_scalar("Loss/train", total_loss / cnt, epoch_id)
            # Evaluate the model on the validation set
            if self.model_selection == "best" and val_set is not None:
                from .eval import ranking_eval

                [current_val_recall], _ = ranking_eval(
                    model=self,
                    metrics=[Recall(k=20)],
                    train_set=train_set,
                    test_set=val_set,
                    mode="last",
                )
                writer.add_scalar("Recall@20/val", current_val_recall, epoch_id)
                self.last_model = deepcopy(self.gsasrec)
                if current_val_recall > best_recall:
                    best_recall = current_val_recall
                    self.best_model = self.last_model
        if self.model_selection == "best":
            self.gsasrec = self.best_model
        return self

    def score(self, user_idx, history_items, **kwargs):
        import numpy as np

        last_seq = history_items[-1] if len(history_items) > 0 and len(history_items[-1]) > 0 else []
        log_seqs = [self.total_items] * (self.max_len - len(last_seq)) + last_seq
        log_seqs = np.array([log_seqs[-self.max_len :]])

        return self.gsasrec.get_predictions(log_seqs, self.total_items)
