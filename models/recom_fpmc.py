# Copyright 2023 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from copy import deepcopy

from tqdm.auto import trange

from cornac.metrics import Recall
from cornac.utils import get_rng

from . import NextRecommender
from .utils import torch_init_seed


class FPMC(NextRecommender):
    def __init__(
        self,
        name="BPR-NextItem",
        embedding_dim=100,
        batch_size=512,
        learning_rate=0.05,
        momentum=0.0,
        n_sample=2048,
        sample_alpha=0.5,
        item_discount=1.0,
        n_epochs=10,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        mode="last",
        model_selection="best",
        personalized_gate=False,
        is_lt=True,
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
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.mode = mode
        self.model_selection = model_selection
        self.personalized_gate = personalized_gate
        self.is_lt = is_lt

    def bpr_loss(self, item_scores):
        import torch
        import torch.nn.functional as F

        pos_item_scores = torch.diag(item_scores)
        pos_item_scores = pos_item_scores.reshape(pos_item_scores.shape[0], -1)

        logits = F.logsigmoid(pos_item_scores - item_scores)

        loss = -torch.sum(logits * (1.0 - torch.eye(*logits.shape, out=torch.empty_like(logits))))
        return loss / logits.size(0) / max(logits.size(1) - 1, 1)

    def get_gate_input(self, in_uids, in_iids, **kwargs):
        if self.personalized_gate:
            return self.model.UI_emb(in_uids)
        else:
            return self.get_item_embeddings(self.model.LI_emb, in_iids, self.total_items, self.discount, **kwargs)

    def get_hidden(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        return self.model.UI_emb(in_uids), self.model.LI_emb(out_iids), self.model.item_biases(out_iids)

    def get_item_scores(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        scores = self.model(in_uids, in_iids, out_iids)
        return scores

    def get_pred_hidden(self, in_uids, history_iids, **kwargs):
        return self.model.UI_emb(in_uids), self.model.LI_emb.weight[:-1], self.model.item_biases.weight

    def _init_model(self, train_set, val_set=None, pretrained_model=None):
        super().fit(train_set, val_set)

        from .nn_models import FPMC_Model

        self.model = FPMC_Model(
            user_num=self.total_users,
            item_num=self.total_items,
            factor_num=self.embedding_dim,
            device=self.device,
        ).to(self.device)

        self.pad_idx = self.total_items

        if pretrained_model is not None:
            self.model.load_state_dict(pretrained_model.state_dict())

    def fit(self, train_set, val_set=None):
        if not self.trainable:
            return self

        self._init_model(train_set, val_set)
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from .nn_models import IndexedAdagradM, uio_iter

        writer = SummaryWriter(comment=f"{self.name}_emb_{self.embedding_dim}_lr_{self.learning_rate}")
        torch_init_seed(self.seed)

        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum)

        best_recall = 0
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            total_loss = 0
            cnt = 0
            for inc, (in_uids, in_iids, out_iids, start_mask, valid_id, history_iids) in enumerate(
                uio_iter(
                    s_iter=self.train_set.s_iter,
                    uir_tuple=self.train_set.uir_tuple,
                    pad_index=self.total_items,
                    batch_size=self.batch_size,
                    n_sample=self.n_sample,
                    sample_alpha=self.sample_alpha,
                    shuffle=True,
                    rng=self.rng,
                )
            ):
                # BPR user embeddings
                self.model.zero_grad()
                in_uids = torch.tensor(in_uids, dtype=torch.int64, device=self.device)
                in_iids = torch.tensor(in_iids, dtype=torch.int64, device=self.device)
                out_iids = torch.tensor(out_iids, dtype=torch.int64, device=self.device)
                history_iids = torch.tensor(history_iids, dtype=torch.int64, device=self.device)

                scores = self.model(in_uids, in_iids, out_iids)
                loss = self.bpr_loss(item_scores=scores)
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

                [current_val_recall], _ = ranking_eval(
                    model=self,
                    metrics=[Recall(k=20)],
                    train_set=train_set,
                    test_set=val_set,
                    mode=self.mode,
                )
                writer.add_scalar("Recall@20/val", current_val_recall, epoch_id)
                self.last_model = deepcopy(self.model)
                if current_val_recall > best_recall:
                    best_recall = current_val_recall
                    self.best_model = self.last_model
        if self.model_selection == "best":
            self.model = self.best_model
        return self

    def score(self, user_idx, history_items, **kwargs):
        import torch

        with torch.no_grad():
            user_idx = torch.tensor([user_idx], dtype=torch.int64, device=self.device)
            if len(history_items) > 0 and len(history_items[-1]) > 0:
                in_iids = torch.tensor(history_items[-1][-1:], dtype=torch.int64, device=self.device)
            else:
                in_iids = torch.tensor([self.total_items], dtype=torch.int64, device=self.device)

            cdds = torch.arange(self.total_items, dtype=torch.int64, device=self.device)
            pred = self.model.predict(user_idx, in_iids, cdds)
            return pred.cpu().numpy()
