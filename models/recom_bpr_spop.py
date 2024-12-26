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

from collections import Counter
from copy import deepcopy

import numpy as np
from tqdm.auto import trange

from cornac.metrics import Recall
from cornac.utils import get_rng

from . import NextRecommender


class PopBPR(NextRecommender):
    """MoSE: Mixture of Experts for Next-Item Recommendation

    Parameters
    ----------
    name: string, default: 'MoSE'
        The name of the recommender model.

    layers: list of int, optional, default: [100]
        The number of hidden units in each layer

    loss: str, optional, default: 'cross-entropy'
        Select the loss function.

    batch_size: int, optional, default: 512
        Batch size

    dropout_p_embed: float, optional, default: 0.0
        Dropout ratio for embedding layers

    dropout_p_hidden: float, optional, default: 0.0
        Dropout ratio for hidden layers

    learning_rate: float, optional, default: 0.05
        Learning rate for the optimizer

    momentum: float, optional, default: 0.0
        Momentum for adaptive learning rate

    sample_alpha: float, optional, default: 0.5
        Tradeoff factor controls the contribution of negative sample towards final loss

    n_sample: int, optional, default: 2048
        Number of negative samples

    embedding: int, optional, default: 0

    constrained_embedding: bool, optional, default: True

    n_epochs: int, optional, default: 10

    bpreg: float, optional, default: 1.0
        Regularization coefficient for 'bpr-max' loss.

    elu_param: float, optional, default: 0.5
        Elu param for 'bpr-max' loss

    logq: float, optional, default: 0,
        LogQ correction  to offset the sampling bias affecting 'cross-entropy' loss.

    device: str, optional, default: 'cpu'
        Set to 'cuda' for GPU support.

    trainable: boolean, optional, default: True
        When False, the model will not be re-trained, and input of pre-trained parameters are required.

    verbose: boolean, optional, default: True
        When True, running logs are displayed.

    seed: int, optional, default: None
        Random seed for weight initialization.

    """

    def __init__(
        self,
        name="BPR-NextItem",
        embedding_dim=100,
        batch_size=512,
        learning_rate=0.05,
        momentum=0.0,
        neg_sample=4,
        n_epochs=10,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        mode="last",
        sample_alpha=0.5,
        n_sample=2048,
        model_selection="best",
        use_session_popularity=True,
        init_params=None,
        normalize=False,
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.neg_sample = neg_sample
        self.n_epochs = n_epochs
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.mode = mode
        self.model_selection = model_selection
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        self.init_params = init_params if init_params is not None else {}
        self.use_session_popularity = use_session_popularity
        self.normalize = normalize

    def _build_model(self):
        from .nn_models import BPR_Model

        self.bpr = BPR_Model(
            user_num=self.total_users,
            item_num=self.total_items + 1,
            factor_num=self.embedding_dim,
            device=self.device,
        )

    def from_pretrained(self, pretrained_bpr, pretrained_gru=None):
        """Load models from pretrained

        Args:
            pretrained_bpr (models.bpr.BPR_Model): _description_
            pretrained_gru (_type_, optional): _description_. Defaults to None.
        """
        self._build_model()

        self.bpr.from_pretrained(pretrained_bpr)
        if pretrained_gru is not None:
            self.gru.from_pretrained(pretrained_gru)

    def negative_sampling(self, item_indices, item_dist, uids, num_neg, matrix):
        neg_items = np.ones((len(uids), num_neg), dtype="int") * self.total_items
        for uid_index, uid in enumerate(uids):
            negative_samples = self.rng.choice(item_indices, size=num_neg, replace=True, p=item_dist)
            for i in range(num_neg):
                while matrix[uid, negative_samples[i]] > 0:
                    negative_samples[i] = self.rng.choice(item_indices, size=1, replace=True, p=item_dist)[0]
            neg_items[uid_index] = negative_samples
        return neg_items

    def bpr_loss(self, bpr_item_scores, spop_item_scores, gate_value):
        import torch
        import torch.nn.functional as F

        if self.normalize:
            bpr_item_scores = F.normalize(bpr_item_scores, p=2, dim=1)
            spop_item_scores = F.normalize(spop_item_scores, p=2, dim=1)

        # aggregate based on gate_value
        item_scores = bpr_item_scores * gate_value[:, :1] + spop_item_scores * gate_value[:, 1:]
        pos_item_scores = torch.diag(item_scores)
        pos_item_scores = pos_item_scores.reshape(pos_item_scores.shape[0], -1)

        # Vectorized loss calculation
        logits = F.logsigmoid(pos_item_scores - item_scores)

        # remove diagonal of logits
        loss = -torch.sum(logits * (1.0 - torch.eye(*logits.shape, out=torch.empty_like(logits))))
        return loss / logits.size(0) / max(logits.size(1) - 1, 1)

    def get_item_popularity_scores(self):
        self.item_freq = Counter(self.train_set.uir_tuple[1])
        item_popularity_scores = np.zeros(self.total_items, dtype=np.float32)
        max_item_freq = max(self.item_freq.values()) if len(self.item_freq) > 0 else 1
        for iid, freq in self.item_freq.items():
            item_popularity_scores[iid] = freq / max_item_freq

        return item_popularity_scores

    def spop_scores(self, item_popularity_scores):
        # clone by total_users
        user_item_scores = np.tile(item_popularity_scores, (self.total_users, 1))

        for uid in range(self.total_users):
            user_history_items = [
                self.train_set.uir_tuple[1][idx]
                for sid in self.train_set.user_session_data[uid]
                for idx in self.train_set.sessions[sid]
            ]
            if self.use_session_popularity:
                s_item_freq = Counter([iid for iid in user_history_items])
                for iid, cnt in s_item_freq.most_common():
                    user_item_scores[uid][iid] += cnt
        return user_item_scores

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        import torch

        from .nn_models import uio_iter

        from .nn_models import BPR_Model

        self.item_popularity_scores = self.get_item_popularity_scores()
        self.user_spop_scores = self.spop_scores(self.item_popularity_scores)  # shape: total_users x total_items
        # item_indices = np.array([iid for iid, _ in item_freq.most_common()], dtype="int")
        # item_dist = np.array([cnt for _, cnt in item_freq.most_common()], dtype="float") ** self.sample_alpha
        # item_dist = item_dist / item_dist.sum()

        self.bpr = BPR_Model(
            user_num=self.total_users,
            item_num=self.total_items + 1,
            factor_num=self.embedding_dim,
            device=self.device,
        )

        if "bpr" in self.init_params:
            self.bpr.from_pretrained(self.init_params["bpr"])

        expert_num = 2
        self.gate = torch.nn.Sequential(torch.nn.Linear(self.embedding_dim, expert_num), torch.nn.Softmax(dim=1)).to(
            self.device
        )

        self.moe = torch.nn.ModuleList([self.bpr, self.gate])

        # opt = IndexedAdagradM(self.moe.parameters(), self.learning_rate, self.momentum)
        opt = torch.optim.SGD(self.moe.parameters(), self.learning_rate, self.momentum)

        best_recall = 0
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            total_loss = 0
            cnt = 0
            for inc, (in_uids, in_iids, out_iids, start_mask, valid_id, _) in enumerate(
                uio_iter(
                    s_iter=self.train_set.s_iter,
                    uir_tuple=self.train_set.uir_tuple,
                    pad_index=self.total_items,
                    batch_size=self.batch_size,
                    n_sample=self.n_sample,
                    sample_alpha=self.sample_alpha,
                    shuffle=True,
                )
            ):
                spop_item_scores = torch.tensor(self.user_spop_scores[in_uids][:, out_iids], device=self.device)

                in_iids = torch.tensor(in_iids, requires_grad=False, device=self.device)
                out_iids = torch.tensor(out_iids, requires_grad=False, device=self.device)

                # BPR user embeddings
                self.bpr.zero_grad()
                u_bpr = self.bpr.embed_user(torch.tensor(in_uids, dtype=torch.int64, device=self.device))
                i_bpr = self.bpr.embed_item(out_iids)
                bpr_item_scores = torch.mm(u_bpr, i_bpr.T)  # batch_size x batch_size (uids x out_iids)

                # gate
                user_emb = self.bpr.embed_user(torch.tensor(in_uids, device=self.device))

                # aggregation
                gate_value = self.gate(user_emb)

                loss = self.bpr_loss(bpr_item_scores, spop_item_scores, gate_value)
                loss.backward()
                opt.step()

                total_loss += loss.cpu().detach().numpy() * len(in_iids)

                cnt += len(in_iids)

                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))

            # Evaluate the model on the validation set
            if self.model_selection == "best" and val_set is not None:
                # recall = self.evaluate(val_set, k=20, batch_size=512)
                from .eval import ranking_eval

                [current_val_recall], _ = ranking_eval(
                    model=self,
                    metrics=[Recall(k=20)],
                    train_set=train_set,
                    test_set=val_set,
                    mode=self.mode,
                )
                print(f"Epoch {epoch_id:03d}, Recall@20: {current_val_recall:.4f}")
                if current_val_recall > best_recall:
                    best_recall = current_val_recall
                    self.best_model = deepcopy(self)
        if self.model_selection == "best":
            self = self.best_model

        return self

    def score(self, user_idx, history_items, **kwargs):
        import torch

        with torch.no_grad():
            history_items = [item for sublist in history_items for item in sublist]
            spop_item_scores = torch.tensor(self.item_popularity_scores, device=self.device)
            s_item_freq = Counter([iid for iid in history_items])
            for iid, cnt in s_item_freq.most_common():
                spop_item_scores[iid] += cnt

            user_emb = self.bpr.embed_user(torch.tensor([user_idx], dtype=torch.int64, device=self.device))
            items_emb = self.bpr.embed_item.weight[:-1]
            bpr_item_scores = torch.mm(user_emb, items_emb.T)
            if self.normalize:
                spop_item_scores = torch.nn.functional.normalize(spop_item_scores, p=2, dim=0)
                bpr_item_scores = torch.nn.functional.normalize(bpr_item_scores, p=2, dim=1)
            gate_value = self.gate(user_emb)
            scores = bpr_item_scores * gate_value[:, 0] + spop_item_scores * gate_value[:, 1]
            return scores.squeeze().cpu().numpy()
