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


class BPR(NextRecommender):
    """BPR: BPR for Next-Item Recommendation

    Parameters
    ----------
    name: string, default: 'OMoE'
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
        self.is_lt = True

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
            return self.model.embed_user(in_uids)
        else:
            return self.get_item_embeddings(self.model.embed_item, in_iids, self.total_items, self.discount, **kwargs)

    def get_hidden(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        return self.model.embed_user(in_uids), self.model.embed_item(out_iids), self.model.item_biases(out_iids)

    def get_item_scores(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        import torch

        u_bpr = self.model.embed_user(in_uids)
        i_bpr = self.model.embed_item(out_iids).unsqueeze(0)
        i_bpr = i_bpr.expand(u_bpr.size(0), -1, -1)
        b_bpr = self.model.item_biases(out_iids)
        b_bpr = b_bpr.expand(u_bpr.size(0), -1, -1)
        bpr_item_scores = torch.einsum("bd,bnd->bn", u_bpr, i_bpr) + b_bpr.squeeze()

        return bpr_item_scores

    def get_pred_hidden(self, in_uids, history_iids, **kwargs):
        return self.model.embed_user(in_uids), self.model.embed_item.weight[:-1], self.model.item_biases.weight[:-1]

    def _init_model(self, train_set, pretrained_model=None):
        super().fit(train_set)

        from .nn_models import BPR_Model

        self.model = BPR_Model(
            user_num=self.total_users,
            item_num=self.total_items + 1,
            factor_num=self.embedding_dim,
            device=self.device,
        )
        self.pad_idx = self.total_items

        if pretrained_model is not None:
            self.model.load_state_dict(pretrained_model.state_dict())

    def fit_mini_batch(self, in_uids, in_iids, out_iids, criteria=None, **kwargs):
        import torch

        criteria = criteria or self.bpr_loss
        batch_size = len(in_uids)

        training_indices = out_iids != self.pad_idx
        breakpoint()
        in_uids = in_uids[training_indices]
        out_iids = out_iids[training_indices]

        u_bpr = self.model.embed_user(in_uids)
        i_bpr = self.model.embed_item(out_iids)

        # return hidden

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)
        import torch
        from torch.utils.tensorboard import SummaryWriter

        from .nn_models import BPR_Model, IndexedAdagradM, uio_iter

        writer = SummaryWriter(comment=f"{self.name}_emb_{self.embedding_dim}_lr_{self.learning_rate}")
        torch_init_seed(self.seed)

        self.model = BPR_Model(
            user_num=self.total_users,
            item_num=self.total_items + 1,
            factor_num=self.embedding_dim,
            device=self.device,
        )

        if not self.trainable:
            return self

        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum)

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
                    rng=self.rng,
                )
            ):
                # BPR user embeddings
                self.model.zero_grad()
                u_bpr = self.model.embed_user(torch.tensor(in_uids, dtype=torch.int64, device=self.device))
                i_bpr = self.model.embed_item(torch.tensor(out_iids, device=self.device))
                bpr_item_scores = torch.mm(u_bpr, i_bpr.T)  # batch_size x batch_size (uids x out_iids)

                # loss
                # r_hi - r_hj
                loss = self.bpr_loss(item_scores=bpr_item_scores)
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
            user_emb = self.model.embed_user(torch.tensor([user_idx], dtype=torch.int64, device=self.device))

            # aggregate all item embeddings
            item_embs = self.model.embed_item.weight[:-1]

            # score
            scores = torch.matmul(item_embs, user_emb.squeeze())
            return scores.cpu().numpy()
