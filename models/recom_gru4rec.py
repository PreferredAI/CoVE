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


class GRU4Rec(NextRecommender):
    """GRU4Rec: Mixture of Experts for Next-Item Recommendation

    Parameters
    ----------
    name: string, default: 'GRU4Rec'
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
        name="GRU4Rec",
        layers=[100],
        loss="cross-entropy",
        batch_size=512,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        learning_rate=0.05,
        momentum=0.0,
        sample_alpha=0.5,
        n_sample=2048,
        item_discount=1.0,
        embedding=0,
        constrained_embedding=True,
        n_epochs=10,
        bpreg=1.0,
        elu_param=0.5,
        logq=0.0,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        mode="last",
        model_selection="best",
        init_params=None,
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)
        self.layers = layers
        self.loss = loss
        self.batch_size = batch_size
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        self.discount = item_discount
        self.embedding = self.layers[0] if embedding == "layersize" else embedding
        self.constrained_embedding = constrained_embedding
        self.n_epochs = n_epochs
        self.bpreg = bpreg
        self.elu_param = elu_param
        self.logq = logq
        self.device = device
        self.seed = seed
        self.rng = get_rng(seed)
        self.mode = mode
        self.model_selection = model_selection
        self.init_params = {} if init_params is None else init_params

    def get_gate_input(self, user_idx, in_iids, **kwargs):
        # return self.get_discounted_avg_items(self.model.Wy, in_iids, self.total_items, self.discount, **kwargs)
        return self.get_item_embeddings(self.model.Wy, in_iids, self.total_items, self.discount, **kwargs)

    def get_gate_input_sequence(self, user_idx, h_session, **kwargs):
        return h_session

    def _init_model(self, train_set, pretrained_model=None):
        super().fit(train_set)

        import torch

        from .nn_models import GRU4RecModel

        item_freq = Counter(self.train_set.uir_tuple[1])
        P0 = (
            torch.tensor(
                [item_freq[iid] for (_, iid) in self.train_set.iid_map.items()],
                dtype=torch.float32,
                device=self.device,
            )
            if self.logq > 0
            else None
        )

        self.model = GRU4RecModel(
            n_items=self.total_items,
            P0=P0,
            layers=self.layers,
            n_sample=self.n_sample,
            dropout_p_embed=self.dropout_p_embed,
            dropout_p_hidden=self.dropout_p_hidden,
            embedding=self.embedding,
            constrained_embedding=self.constrained_embedding,
            logq=self.logq,
            sample_alpha=self.sample_alpha,
            bpreg=self.bpreg,
            elu_param=self.elu_param,
            loss=self.loss,
        ).to(self.device)

        self.reset_new_epoch()

        if pretrained_model is not None:
            self.model.load_state_dict(pretrained_model.state_dict())

    def reset_new_epoch(self):
        import torch

        self.H = []
        for i in range(len(self.layers)):
            self.H.append(
                torch.zeros(
                    (self.batch_size, self.layers[i]),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                )
            )

    def cleanup(self):
        del self.H

    def get_hidden(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        for i in range(len(self.H)):
            self.H[i][np.nonzero(start_mask)[0], :] = 0
            self.H[i].detach_()
            # breakpoint()
            self.H[i] = self.H[i][valid_id]

        h_gru, i_gru, b_gru = self.model.forward(in_iids, self.H, out_iids, return_hidden=True, training=True)
        return h_gru, i_gru, b_gru

    def get_item_scores(self, in_uids, in_iids, out_iids, start_mask, valid_id, history_iids, **kwargs):
        for i in range(len(self.H)):
            self.H[i][np.nonzero(start_mask)[0], :] = 0
            self.H[i].detach_()
            self.H[i] = self.H[i][valid_id]

        self.model.zero_grad()
        scores = self.model.forward(in_iids, self.H, out_iids, return_hidden=False, training=True)

        return scores

    def get_pred_hidden(self, in_uids, history_iids):
        import torch

        H = []
        for i in range(len(self.layers)):
            H.append(
                torch.zeros(
                    (1, self.model.layers[i]),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                )
            )

        for iid in history_iids:
            h_gru, i_gru, b_gru = self.model.forward(
                torch.tensor([iid], device=self.device),
                H,
                torch.arange(self.total_items, device=self.device),
                return_hidden=True,
                training=False,
            )

        return h_gru, i_gru, b_gru

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        if not self.trainable:
            return self

        import torch
        from torch.utils.tensorboard import SummaryWriter

        from .nn_models import GRU4RecModel, IndexedAdagradM, uio_iter

        item_freq = Counter(self.train_set.uir_tuple[1])
        writer = SummaryWriter(comment=f"{self.name}_emb_{self.layers[0]}_lr_{self.learning_rate}")
        P0 = (
            torch.tensor(
                [item_freq[iid] for (_, iid) in self.train_set.iid_map.items()],
                dtype=torch.float32,
                device=self.device,
            )
            if self.logq > 0
            else None
        )

        self.model = GRU4RecModel(
            n_items=self.total_items,
            P0=P0,
            layers=self.layers,
            n_sample=self.n_sample,
            dropout_p_embed=self.dropout_p_embed,
            dropout_p_hidden=self.dropout_p_hidden,
            embedding=self.embedding,
            constrained_embedding=self.constrained_embedding,
            logq=self.logq,
            sample_alpha=self.sample_alpha,
            bpreg=self.bpreg,
            elu_param=self.elu_param,
            loss=self.loss,
        ).to(self.device)

        if "pretrained_model" in self.init_params:
            self.model.from_pretrained(self.init_params["pretrained_model"])
        else:
            self.model._reset_weights_to_compatibility_mode()

        opt = IndexedAdagradM(self.model.parameters(), self.learning_rate, self.momentum)

        best_recall = 0
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            H = []
            for i in range(len(self.layers)):
                H.append(
                    torch.zeros(
                        (self.batch_size, self.layers[i]),
                        dtype=torch.float32,
                        requires_grad=False,
                        device=self.device,
                    )
                )
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
                    rng=self.rng,
                    shuffle=True,
                )
            ):
                # GRU4Rec iter
                for i in range(len(H)):
                    H[i][np.nonzero(start_mask)[0], :] = 0
                    H[i].detach_()
                    H[i] = H[i][valid_id]
                in_iids = torch.tensor(in_iids, requires_grad=False, device=self.device)
                out_iids = torch.tensor(out_iids, requires_grad=False, device=self.device)
                self.model.zero_grad()
                R = self.model.forward(in_iids, H, out_iids, return_hidden=False, training=True)
                loss = self.model.loss_function(R, out_iids, len(in_iids)) / len(in_iids)
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
            H = []
            for i in range(len(self.model.layers)):
                H.append(
                    torch.zeros(
                        (1, self.model.layers[i]), dtype=torch.float32, requires_grad=False, device=self.device
                    )
                )
            if len(history_items) > 0 and len(history_items[-1]) > 0:
                for iid in history_items[-1]:
                    h_gru = self.model.forward(
                        torch.tensor([iid], device=self.device), H, None, return_hidden=False, training=False
                    )
            else:
                h_gru = self.model.forward(
                    torch.tensor([self.total_items], device=self.device), H, None, return_hidden=False, training=False
                )

            # remove padding item
            item_scores = h_gru.squeeze()[:-1].cpu().numpy()

            return item_scores
