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

from cornac.metrics import NDCG, Recall
from cornac.utils import get_rng

from . import MoE_Base


class MoSE(MoE_Base):
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
        name="MoSE",
        embedding_dim=100,
        loss="cross-entropy",
        batch_size=512,
        learning_rate=0.05,
        momentum=0.0,
        sample_alpha=0.5,
        n_sample=2048,
        n_epochs=10,
        device="cpu",
        trainable=True,
        verbose=False,
        seed=None,
        mode="last",
        model_selection="best",
        init_params=None,
        experts=[],
        num_top_experts=None,
        num_gate=None,
        expert_loss=[],
        max_len=200,
        warmup_gate=5,
        warmup_expert=5,
        tau=1.0,
        inference="sparse",
        gate_type="single",
    ):
        super().__init__(
            name=name,
            embedding_dim=embedding_dim,
            loss=loss,
            batch_size=batch_size,
            learning_rate=learning_rate,
            momentum=momentum,
            sample_alpha=sample_alpha,
            n_sample=n_sample,
            n_epochs=n_epochs,
            device=device,
            trainable=trainable,
            verbose=verbose,
            seed=seed,
            mode=mode,
            model_selection=model_selection,
            init_params=init_params,
            experts=experts,
            num_top_experts=num_top_experts,
            num_gate=num_gate,
            expert_loss=expert_loss,
            max_len=max_len,
            warmup_gate=warmup_gate,
            warmup_expert=warmup_expert,
            tau=tau,
            inference=inference,
            gate_type=gate_type,
        )

        self.cached_gate_values = []  # debugging purpose
        self.expert_choice = np.zeros(len(experts), dtype=int)

    def get_gate_input(self, in_uids, in_iids, **kwargs):
        return super().get_gate_input(in_uids, in_iids, **kwargs)

    def get_hidden(self, in_uids, h_session, **kwargs):
        return super().get_hidden(in_uids, h_session, **kwargs)

    def experts_zero_grad(self):
        for expert in self.experts:
            expert.model.zero_grad()

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        if not self.trainable:
            return self

        import torch

        from .nn_models import IndexedAdagradM, uio_iter

        if self.loss == "bpr":
            loss_fn = self.bpr_loss
        else:
            loss_fn = self.bpr_max_loss

        opt = IndexedAdagradM(self.moe.parameters(), self.learning_rate, self.momentum)

        best_epoch_id = 0
        best_recall = 0
        best_ndcg = 0
        best_val_loss = +np.inf
        progress_bar = trange(1, self.n_epochs + 1, disable=not self.verbose)
        for epoch_id in progress_bar:
            if epoch_id <= self.warmup_expert:
                self.gate.requires_grad_(False)
            else:
                self.gate.requires_grad_(True)

            if self.warmup_expert < epoch_id <= self.warmup_gate + self.warmup_expert:
                for expert in self.experts:
                    expert.model.requires_grad_(False)
            else:
                for expert in self.experts:
                    expert.model.requires_grad_(True)

            for expert in self.experts:
                if hasattr(expert, "reset_new_epoch"):
                    expert.reset_new_epoch()
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
                    rng=self.rng,
                    shuffle=True,
                    max_len=self.max_len,
                )
            ):
                in_uids = torch.tensor(in_uids, requires_grad=False, device=self.device)
                in_iids = torch.tensor(in_iids, requires_grad=False, device=self.device)
                out_iids = torch.tensor(out_iids, requires_grad=False, device=self.device)
                history_iids = torch.tensor(history_iids, requires_grad=False, device=self.device)

                # get gate inputs
                gate_inputs = self.get_gate_input(in_uids, in_iids)
                gate_value = self.gate(gate_inputs)

                # get experts' hidden
                self.experts_zero_grad()
                all_item_scores = []
                for expert in self.experts:
                    item_scores = expert.get_item_scores(
                        in_uids, in_iids, out_iids, start_mask, valid_id, history_iids
                    )
                    all_item_scores.append(item_scores)

                all_item_scores = torch.stack(all_item_scores, dim=1)
                scores = torch.einsum("ben,be->bn", all_item_scores, gate_value)

                opt.zero_grad()
                loss = loss_fn(scores)
                loss.backward()
                opt.step()

                total_loss += loss.cpu().detach().numpy() * len(in_iids)

                cnt += len(in_iids)

                if inc % 10 == 0:
                    progress_bar.set_postfix(loss=(total_loss / cnt))
            self.writer.add_scalar("Loss/train", total_loss / cnt, epoch_id)

            # Evaluate the model on the validation set
            if self.model_selection == "best" and val_set is not None:
                from .eval import ranking_eval

                [current_val_ndcg], _ = ranking_eval(
                    model=self,
                    metrics=[NDCG(k=20)],
                    train_set=train_set,
                    test_set=val_set,
                    mode=self.mode,
                )
                self.writer.add_scalar("NDCG@20/val", current_val_ndcg, epoch_id)
                self.last_model = deepcopy(self.moe)
                if current_val_ndcg > best_recall:
                    best_epoch_id = epoch_id
                    best_ndcg = current_val_ndcg
                    self.best_model = self.last_model

            for expert in self.experts:
                if hasattr(expert, "cleanup"):
                    expert.cleanup()

        if self.model_selection == "best":
            self.moe = self.best_model
            print("#" * 10, "\n", f"Best model found at epoch {best_epoch_id}", "\n", "#" * 10)
        return self

    def score(self, user_idx, history_items, **kwargs):
        import torch

        with torch.no_grad():
            all_item_scores = []

            in_uids = torch.tensor([user_idx], dtype=torch.int64, device=self.device)
            if len(history_items) > 0 and len(history_items[-1]) > 0:
                in_iid_seq = torch.tensor(history_items[-1], device=self.device)
            else:
                in_iid_seq = torch.tensor([self.total_items], device=self.device)

            gate_inputs = self.get_gate_input(in_uids, in_iid_seq[-1:])
            if self.gate_type == "single":
                gate_value = self.gate(gate_inputs)  # 1 x num_experts
                weights, selected_experts = torch.topk(gate_value, k=self.num_top_experts)
            else:
                gate_value, selected_experts = self.gate(gate_inputs, return_top_expert=True)
                weights = gate_value[:, selected_experts[0]]

            self.cached_gate_values.append(gate_value.detach().squeeze().cpu().numpy())
            self.expert_choice[selected_experts.detach().squeeze().cpu().numpy()] += 1

            for e_idx in selected_experts.squeeze(0):
                scores = self.experts[e_idx].score(user_idx, history_items)
                all_item_scores.append(scores)

            all_item_scores = np.stack(all_item_scores, axis=0)
            scores = np.einsum("en,be->bn", all_item_scores, weights.cpu().numpy()).squeeze()

            return scores
