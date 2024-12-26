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


from cornac.models.recommender import NextItemRecommender


class NextRecommender(NextItemRecommender):
    """Abstract class for triple-next recommendation models."""

    def __init__(self, name="NextRecommender", trainable=True, verbose=False):
        super().__init__(name, trainable=trainable, verbose=verbose)

    def get_gate_input(self, in_uids, in_iids, **kwargs):
        raise NotImplementedError()

    def get_hidden(self, in_uids, in_iids, out_iids, negatives, **kwargs):
        raise NotImplementedError()

    def combine_experts(self, experts, gate_output, **kwargs):
        raise NotImplementedError()

    def get_discounted_avg_items(self, item_embeddings, in_iids, pad_idx, discount=1.0, **kwargs):
        import torch

        # in_iids: (batch_size, seq_len)

        indices = torch.arange(in_iids.shape[1], 0, -1, dtype=torch.float32)
        weights = torch.pow(discount, -indices)
        weights = torch.stack([weights] * in_iids.shape[0], dim=0)
        weights[in_iids == pad_idx] = 0
        weights /= torch.sum(weights, dim=1, keepdim=True)

        return torch.sum(item_embeddings(in_iids).detach() * weights.unsqueeze(-1).to(self.device), dim=1)

    def get_item_embeddings(self, item_embeddings, in_iids, pad_idx, discount=1.0, **kwargs):
        # in_iids: (batch_size)
        return item_embeddings(in_iids).detach()

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

    def fit_mini_batch(self, in_uids, in_iids, out_iids, negatives, criteria=None, **kwargs):
        raise NotImplementedError()

    def score(self, user_idx, history_items, **kwargs):
        raise NotImplementedError()

    def bpr_loss(self, item_scores):
        import torch
        import torch.nn.functional as F

        pos_item_scores = torch.diag(item_scores)
        pos_item_scores = pos_item_scores.reshape(pos_item_scores.shape[0], -1)

        # Vectorized loss calculation
        logits = F.logsigmoid(pos_item_scores - item_scores)

        # remove diagonal of logits
        loss = -torch.sum(logits * (1.0 - torch.eye(*logits.shape, out=torch.empty_like(logits))))
        return loss / logits.size(0) / max(logits.size(1) - 1, 1)

    def softmax_neg(self, X):
        import torch

        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        if e_x.size(0) == 1:
            return e_x
        return e_x / e_x.sum(dim=1, keepdim=True)

    def bpr_max_loss(self, item_scores, elu_param=0.5):
        import torch
        import torch.nn.functional as F

        if elu_param > 0:
            item_scores = F.elu(item_scores, elu_param)
        softmax_scores = self.softmax_neg(item_scores)
        target_scores = torch.diag(item_scores)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        return torch.sum(
            (
                -torch.log(torch.sum(torch.sigmoid(target_scores - item_scores) * softmax_scores, dim=1) + 1e-24)
                + self.bpreg * torch.sum((item_scores**2) * softmax_scores, dim=1)
            )
        ) / item_scores.size(0)

    def xe_loss_with_softmax(self, item_scores, out_iids=None, batch_size=None):
        # item_scores: (batch_size, batch_size+n_samples)s

        import torch

        X = torch.exp(item_scores - item_scores.max(dim=1, keepdim=True)[0])
        X = X / X.sum(dim=1, keepdim=True)
        return -torch.sum(torch.log(torch.diag(X) + 1e-24))
