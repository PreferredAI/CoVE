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

import numpy as np

from cornac.utils import get_rng

from . import Gate, NextRecommender


class MoE_Base(NextRecommender):
    """MoE: Mixture of Experts for Next-Item Recommendation

    Parameters
    ----------
    name: string, default: 'MoE'
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
        name="MoE",
        embedding_dim=100,
        loss="cross-entropy",
        batch_size=512,
        learning_rate=0.05,
        momentum=0.0,
        sample_alpha=0.5,
        n_sample=2048,
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
        experts=[],
        num_top_experts=None,
        num_gate=1,
        expert_loss=[],
        max_len=200,
        warmup_gate=5,
        warmup_expert=5,
        tau=1.0,
        inference="sparse",
        gate_type="single",
    ):
        super().__init__(name, trainable=trainable, verbose=verbose)

        assert len(experts) > 0, "At least one expert model should be provided."

        self.embedding_dim = embedding_dim
        self.loss = loss
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.sample_alpha = sample_alpha
        self.n_sample = n_sample
        self.n_epochs = n_epochs
        self.bpreg = bpreg
        self.elu_param = elu_param
        self.logq = logq
        self.seed = seed
        self.rng = get_rng(seed)
        self.mode = mode
        self.model_selection = model_selection
        self.init_params = init_params if init_params is not None else {}
        self.experts = experts
        self.expert_loss = expert_loss
        self.num_expert = len(self.experts)
        self.num_top_experts = num_top_experts if num_top_experts else int(np.ceil(self.num_expert / 3))
        self.num_gate = num_gate if num_gate else self.num_expert // self.num_top_experts
        self.max_len = max_len
        self.warmup_gate = warmup_gate
        self.warmup_expert = warmup_expert
        self.device = device
        self.tau = tau
        self.inference = inference
        self.gate_type = gate_type

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

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.

        Parameters
        ----------
        save_dir: str, default: None
            Path to a directory for the model to be stored.

        """
        import torch

        if save_dir is None:
            return

        writer = self.writer
        del self.writer

        model_file = NextRecommender.save(self, save_dir)

        self.writer = writer
        return model_file

    def combine_scores(self, bpr_item_scores=None, gru_item_scores=None, gate_value=None, normalize=True):
        import torch.nn.functional as F

        if normalize:
            bpr_item_scores = F.normalize(bpr_item_scores, p=2, dim=1) if bpr_item_scores is not None else None
            gru_item_scores = F.normalize(gru_item_scores, p=2, dim=1) if gru_item_scores is not None else None

        if gate_value is None:
            if bpr_item_scores is not None:  # only bpr
                item_scores = bpr_item_scores
            elif gru_item_scores is not None:  # only gru
                item_scores = gru_item_scores
            else:
                raise ValueError("At least one of bpr_item_scores or gru_item_scores should be provided.")
        else:
            # aggregate based on gate_value
            item_scores = bpr_item_scores * gate_value[:, :1] + gru_item_scores * gate_value[:, 1:]

        return item_scores

    def get_gate_input(self, in_uids, in_iids, **kwargs):
        """Return the input for gate. Stack all expert's individual input."""
        # return output: embedding_dim*num_expert
        return [expert.get_gate_input(in_uids, in_iids, **kwargs) for expert in self.experts]

    def get_hidden(self, in_uids, h_session, **kwargs):
        # return output: embedding_dim*num_expert
        return [expert.get_hidden(in_uids, h_session, **kwargs) for expert in self.experts]

    def fit(self, train_set, val_set=None):
        super().fit(train_set, val_set)

        if not self.trainable:
            return self

        import torch
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(comment=f"{self.name}_emb_{self.embedding_dim}_lr_{self.learning_rate}")

        # print all expert names
        print(f"{expert.name} \n" for expert in self.experts)

        self.gate = Gate(
            embedding_dim=self.embedding_dim,
            num_expert=self.num_expert,
            num_top_expert=self.num_top_experts,
            num_gate=self.num_gate,
            tau=self.tau,
            type=self.gate_type,
            device=self.device,
        )

        # self.gate.requires_grad_(False)

        for expert in self.experts:
            if not hasattr(expert, "model"):
                expert._init_model(train_set)

        self.moe = torch.nn.ModuleList([expert.model for expert in self.experts] + [self.gate])
