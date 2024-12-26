from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Optimizer
from transformers import BertConfig, BertModel, GPT2Config, GPT2Model

from cornac.utils.common import get_rng


class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, bert_config, add_head=True, tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.vocab_size = vocab_size
        self.bert_config = bert_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=bert_config["hidden_size"], padding_idx=padding_idx
        )
        self.transformer_model = BertModel(BertConfig(**bert_config))

        if self.add_head:
            self.head = nn.Linear(bert_config["hidden_size"], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    def forward(self, input_ids, attention_mask):

        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(inputs_embeds=embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs


class GPT4Rec(nn.Module):

    def __init__(self, vocab_size, gpt_config, add_head=True, tie_weights=True, padding_idx=0, init_std=0.02):

        super().__init__()

        self.vocab_size = vocab_size
        self.gpt_config = gpt_config
        self.add_head = add_head
        self.tie_weights = tie_weights
        self.padding_idx = padding_idx
        self.init_std = init_std

        self.embed_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=gpt_config["n_embd"], padding_idx=padding_idx
        )
        self.transformer_model = GPT2Model(GPT2Config(**gpt_config))

        if self.add_head:
            self.head = nn.Linear(gpt_config["n_embd"], vocab_size, bias=False)
            if self.tie_weights:
                self.head.weight = self.embed_layer.weight

        self.init_weights()

    def init_weights(self):

        self.embed_layer.weight.data.normal_(mean=0.0, std=self.init_std)
        if self.padding_idx is not None:
            self.embed_layer.weight.data[self.padding_idx].zero_()

    def forward(self, input_ids, attention_mask):

        embeds = self.embed_layer(input_ids)
        transformer_outputs = self.transformer_model(inputs_embeds=embeds, attention_mask=attention_mask)
        outputs = transformer_outputs.last_hidden_state

        if self.add_head:
            outputs = self.head(outputs)

        return outputs


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        # self.conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        # self.dropout1 = nn.Dropout(p=dropout_rate)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        # self.dropout2 = nn.Dropout(p=dropout_rate)
        conv1 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        dropout1 = nn.Dropout(p=dropout_rate)
        relu = nn.ReLU()
        conv2 = nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        dropout2 = nn.Dropout(p=dropout_rate)

        self.process = nn.Sequential(conv1, dropout1, relu, conv2, dropout2)

    def forward(self, inputs):
        # outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = self.process(inputs.transpose(-1, -2))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


class SASRecModel(nn.Module):
    def __init__(
        self,
        user_num,
        item_num,
        embedding_dim=100,
        maxlen=20,
        n_layers=2,
        n_heads=1,
        use_pos_emb=True,
        use_biases=True,
        dropout=0.2,
        pad_idx=-1,
        initializer_range=0.02,
        add_head=True,
        device="cpu",
    ):
        super(SASRecModel, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.pad_idx = pad_idx if pad_idx >= 0 else item_num
        self.maxlen = maxlen
        self.dev = device
        self.initializer_range = initializer_range
        self.add_head = add_head

        self.item_emb = nn.Embedding(self.item_num + 1, embedding_dim, padding_idx=pad_idx)
        if use_pos_emb:
            self.pos_emb = nn.Embedding(maxlen + 1, embedding_dim)
        if use_biases:
            self.item_biases = nn.Embedding(self.item_num + 1, 1, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(p=dropout)

        self.attention_layernorms = nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        self.last_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)

        for _ in range(n_layers):
            new_attn_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = nn.MultiheadAttention(embedding_dim, n_heads, dropout)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = nn.LayerNorm(embedding_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(embedding_dim, dropout)
            self.forward_layers.append(new_fwd_layer)

        self.apply(self._init_weights)
        self.to(device)

    def _init_weights(self, module):
        """Initialize weights.

        Examples:
        https://github.com/huggingface/transformers/blob/v4.25.1/src/transformers/models/gpt2/modeling_gpt2.py#L454
        https://recbole.io/docs/_modules/recbole/model/sequential_recommender/sasrec.html#SASRec
        """

        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def score_items(self, hidden, cand_items, B=None):
        # hidden: (B, D)
        # cand_items: (B+N, D) where N is the number of negative samples
        # B: (B+N, 1) bias
        # return: (B, B+N) scores
        scores = torch.mm(hidden, cand_items.T)
        if B is not None:
            return scores + B.T
        return scores

    def log2feats(self, log_seqs):
        if hasattr(log_seqs, "device"):
            log_seqs = log_seqs.unsqueeze(0).cpu().numpy()
            pad_seqs = np.ones((1, self.maxlen)) * self.pad_idx
            log_seqs = np.concatenate([pad_seqs, log_seqs], axis=1)[:, -self.maxlen :]
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim**0.5

        if hasattr(self, "pos_emb"):
            poss = np.tile(np.arange(1, log_seqs.shape[1] + 1), [log_seqs.shape[0], 1])
            poss *= log_seqs != self.pad_idx
            seqs += self.pos_emb(torch.LongTensor(poss).to(self.dev))
        seqs = self.emb_dropout(seqs)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, hist_iids, out_iids, return_hidden=True, return_scores=False):
        seqs = self.item_emb(hist_iids)
        seqs *= self.item_emb.embedding_dim**0.5
        positions = np.tile(np.array(range(hist_iids.shape[1])), [hist_iids.shape[0], 1])
        # need to be on the same device
        if hasattr(self, "pos_emb"):
            seqs += self.pos_emb(torch.LongTensor(positions).to(seqs.device))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.Tensor(hist_iids == self.pad_idx)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        # need to be on the same device
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool).to(seqs.device))

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        hidden = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        hidden = hidden[:, -1, :]  # only use the last hidden state

        if return_hidden and return_scores:
            return hidden, self.score_items(hidden, self.item_emb(out_iids), self.item_biases(out_iids))
        if return_hidden:
            return hidden, self.item_emb(out_iids), self.item_biases(out_iids)

        return self.score_items(hidden, self.item_emb(out_iids), self.item_biases(out_iids))

    @torch.no_grad()
    def predict(self, user_ids, log_seqs, item_indices=None):
        if item_indices is None:
            item_indices = torch.arange(self.item_num).to(self.dev)
        else:
            item_indices = torch.Tensor(item_indices).to(self.dev)

        log_feats = self.log2feats(log_seqs)
        final_feat = log_feats[:, -1, :]
        item_embs = self.item_emb(item_indices)

        logits = torch.matmul(item_embs, final_feat.view(-1, 1)).squeeze()

        return logits.cpu().numpy()  # except the last one, which is padding


class Gate(nn.Module):
    def __init__(
        self,
        embedding_dim=100,
        num_expert=2,
        num_top_expert=1,
        num_gate=1,
        tau=None,
        type="single",
        device="cpu",
    ):
        """_summary_

        Args:
            embedding_dim (int, optional): _description_. Defaults to 100.
            num_expert (int, optional): _description_. Defaults to 2.
            tau (_type_, optional): _description_. Defaults to None.
            type (str, optional): Gate type, choices=["single", "multi"]. Defaults to "single".
            device (str, optional): _description_. Defaults to "cpu".
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_top_expert = num_top_expert
        self.num_gate = num_gate
        self.tau = tau
        self.type = type

        if type == "single":
            self.gate = nn.Linear(embedding_dim * num_expert, num_expert, bias=False)
        elif type == "multi":
            self.n_expert_per_gate = num_expert // num_gate
            self.num_expert = num_expert
            self.offset_idx = (
                torch.arange(0, self.num_expert, self.n_expert_per_gate).view(1, self.num_gate).to(device)
            )

            self.lower_gate_dim = self.n_expert_per_gate * self.embedding_dim
            self.lower_gates = nn.ModuleList(
                [
                    nn.Linear(embedding_dim * self.n_expert_per_gate, self.n_expert_per_gate, bias=False)
                    for _ in range(num_gate)
                ]
            )
            self.top_gate = nn.Linear(embedding_dim * num_gate, num_gate, bias=False)
            self.gate = nn.ModuleList(self.lower_gates + [self.top_gate])

        self.gate.to(device)

    def forward(self, embs, return_top_expert=False):
        # embs: (embedding_dim, num_expert)
        emb = torch.cat(embs, dim=1)

        if self.type == "single":
            h_gate = self.gate(emb)

            # here because gumbel_softmax is not available as nn.Module
            if self.tau is None or self.tau == 1.0:
                h_gate = F.softmax(h_gate, dim=1)
            else:
                h_gate = F.gumbel_softmax(h_gate, tau=self.tau, dim=1)
            return h_gate
        else:
            h_gates = torch.stack(
                [
                    gate(emb[:, idx * self.lower_gate_dim : (idx + 1) * self.lower_gate_dim])
                    for idx, gate in enumerate(self.lower_gates)
                ],
                dim=1,
            )

            if self.tau is None or self.tau == 1.0:
                h_gates = F.softmax(h_gates, dim=2)
            else:
                h_gates = F.gumbel_softmax(h_gates, tau=self.tau, dim=2)

            # emb: (batch_size, num_expert * embedding_dim)
            # selected_expert_id: (batch_size, num_gate)
            # mask the non-selected expert emb to zero

            gate_input = emb.view(-1, self.num_gate, self.n_expert_per_gate, self.embedding_dim)
            gate_input = torch.einsum("bned,bne->bnd", gate_input, h_gates).view(
                -1, self.num_gate * self.embedding_dim
            )
            h_top_gate = self.top_gate(gate_input)

            if self.tau is None or self.tau == 1.0:
                h_top_gate = F.softmax(h_top_gate, dim=1)
            else:
                h_top_gate = F.gumbel_softmax(h_top_gate, tau=self.tau, dim=1)
            gate_values = torch.einsum("bne,bn->bne", h_gates, h_top_gate).view(-1, self.num_expert)

            if return_top_expert:
                selected_expert_id = torch.topk(h_top_gate, self.num_top_expert, dim=1)[1]
                selected_expert_id = torch.concatenate(
                    [selected_expert_id * self.n_expert_per_gate + offset for offset in range(self.n_expert_per_gate)],
                    dim=1,
                )
                return gate_values, selected_expert_id

            return gate_values


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout_rate=0.5):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout_rate)  # Change the dropout rate as needed

    def forward(self, queries, keys, causality=False):
        Q = self.query_proj(queries)
        K = self.key_proj(keys)
        V = self.val_proj(keys)

        # Split and concat
        Q_ = torch.cat(Q.chunk(self.num_heads, dim=2), dim=0)
        K_ = torch.cat(K.chunk(self.num_heads, dim=2), dim=0)
        V_ = torch.cat(V.chunk(self.num_heads, dim=2), dim=0)

        # Multiplication
        outputs = torch.matmul(Q_, K_.transpose(1, 2))

        # Scale
        outputs = outputs / (K_.size(-1) ** 0.5)

        # Key Masking
        key_masks = torch.sign(torch.sum(torch.abs(keys), dim=-1))
        key_masks = key_masks.repeat(self.num_heads, 1)
        key_masks = key_masks.unsqueeze(1).repeat(1, queries.size(1), 1)

        outputs = outputs.masked_fill(key_masks == 0, float("-inf"))

        # Causality
        if causality:
            diag_vals = torch.ones_like(outputs[0])
            tril = torch.tril(diag_vals)
            masks = tril[None, :, :].repeat(outputs.size(0), 1, 1)

            outputs = outputs.masked_fill(masks == 0, float("-inf"))

        # Activation
        outputs = F.softmax(outputs, dim=-1)
        outputs = torch.nan_to_num(outputs, nan=0.0, posinf=0.0, neginf=0.0)

        # Query Masking
        query_masks = torch.sign(torch.sum(torch.abs(queries), dim=-1))
        query_masks = query_masks.repeat(self.num_heads, 1)
        query_masks = query_masks.unsqueeze(-1).repeat(1, 1, keys.size(1))

        outputs *= query_masks

        attention_chunks = outputs.chunk(self.num_heads, dim=0)
        attention_weights = torch.stack(attention_chunks, dim=1)

        # Dropouts
        outputs = self.dropout(outputs)

        # Weighted sum
        outputs = torch.matmul(outputs, V_)

        # Restore shape
        outputs = torch.cat(outputs.chunk(self.num_heads, dim=0), dim=2)
        return outputs, attention_weights


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, dropout_rate=0.5, causality=True):
        super(TransformerBlock, self).__init__()

        self.first_norm = nn.LayerNorm(dim)
        self.second_norm = nn.LayerNorm(dim)

        self.multihead_attention = MultiHeadAttention(dim, num_heads, dropout_rate)

        self.dense1 = nn.Linear(dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.causality = causality

    def forward(self, seq, mask=None):
        x = self.first_norm(seq)
        queries = x
        keys = seq
        x, attentions = self.multihead_attention(queries, keys, self.causality)

        # Add & Norm
        x = x + queries
        x = self.second_norm(x)

        # Feed Forward
        residual = x
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        # Add & Norm
        x = x + residual

        # Apply mask if provided
        if mask is not None:
            x *= mask

        return x, attentions


def uio_iter_pad(
    s_iter,
    uir_tuple,
    pad_index,
    batch_size=1,
    n_sample=2048,
    sample_alpha=0.4,
    rng=None,
    shuffle=False,
    max_len=200,  # max_len is the maximum length of the sequence without the last item
):
    """Parallelize mini-batch of input-output items. Create an iterator over data yielding batch of input user indices, batch of input item indices, batch of output item indices,
    batch of start masking, batch of end masking, and batch of valid ids (relative positions of current sequences in the last batch).

    Parameters
    ----------
    batch_size: int, optional, default = 1

    shuffle: bool, optional, default: False
        If `True`, orders of triplets will be randomized. If `False`, default orders kept.

    Returns
    -------
    iterator : batch of input user indices, batch of input item indices, batch of output item indices, batch of starting sequence mask, batch of ending sequence mask, batch of valid ids

    """
    max_len_with_last = max_len + 1
    rng = rng if rng is not None else get_rng(None)
    negative_samples = None
    if n_sample > 0:
        item_count = Counter(uir_tuple[1])
        item_indices = np.array([iid for iid, _ in item_count.most_common()], dtype="int")
        item_dist = np.array([cnt for _, cnt in item_count.most_common()], dtype="float") ** sample_alpha
        item_dist = item_dist / item_dist.sum()
    for _, batch_mapped_ids in s_iter(batch_size, shuffle):
        batch_iids = [uir_tuple[1][mapped_ids[-max_len_with_last:]] for mapped_ids in batch_mapped_ids]
        batch_uids = [uir_tuple[0][mapped_ids[0]] for mapped_ids in batch_mapped_ids]
        batch_iids = [
            ([pad_index] * (max_len_with_last - len(session_items)) + list(session_items))
            for session_items in batch_iids
        ]
        batch_iids = np.array(batch_iids)
        batch_in_iids = batch_iids[:, :-1]
        batch_out_iids = batch_iids[:, 1:]
        if n_sample > 0:
            negative_samples = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
        yield batch_uids, batch_in_iids, batch_out_iids, negative_samples


def softmax_loss(logits=None):
    return -F.log_softmax(logits, dim=-1)[:, :, 0].sum()


class FPMC_Model(nn.Module):
    def __init__(self, user_num, item_num, factor_num, device="cpu"):
        super(FPMC_Model, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.factor_num = factor_num
        self.device = device

        self.UI_emb = nn.Embedding(user_num, factor_num)
        self.IU_emb = nn.Embedding(item_num, factor_num)
        self.LI_emb = nn.Embedding(item_num + 1, factor_num, padding_idx=item_num)
        self.IL_emb = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.UI_emb.weight, std=0.01)
        nn.init.normal_(self.IU_emb.weight, std=0.01)
        nn.init.normal_(self.LI_emb.weight, std=0.01)
        nn.init.normal_(self.IL_emb.weight, std=0.01)

        self.item_biases = nn.Embedding(item_num, 1).to(device)
        nn.init.constant_(self.item_biases.weight, 0)

    def forward(self, in_uids, in_iids, out_iids, **kwargs):
        item_seq_emb = self.LI_emb(in_iids)  # [b,emb]
        user_emb = self.UI_emb(in_uids)  # [b,emb]
        iu_emb = self.IU_emb(out_iids)  # [n_items,emb]; n_items=b+num_neg
        il_emb = self.IL_emb(out_iids)  # [n_items,emb]; n_items=b+num_neg

        # FPMC's core part, can be expressed by a combination of a MF and a FMC model
        mf = torch.einsum("be,ne->bn", user_emb, iu_emb)  # [b,n_items]
        fmc = torch.einsum("ne,be->bn", il_emb, item_seq_emb)  # [b,n_items]

        return mf + fmc  # [b,n_items]

    def predict(self, in_uids, in_iids, candidate_items, **kwargs):
        return self.forward(in_uids, in_iids, candidate_items, **kwargs).squeeze()


def init_parameter_matrix(tensor: torch.Tensor, dim0_scale: int = 1, dim1_scale: int = 1):
    sigma = np.sqrt(6.0 / float(tensor.size(0) / dim0_scale + tensor.size(1) / dim1_scale))
    return nn.init._no_grad_uniform_(tensor, -sigma, sigma)


class IndexedAdagradM(Optimizer):
    def __init__(self, params, lr=0.05, momentum=0.0, eps=1e-6):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if eps <= 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, momentum=momentum, eps=eps)
        super(IndexedAdagradM, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"] = torch.full_like(p, 0, memory_format=torch.preserve_format)
                if momentum > 0:
                    state["mom"] = torch.full_like(p, 0, memory_format=torch.preserve_format)

    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                state["acc"].share_memory_()
                if group["momentum"] > 0:
                    state["mom"].share_memory_()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                clr = group["lr"]
                momentum = group["momentum"]
                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()
                    accs = state["acc"][grad_indices] + grad_values.pow(2)
                    state["acc"].index_copy_(0, grad_indices, accs)
                    accs.add_(group["eps"]).sqrt_().mul_(-1 / clr)
                    if momentum > 0:
                        moma = state["mom"][grad_indices]
                        moma.mul_(momentum).add_(grad_values / accs)
                        state["mom"].index_copy_(0, grad_indices, moma)
                        p.index_add_(0, grad_indices, moma)
                    else:
                        p.index_add_(0, grad_indices, grad_values / accs)
                else:
                    state["acc"].add_(grad.pow(2))
                    accs = state["acc"].add(group["eps"])
                    accs.sqrt_()
                    if momentum > 0:
                        mom = state["mom"]
                        mom.mul_(momentum).addcdiv_(grad, accs, value=-clr)
                        p.add_(mom)
                    else:
                        p.addcdiv_(grad, accs, value=-clr)
        return loss


class GRUEmbedding(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GRUEmbedding, self).__init__()
        self.Wx0 = nn.Embedding(dim_in, dim_out * 3, sparse=True)
        self.Wrz0 = nn.Parameter(torch.empty((dim_out, dim_out * 2), dtype=torch.float))
        self.Wh0 = nn.Parameter(torch.empty((dim_out, dim_out * 1), dtype=torch.float))
        self.Bh0 = nn.Parameter(torch.zeros(dim_out * 3, dtype=torch.float))
        self.reset_parameters()

    def reset_parameters(self):
        init_parameter_matrix(self.Wx0.weight, dim1_scale=3)
        init_parameter_matrix(self.Wrz0, dim1_scale=2)
        init_parameter_matrix(self.Wh0, dim1_scale=1)
        nn.init.zeros_(self.Bh0)

    def forward(self, X, H):
        Vx = self.Wx0(X) + self.Bh0
        Vrz = torch.mm(H, self.Wrz0)
        vx_x, vx_r, vx_z = Vx.chunk(3, 1)
        vh_r, vh_z = Vrz.chunk(2, 1)
        r = torch.sigmoid(vx_r + vh_r)
        z = torch.sigmoid(vx_z + vh_z)
        h = torch.tanh(torch.mm(r * H, self.Wh0) + vx_x)
        h = (1.0 - z) * H + z * h
        return h


class GRU4RecModel(nn.Module):
    def __init__(
        self,
        n_items,
        P0=None,
        layers=[100],
        n_sample=2048,
        dropout_p_embed=0.0,
        dropout_p_hidden=0.0,
        embedding=0,
        constrained_embedding=True,
        logq=0.0,
        sample_alpha=0.5,
        bpreg=1.0,
        elu_param=0.5,
        loss="cross-entropy",
    ):
        super(GRU4RecModel, self).__init__()
        self.n_items = n_items
        self.P0 = P0
        self.layers = layers
        self.dropout_p_embed = dropout_p_embed
        self.dropout_p_hidden = dropout_p_hidden
        self.embedding = embedding
        self.constrained_embedding = constrained_embedding
        self.logq = logq
        self.n_sample = n_sample
        self.sample_alpha = sample_alpha
        self.elu_param = elu_param
        self.bpreg = bpreg
        self.loss = loss
        self.set_loss_function(self.loss)
        self.start = 0
        if constrained_embedding:
            n_input = layers[-1]
        elif embedding:
            self.E = nn.Embedding(n_items + 1, embedding, padding_idx=n_items, sparse=True)
            n_input = embedding
        else:
            self.GE = GRUEmbedding(n_items, layers[0])
            n_input = n_items
            self.start = 1
        self.DE = nn.Dropout(dropout_p_embed)
        self.G = []
        self.D = []
        for i in range(self.start, len(layers)):
            self.G.append(nn.GRUCell(layers[i - 1] if i > 0 else n_input, layers[i]))
            self.D.append(nn.Dropout(dropout_p_hidden))
        self.G = nn.ModuleList(self.G)
        self.D = nn.ModuleList(self.D)
        self.Wy = nn.Embedding(n_items + 1, layers[-1], padding_idx=n_items, sparse=True)
        self.By = nn.Embedding(n_items + 1, 1, padding_idx=n_items, sparse=True)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        if self.embedding:
            init_parameter_matrix(self.E.weight)
        elif not self.constrained_embedding:
            self.GE.reset_parameters()
        for i in range(len(self.G)):
            init_parameter_matrix(self.G[i].weight_ih, dim1_scale=3)
            init_parameter_matrix(self.G[i].weight_hh, dim1_scale=3)
            nn.init.zeros_(self.G[i].bias_ih)
            nn.init.zeros_(self.G[i].bias_hh)
        init_parameter_matrix(self.Wy.weight)
        nn.init.zeros_(self.By.weight)

    def set_loss_function(self, loss):
        if loss == "cross-entropy":
            self.loss_function = self.xe_loss_with_softmax
        elif loss == "bpr-max":
            self.loss_function = self.bpr_max_loss_with_elu
        elif loss == "top1":
            self.loss_function = self.top1
        else:
            raise NotImplementedError

    def xe_loss_with_softmax(self, O, Y, M):
        if self.logq > 0:
            O = O - self.logq * torch.log(torch.cat([self.P0[Y[:M]], self.P0[Y[M:]] ** self.sample_alpha]))
        X = torch.exp(O - O.max(dim=1, keepdim=True)[0])
        X = X / X.sum(dim=1, keepdim=True)
        return -torch.sum(torch.log(torch.diag(X) + 1e-24))

    def softmax_neg(self, X):
        hm = 1.0 - torch.eye(*X.shape, out=torch.empty_like(X))
        X = X * hm
        e_x = torch.exp(X - X.max(dim=1, keepdim=True)[0]) * hm
        if e_x.size(0) == 1:
            return e_x
        return e_x / e_x.sum(dim=1, keepdim=True)

    def bpr_max_loss_with_elu(self, O, Y, M):
        if self.elu_param > 0:
            O = nn.functional.elu(O, self.elu_param)
        softmax_scores = self.softmax_neg(O)
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        return torch.sum(
            (
                -torch.log(torch.sum(torch.sigmoid(target_scores - O) * softmax_scores, dim=1) + 1e-24)
                + self.bpreg * torch.sum((O**2) * softmax_scores, dim=1)
            )
        )

    def top1(self, O, Y, M):
        target_scores = torch.diag(O)
        target_scores = target_scores.reshape(target_scores.shape[0], -1)
        return torch.sum(
            (
                torch.mean(torch.sigmoid(O - target_scores) + torch.sigmoid(O**2), axis=1)
                - torch.sigmoid(target_scores**2) / (M + self.n_sample)
            )
        )

    def _init_numpy_weights(self, shape):
        sigma = np.sqrt(6.0 / (shape[0] + shape[1]))
        m = np.random.rand(*shape).astype("float32") * 2 * sigma - sigma
        return m

    @torch.no_grad()
    def _reset_weights_to_compatibility_mode(self):
        np.random.seed(42)
        if self.constrained_embedding:
            n_input = self.layers[-1]
        elif self.embedding:
            n_input = self.embedding
            self.E.weight.set_(
                torch.tensor(
                    self._init_numpy_weights((self.n_items, n_input)),
                    device=self.E.weight.device,
                )
            )
        else:
            n_input = self.n_items
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            m.append(self._init_numpy_weights((n_input, self.layers[0])))
            self.GE.Wx0.weight.set_(torch.tensor(np.hstack(m), device=self.GE.Wx0.weight.device))
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            m2.append(self._init_numpy_weights((self.layers[0], self.layers[0])))
            self.GE.Wrz0.set_(torch.tensor(np.hstack(m2), device=self.GE.Wrz0.device))
            self.GE.Wh0.set_(
                torch.tensor(
                    self._init_numpy_weights((self.layers[0], self.layers[0])),
                    device=self.GE.Wh0.device,
                )
            )
            self.GE.Bh0.set_(torch.zeros((self.layers[0] * 3,), device=self.GE.Bh0.device))
        for i in range(self.start, len(self.layers)):
            m = []
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            m.append(self._init_numpy_weights((n_input, self.layers[i])))
            self.G[i].weight_ih.set_(
                torch.tensor(np.vstack(m), dtype=torch.float32, device=self.G[i].weight_ih.device)
            )
            m2 = []
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            m2.append(self._init_numpy_weights((self.layers[i], self.layers[i])))
            self.G[i].weight_hh.set_(
                torch.tensor(np.vstack(m2), dtype=torch.float32, device=self.G[i].weight_hh.device)
            )
            self.G[i].bias_hh.set_(torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_hh.device))
            self.G[i].bias_ih.set_(torch.zeros((self.layers[i] * 3,), device=self.G[i].bias_ih.device))
        self.Wy.weight.set_(
            torch.tensor(
                self._init_numpy_weights((self.Wy.weight.shape[0], self.layers[-1])),
                dtype=torch.float32,
                device=self.Wy.weight.device,
            )
        )
        self.By.weight.set_(torch.zeros((self.By.weight.shape[0], 1), device=self.By.weight.device))

    @torch.no_grad()
    def from_pretrained(self, pretrained_model):
        if self.embedding:
            self.E.weight.set_(pretrained_model.E.weight.to(self.E.weight.device))
        elif not self.constrained_embedding:
            self.GE.Wx0.weight.set_(pretrained_model.GE.Wx0.weight.to(self.GE.Wx0.weight.device))
            self.GE.Wrz0.set_(pretrained_model.GE.Wrz0.to(self.GE.Wrz0.device))
            self.GE.Wh0.set_(pretrained_model.GE.Wh0.to(self.GE.Wh0.device))
            self.GE.Bh0.set_(pretrained_model.GE.Bh0.to(self.GE.Bh0.device))
        for i in range(self.start, len(self.layers)):
            self.G[i].weight_ih.set_(pretrained_model.G[i].weight_ih.to(self.G[i].weight_ih.device))
            self.G[i].weight_hh.set_(pretrained_model.G[i].weight_hh.to(self.G[i].weight_hh.device))
            self.G[i].bias_hh.set_(pretrained_model.G[i].bias_hh.to(self.G[i].bias_hh.device))
            self.G[i].bias_ih.set_(pretrained_model.G[i].bias_ih.to(self.G[i].bias_ih.device))
        self.Wy.weight.set_(pretrained_model.Wy.weight.to(self.Wy.weight.device))
        self.By.weight.set_(pretrained_model.By.weight.to(self.By.weight.device))

    def embed_constrained(self, X, Y=None):
        if Y is not None:
            XY = torch.cat([X, Y])
            EXY = self.Wy(XY)
            split = X.shape[0]
            E = EXY[:split]
            O = EXY[split:]
            B = self.By(Y)
        else:
            E = self.Wy(X)
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed_separate(self, X, Y=None):
        E = self.E(X)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed_gru(self, X, H, Y=None):
        E = self.GE(X, H)
        if Y is not None:
            O = self.Wy(Y)
            B = self.By(Y)
        else:
            O = self.Wy.weight
            B = self.By.weight
        return E, O, B

    def embed(self, X, H, Y=None):
        if self.constrained_embedding:
            E, O, B = self.embed_constrained(X, Y)
        elif self.embedding > 0:
            E, O, B = self.embed_separate(X, Y)
        else:
            E, O, B = self.embed_gru(X, H[0], Y)
        return E, O, B

    def hidden_step(self, X, H, training=False):
        for i in range(self.start, len(self.layers)):
            X = self.G[i](X, Variable(H[i]))
            if training:
                X = self.D[i](X)
            H[i] = X
        return X

    def score_items(self, X, O, B):
        res = torch.mm(X, O.T) + B.T
        return res

    def forward(self, X, H, Y, return_hidden=True, return_scores=False, training=False):
        E, O, B = self.embed(X, H, Y)
        if training:
            E = self.DE(E)
        if not (self.constrained_embedding or self.embedding):
            H[0] = E
        Xh = self.hidden_step(E, H, training=training)
        if return_hidden and return_scores:
            return Xh, self.score_items(Xh, O, B)
        if return_hidden:
            return Xh, O, B

        return self.score_items(Xh, O, B)


def io_iter(s_iter, uir_tuple, n_sample=0, sample_alpha=0, rng=None, batch_size=1, shuffle=False):
    """Parallelize mini-batch of input-output items. Create an iterator over data yielding batch of input item indices, batch of output item indices,
    batch of start masking, batch of end masking, and batch of valid ids (relative positions of current sequences in the last batch).

    Parameters
    ----------
    batch_size: int, optional, default = 1

    shuffle: bool, optional, default: False
        If `True`, orders of triplets will be randomized. If `False`, default orders kept.

    Returns
    -------
    iterator : batch of input item indices, batch of output item indices, batch of starting sequence mask, batch of ending sequence mask, batch of valid ids

    """
    rng = rng if rng is not None else get_rng(None)
    start_mask = np.zeros(batch_size, dtype="int")
    end_mask = np.ones(batch_size, dtype="int")
    input_iids = None
    output_iids = None
    l_pool = []
    c_pool = [None for _ in range(batch_size)]
    sizes = np.zeros(batch_size, dtype="int")
    if n_sample > 0:
        item_count = Counter(uir_tuple[1])
        item_indices = np.array([iid for iid, _ in item_count.most_common()], dtype="int")
        item_dist = np.array([cnt for _, cnt in item_count.most_common()], dtype="float") ** sample_alpha
        item_dist = item_dist / item_dist.sum()
    for _, batch_mapped_ids in s_iter(batch_size, shuffle):
        l_pool += batch_mapped_ids
        while len(l_pool) > 0:
            if end_mask.sum() == 0:
                input_iids = uir_tuple[1][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool)]]
                output_iids = uir_tuple[1][[mapped_ids[-sizes[idx] + 1] for idx, mapped_ids in enumerate(c_pool)]]
                sizes -= 1
                for idx, size in enumerate(sizes):
                    if size == 1:
                        end_mask[idx] = 1
                if n_sample > 0:
                    negative_samples = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
                    output_iids = np.concatenate([output_iids, negative_samples])
                yield input_iids, output_iids, start_mask, np.arange(batch_size, dtype="int")
                start_mask.fill(0)  # reset start masking
            while end_mask.sum() > 0 and len(l_pool) > 0:
                next_seq = l_pool.pop()
                if len(next_seq) > 1:
                    idx = np.nonzero(end_mask)[0][0]
                    end_mask[idx] = 0
                    start_mask[idx] = 1
                    c_pool[idx] = next_seq
                    sizes[idx] = len(c_pool[idx])

    valid_id = np.ones(batch_size, dtype="int")
    while True:
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
                valid_id[idx] = 0
        input_iids = uir_tuple[1][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool) if sizes[idx] > 1]]
        output_iids = uir_tuple[1][
            [mapped_ids[-sizes[idx] + 1] for idx, mapped_ids in enumerate(c_pool) if sizes[idx] > 1]
        ]
        sizes -= 1
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
        start_mask = start_mask[np.nonzero(valid_id)[0]]
        end_mask = end_mask[np.nonzero(valid_id)[0]]
        sizes = sizes[np.nonzero(valid_id)[0]]
        c_pool = [_ for _, valid in zip(c_pool, valid_id) if valid > 0]
        if n_sample > 0:
            negative_samples = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
            output_iids = np.concatenate([output_iids, negative_samples])
        yield input_iids, output_iids, start_mask, np.nonzero(valid_id)[0]
        valid_id = np.ones(len(input_iids), dtype="int")
        if end_mask.sum() == len(input_iids):
            break
        start_mask.fill(0)  # reset start masking


def uio_iter(
    s_iter,
    uir_tuple,
    pad_index,
    batch_size=1,
    n_sample=2048,
    sample_alpha=0.4,
    rng=None,
    shuffle=False,
    max_len=20,
):
    """Parallelize mini-batch of input-output items. Create an iterator over data yielding batch of input user indices, batch of input item indices, batch of output item indices,
    batch of start masking, batch of end masking, and batch of valid ids (relative positions of current sequences in the last batch).

    Parameters
    ----------
    batch_size: int, optional, default = 1

    shuffle: bool, optional, default: False
        If `True`, orders of triplets will be randomized. If `False`, default orders kept.

    Returns
    -------
    iterator : batch of input user indices, batch of input item indices, batch of output item indices, batch of starting sequence mask, batch of ending sequence mask, batch of valid ids

    """
    rng = rng if rng is not None else get_rng(None)
    start_mask = np.zeros(batch_size, dtype="int")
    end_mask = np.ones(batch_size, dtype="int")
    input_uids = None
    input_iids = None
    output_iids = None
    history_iids = [[pad_index for _ in range(max_len)] for _ in range(batch_size)]
    l_pool = []
    c_pool = [None for _ in range(batch_size)]
    sizes = np.zeros(batch_size, dtype="int")
    if n_sample > 0:
        item_count = Counter(uir_tuple[1])
        item_indices = np.array([iid for iid, _ in item_count.most_common()], dtype="int")
        item_dist = np.array([cnt for _, cnt in item_count.most_common()], dtype="float") ** sample_alpha
        item_dist = item_dist / item_dist.sum()
    for _, batch_mapped_ids in s_iter(batch_size, shuffle):
        l_pool += batch_mapped_ids
        while len(l_pool) > 0:
            if end_mask.sum() == 0:
                input_uids = uir_tuple[0][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool)]]
                input_iids = uir_tuple[1][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool)]]
                output_iids = uir_tuple[1][[mapped_ids[-sizes[idx] + 1] for idx, mapped_ids in enumerate(c_pool)]]
                sizes -= 1
                for idx, size in enumerate(sizes):
                    if size == 1:
                        end_mask[idx] = 1

                # reset history_iids to [pad_index]*max_len where end_mask is 1
                # otherwise, append input_iids to history_iids
                history_iids = [
                    (([pad_index] * (max_len - 1) + [iid]) if isEnd else (history_iids[cnt] + [iid])[-max_len:])
                    for cnt, (iid, isEnd) in enumerate(zip(input_iids, end_mask))
                ]
                if n_sample > 0:
                    negative_samples = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
                    output_iids = np.concatenate([output_iids, negative_samples])
                yield input_uids, input_iids, output_iids, start_mask, np.arange(batch_size, dtype="int"), np.array(
                    history_iids
                )
                start_mask.fill(0)  # reset start masking
            while end_mask.sum() > 0 and len(l_pool) > 0:
                next_seq = l_pool.pop()
                # pad start of sequence
                next_seq = [pad_index] + next_seq
                if len(next_seq) > 1:
                    idx = np.nonzero(end_mask)[0][0]
                    end_mask[idx] = 0
                    start_mask[idx] = 1
                    c_pool[idx] = next_seq
                    sizes[idx] = len(c_pool[idx])

    valid_id = np.ones(batch_size, dtype="int")
    while True:
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
                valid_id[idx] = 0
        input_uids = uir_tuple[0][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool) if sizes[idx] > 1]]
        input_iids = uir_tuple[1][[mapped_ids[-sizes[idx]] for idx, mapped_ids in enumerate(c_pool) if sizes[idx] > 1]]
        output_iids = uir_tuple[1][
            [mapped_ids[-sizes[idx] + 1] for idx, mapped_ids in enumerate(c_pool) if sizes[idx] > 1]
        ]
        sizes -= 1
        for idx, size in enumerate(sizes):
            if size == 1:
                end_mask[idx] = 1
        start_mask = start_mask[np.nonzero(valid_id)[0]]
        end_mask = end_mask[np.nonzero(valid_id)[0]]
        sizes = sizes[np.nonzero(valid_id)[0]]
        c_pool = [_ for _, valid in zip(c_pool, valid_id) if valid > 0]
        history_iids = [
            (([pad_index] * (max_len - 1) + [iid]) if isEnd else (history_iids[cnt] + [iid])[-max_len:])
            for cnt, (iid, isEnd) in enumerate(zip(input_iids, end_mask))
        ]
        if n_sample > 0:
            negative_samples = rng.choice(item_indices, size=n_sample, replace=True, p=item_dist)
            output_iids = np.concatenate([output_iids, negative_samples])
        yield input_uids, input_iids, output_iids, start_mask, np.nonzero(valid_id)[0], np.array(history_iids)
        valid_id = np.ones(len(input_iids), dtype="int")
        if end_mask.sum() == len(input_iids):
            break
        start_mask.fill(0)  # reset start masking


def score(model, layers, device, history_items):
    model.eval()
    H = []
    for i in range(len(layers)):
        H.append(torch.zeros((1, layers[i]), dtype=torch.float32, requires_grad=False, device=device))
    for iid in history_items:
        O = model.forward(
            torch.tensor([iid], requires_grad=False, device=device),
            H,
            None,
            training=False,
        )
    return O


class BPR_Model(nn.Module):
    def __init__(self, user_num, item_num, factor_num, device="cuda:0"):
        super(BPR_Model, self).__init__()
        """
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.

		"""
        self.embed_user = nn.Embedding(user_num, factor_num).to(device)
        self.embed_item = nn.Embedding(item_num, factor_num).to(device)
        self.item_biases = nn.Embedding(item_num, 1).to(device)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)
        nn.init.constant_(self.item_biases.weight, 0)

    def from_pretrained(self, pretrained_model):
        self.embed_user.weight.data.copy_(pretrained_model.embed_user.weight)
        self.embed_item.weight.data.copy_(pretrained_model.embed_item.weight)
        self.item_biases.weight.data.copy_(pretrained_model.item_biases.weight)

    def forward(self, user, item_i, item_j):
        user = self.embed_user(user)
        item_i = self.embed_item(item_i)
        item_j = self.embed_item(item_j)

        pointwise_emb = user * item_i
        prediction_i = pointwise_emb.sum(dim=-1) + self.item_biases(item_i).view(-1)
        prediction_j = (user * item_j).sum(dim=-1) + self.item_biases(item_j).view(-1)
        return prediction_i, prediction_j, pointwise_emb


class GSASRecModel(torch.nn.Module):
    def __init__(
        self,
        num_items,
        sequence_length=200,
        embedding_dim=128,
        num_heads=1,
        num_blocks=2,
        dropout_rate=0.5,
        reuse_item_embeddings=False,
    ):
        super(GSASRecModel, self).__init__()
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)

        self.num_heads = num_heads

        self.item_embedding = torch.nn.Embedding(
            self.num_items + 2, self.embedding_dim
        )  # items are enumerated from 1;  +1 for padding
        self.position_embedding = torch.nn.Embedding(self.sequence_length, self.embedding_dim)

        self.transformer_blocks = torch.nn.ModuleList(
            [
                TransformerBlock(self.embedding_dim, self.num_heads, self.embedding_dim, dropout_rate)
                for _ in range(num_blocks)
            ]
        )
        self.seq_norm = torch.nn.LayerNorm(self.embedding_dim)
        self.reuse_item_embeddings = reuse_item_embeddings
        if not self.reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(self.num_items + 2, self.embedding_dim)

    def get_output_embeddings(self) -> torch.nn.Embedding:
        if self.reuse_item_embeddings:
            return self.item_embedding
        else:
            return self.output_embedding

    # returns last hidden state and the attention weights
    def forward(self, input):
        seq = self.item_embedding(input.long())
        mask = (input != self.num_items + 1).float().unsqueeze(-1)

        bs = seq.size(0)
        positions = torch.arange(seq.shape[1]).unsqueeze(0).repeat(bs, 1).to(input.device)
        pos_embeddings = self.position_embedding(positions)[: input.size(0)]
        seq = seq + pos_embeddings
        seq = self.embeddings_dropout(seq)
        seq *= mask

        attentions = []
        for i, block in enumerate(self.transformer_blocks):
            seq, attention = block(seq, mask)
            attentions.append(attention)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    def get_predictions(self, input, limit, rated=None):
        with torch.no_grad():
            model_out, _ = self.forward(torch.tensor(input).to(self.item_embedding.weight.device))
            seq_emb = model_out[:, -1, :]
            output_embeddings = self.get_output_embeddings()
            scores = torch.einsum("bd,nd->bn", seq_emb, output_embeddings.weight)
            scores[:, 0] = float("-inf")
            scores[:, self.num_items + 1 :] = float("-inf")
            if rated is not None:
                for i in range(len(input)):
                    for j in rated[i]:
                        scores[i, j] = float("-inf")
            _, result = torch.topk(scores, limit, dim=1)
            return result.cpu().numpy().flatten()
