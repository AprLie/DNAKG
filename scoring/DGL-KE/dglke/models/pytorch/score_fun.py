# -*- coding: utf-8 -*-
#
# score_fun.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
#

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as INIT
import numpy as np

def batched_l2_dist(a, b):
    a_squared = a.norm(dim=-1).pow(2)
    b_squared = b.norm(dim=-1).pow(2)

    squared_res = th.baddbmm(
        b_squared.unsqueeze(-2), a, b.transpose(-2, -1), alpha=-2
    ).add_(a_squared.unsqueeze(-1))
    res = squared_res.clamp_min_(1e-30).sqrt_()
    return res

def batched_l1_dist(a, b):
    res = th.cdist(a, b, p=1)
    return res

class TransEScore(nn.Module):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """
    def __init__(self, gamma, dist_func='l2'):
        super(TransEScore, self).__init__()
        self.gamma = gamma
        if dist_func == 'l1':
            self.neg_dist_func = batched_l1_dist
            self.dist_ord = 1
        else:  # default use l2
            self.neg_dist_func = batched_l2_dist
            self.dist_ord = 2

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=self.dist_ord, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb + rel_emb).unsqueeze(2) - tail_emb.unsqueeze(0).unsqueeze(0)

        return self.gamma - th.norm(score, p=self.dist_ord, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, chunk_size, hidden_dim)
                return gamma - self.neg_dist_func(tails, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                return gamma - self.neg_dist_func(heads, tails)
            return fn

class TransRScore(nn.Module):
    """TransR score function
    Paper link: https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9571/9523
    """
    def __init__(self, gamma, projection_emb, relation_dim, entity_dim):
        super(TransRScore, self).__init__()
        self.gamma = gamma
        self.projection_emb = projection_emb
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.data['head_emb']
        tail = edges.data['tail_emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        pass

    def prepare(self, g, gpu_id, trace=False):
        head_ids, tail_ids = g.all_edges(order='eid')
        projection = self.projection_emb(g.edata['id'], gpu_id, trace)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        g.edata['head_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][head_ids], projection)
        g.edata['tail_emb'] = th.einsum('ab,abc->ac', g.ndata['emb'][tail_ids], projection)

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(num_chunks, -1, 1, self.entity_dim)
                tail = th.matmul(tail, projection)
                tail = tail.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                head = head.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                head = th.matmul(head, projection)
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(num_chunks, -1, self.entity_dim, self.relation_dim)
                head = head.reshape(num_chunks, -1, 1, self.entity_dim)
                head = th.matmul(head, projection)
                head = head.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                tail = tail.reshape(num_chunks, 1, -1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                tail = th.matmul(tail, projection)
                return head, tail
            return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def reset_parameters(self):
        self.projection_emb.init(1.0)

    def update(self, gpu_id=-1):
        self.projection_emb.update(gpu_id)

    def save(self, path, name):
        self.projection_emb.save(path, name + 'projection')

    def load(self, path, name):
        self.projection_emb.load(path, name + 'projection')

    def prepare_local_emb(self, projection_emb):
        self.global_projection_emb = self.projection_emb
        self.projection_emb = projection_emb

    def prepare_cross_rels(self, cross_rels):
        self.projection_emb.setup_cross_rels(cross_rels, self.global_projection_emb)

    def writeback_local_emb(self, idx):
        self.global_projection_emb.emb[idx] = self.projection_emb.emb.cpu()[idx]

    def load_local_emb(self, projection_emb):
        device = projection_emb.emb.device
        projection_emb.emb = self.projection_emb.emb.to(device)
        self.projection_emb = projection_emb

    def share_memory(self):
        self.projection_emb.share_memory()

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                heads = heads - relations
                heads = heads.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - th.norm(score, p=1, dim=-1)
            return fn

class DistMultScore(nn.Module):
    """DistMult score function
    Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1)
        rel_emb = rel_emb.unsqueeze(0)
        score = (head_emb * rel_emb).unsqueeze(2) * tail_emb.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)

            return fn


class ComplExScore(nn.Module):
    """ComplEx score function
    Paper link: https://arxiv.org/abs/1606.06357
    """

    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, -1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        real_head, img_head = th.chunk(head_emb, 2, dim=-1)
        real_tail, img_tail = th.chunk(tail_emb, 2, dim=-1)
        real_rel, img_rel = th.chunk(rel_emb, 2, dim=-1)

        score = (real_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0) \
                + (img_head.unsqueeze(1) * real_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
                + (real_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * img_tail.unsqueeze(0).unsqueeze(0) \
                - (img_head.unsqueeze(1) * img_rel.unsqueeze(0)).unsqueeze(2) * real_tail.unsqueeze(0).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                return th.bmm(tmp, heads)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]
                rel_real = relations[..., :hidden_dim // 2]
                rel_imag = relations[..., hidden_dim // 2:]
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                return th.bmm(tmp, tails)
            return fn

class RESCALScore(nn.Module):
    """RESCAL score function
    Paper link: http://www.icml-2011.org/papers/438_icmlpaper.pdf
    """
    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        score = head * th.matmul(rel, tail).squeeze(-1)
        # TODO: check if use self.gamma
        return {'score': th.sum(score, dim=-1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_emb = head_emb.unsqueeze(1).unsqueeze(1)
        rel_emb = rel_emb.view(-1, self.relation_dim, self.entity_dim)
        score = head_emb * th.einsum('abc,dc->adb', rel_emb, tail_emb).unsqueeze(0)

        return th.sum(score, dim=-1)

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                tails = tails.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, tails).squeeze(-1)
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                heads = heads.unsqueeze(-1)
                relations = relations.view(-1, self.relation_dim, self.entity_dim)
                tmp = th.matmul(relations, heads).squeeze(-1)
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return th.bmm(tmp, tails)
            return fn

class RotatEScore(nn.Module):
    """RotatE score function
    Paper link: https://arxiv.org/abs/1902.10197
    """
    def __init__(self, gamma, emb_init):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        re_head, im_head = th.chunk(edges.src['emb'], 2, dim=-1)
        re_tail, im_tail = th.chunk(edges.dst['emb'], 2, dim=-1)

        phase_rel = edges.data['emb'] / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head * re_rel - im_head * im_rel
        im_score = re_head * im_rel + im_head * re_rel
        re_score = re_score - re_tail
        im_score = im_score - im_tail
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return {'score': self.gamma - score.sum(-1)}

    def infer(self, head_emb, rel_emb, tail_emb):
        re_head, im_head = th.chunk(head_emb, 2, dim=-1)
        re_tail, im_tail = th.chunk(tail_emb, 2, dim=-1)

        phase_rel = rel_emb / (self.emb_init / np.pi)
        re_rel, im_rel = th.cos(phase_rel), th.sin(phase_rel)
        re_score = re_head.unsqueeze(1) * re_rel.unsqueeze(0) - im_head.unsqueeze(1) * im_rel.unsqueeze(0)
        im_score = re_head.unsqueeze(1) * im_rel.unsqueeze(0) + im_head.unsqueeze(1) * re_rel.unsqueeze(0)

        re_score = re_score.unsqueeze(2) - re_tail.unsqueeze(0).unsqueeze(0)
        im_score = im_score.unsqueeze(2) - im_tail.unsqueeze(0).unsqueeze(0)
        score = th.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        return self.gamma - score.sum(-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        gamma = self.gamma
        emb_init = self.emb_init
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = tails[..., :hidden_dim // 2]
                emb_imag = tails[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real + emb_imag * rel_imag
                imag = -emb_real * rel_imag + emb_imag * rel_real
                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - heads
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)
                return gamma - score.sum(-1)

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real = heads[..., :hidden_dim // 2]
                emb_imag = heads[..., hidden_dim // 2:]

                phase_rel = relations / (emb_init / np.pi)
                rel_real, rel_imag = th.cos(phase_rel), th.sin(phase_rel)
                real = emb_real * rel_real - emb_imag * rel_imag
                imag = emb_real * rel_imag + emb_imag * rel_real

                emb_complex = th.cat((real, imag), dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, 1, hidden_dim)
                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                score = tmp - tails
                score = th.stack([score[..., :hidden_dim // 2],
                                  score[..., hidden_dim // 2:]], dim=-1).norm(dim=-1)

                return gamma - score.sum(-1)

            return fn

class SimplEScore(nn.Module):
    """SimplE score function
    Paper link: http://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf
    """
    def __init__(self):
        super(SimplEScore, self).__init__()

    def edge_func(self, edges):
        head_i, head_j = th.chunk(edges.src['emb'], 2, dim=-1)
        tail_i, tail_j = th.chunk(edges.dst['emb'], 2, dim=-1)
        rel, rel_inv = th.chunk(edges.data['emb'], 2, dim=-1)
        forward_score = head_i * rel * tail_j
        backward_score = tail_i * rel_inv * head_j
        # clamp as official implementation does to avoid NaN output
        # might because of gradient explode
        score = th.clamp(1 / 2 * (forward_score + backward_score).sum(-1), -20, 20)
        return {'score': score}

    def infer(self, head_emb, rel_emb, tail_emb):
        head_i, head_j = th.chunk(head_emb.unsqueeze(1), 2, dim=-1)
        tail_i, tail_j = th.chunk(tail_emb.unsqueeze(0).unsqueeze(0), 2, dim=-1)
        rel, rel_inv = th.chunk(rel_emb.unsqueeze(0), 2, dim=-1)
        forward_tmp = (head_i * rel).unsqueeze(2) * tail_j
        backward_tmp = (head_j * rel_inv).unsqueeze(2) * tail_i
        score = (forward_tmp + backward_tmp) * 1 / 2
        return th.sum(score, dim=-1)

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn


    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = tails.shape[1]
                tail_i = tails[..., :hidden_dim // 2]
                tail_j = tails[..., hidden_dim // 2:]
                rel = relations[..., : hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (rel * tail_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * tail_i).reshape(num_chunks, chunk_size, hidden_dim//2)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                head_i = heads[..., :hidden_dim // 2, :]
                head_j = heads[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, head_i) + th.bmm(backward_tmp, head_j))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                head_i = heads[..., :hidden_dim // 2]
                head_j = heads[..., hidden_dim // 2:]
                rel = relations[..., :hidden_dim // 2]
                rel_inv = relations[..., hidden_dim // 2:]
                forward_tmp = (head_i * rel).reshape(num_chunks, chunk_size, hidden_dim//2)
                backward_tmp = (rel_inv * head_j).reshape(num_chunks, chunk_size, hidden_dim//2)
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = th.transpose(tails, 1, 2)
                tail_i = tails[..., :hidden_dim // 2, :]
                tail_j = tails[..., hidden_dim // 2:, :]
                tmp = 1 / 2 * (th.bmm(forward_tmp, tail_j) + th.bmm(backward_tmp, tail_i))
                score = th.clamp(tmp, -20, 20)
                return score
            return fn


class PairRE(nn.Module):
    def __init__(self, gamma):
        self.gamma = gamma
        super(PairRE, self).__init__()

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def prepare(self, g, gpu_id, trace=False):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def edge_func(self, edges):

        head = edges.src['emb'].unsqueeze(1)
        relation = edges.data['emb'].unsqueeze(1)
        tail = edges.dst['emb'].unsqueeze(1)

        # print('head.shape', head.shape)
        # print('relation.shape', relation.shape)
        # print('tail.shape', tail.shape)

        re_head, re_tail = th.chunk(relation, 2, dim=2)

        head = functional.normalize(head, 2, -1)
        tail = functional.normalize(tail, 2, -1)

        # print('re_head.shape', re_head.shape)
        # print('re_tail.shape', re_tail.shape)
        # print('head.shape', head.shape)
        # print('tail.shape', tail.shape)

        score = head * re_head - tail * re_tail
        score = self.gamma - th.norm(score, p=1, dim=2)

        return {'score': score}
        # return score

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):

                hidden_dim = heads.shape[1]

                heads = heads.unsqueeze(1)
                relations = relations.unsqueeze(1)
                tails = tails.unsqueeze(1)

                re_head, re_tail = th.chunk(relations, 2, dim=2)

                head = functional.normalize(heads, 2, -1)
                tail = functional.normalize(tails, 2, -1)
                # print('---head neg---')
                # print('re_head.shape', re_head.shape)
                # print('re_tail.shape', re_tail.shape)
                # print('head.shape', head.shape)
                # print('tail.shape', tail.shape)

                # score = head * re_head - tail * re_tail
                # print('head * re_head .shape:', (head * re_head).shape)
                # print('tail * re_tail.shape:', (tail * re_tail).shape)
                tt = th.transpose(head * re_head, 1, 0)
                mm = tail * re_tail
                # print('tt.shape', tt.shape)
                score = tt - mm
                # print('nn.shape', nn.shape)

                score = self.gamma - th.norm(score, p=1, dim=2)
                # print('score.shape', score.shape)

                return score

            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):

                hidden_dim = heads.shape[1]

                heads = heads.unsqueeze(1)
                relations = relations.unsqueeze(1)
                tails = tails.unsqueeze(1)

                re_head, re_tail = th.chunk(relations, 2, dim=2)

                head = functional.normalize(heads, 2, -1)
                tail = functional.normalize(tails, 2, -1)
                tail = th.reshape(tail, (head.shape[0], -1, hidden_dim))
                # print('--tail neg---')
                # print('num_chunks:', num_chunks)
                # print('chunk_size:', chunk_size)
                # print('neg_sample_size:', neg_sample_size)
                # print('re_head.shape', re_head.shape)
                # print('re_tail.shape', re_tail.shape)
                # print('head.shape', head.shape)
                # print('tail.shape', tail.shape)

                # if neg_sample_size != 1001:
                #     tt = th.transpose(head * re_head, 1, 0)
                #     mm = tail * re_tail
                #     score = tt - mm
                #     # print('score0.shape', score.shape)
                # else:
                score = head * re_head - tail * re_tail
                    # print('score0.shape', score.shape)
                #
                score = self.gamma - th.norm(score, p=1, dim=2)
                # print('score.shape', score.shape)

                return score

            return fn


class AutoSFScore(nn.Module):

    def __init__(self):
        super(AutoSFScore, self).__init__()

    def update(self, gpu_id=-1):
        pass

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def prepare(self, g, gpu_id, trace=False):
        pass

    def edge_func(self, edges):
        # print('edges.src:', edges.src['emb'].shape)
        # print('edges.dst:', edges.dst['emb'].shape)
        # print('edges.data:', edges.data['emb'].shape)

        head = edges.src['emb']
        relation = edges.data['emb']
        tail = edges.dst['emb']

        # if mode == 'head-batch':
        #     rs = torch.chunk(relation, 4, dim=-1)
        #     ts = torch.chunk(tail, 4, dim=-1)
        #     rt0 = rs[0] * ts[0]
        #     rt1 = rs[1] * ts[1] + rs[2] * ts[3]
        #     rt2 = rs[0] * ts[2] + rs[2] * ts[3]
        #     rt3 = -rs[1]* ts[1] + rs[3] * ts[2]
        #     rts = torch.cat([rt0,rt1,rt2,rt3], dim=-1)
        #     score = torch.sum(head * rts, dim=-1)

        # else:
        hs = th.chunk(head, 4, dim=-1)
        rs = th.chunk(relation, 4, dim=-1)
        hr0 = hs[0] * rs[0]
        hr1 = hs[1] * rs[1] - hs[3] * rs[1]
        hr2 = hs[2] * rs[0] + hs[3] * rs[3]
        hr3 = hs[1] * rs[2] + hs[2] * rs[2]
        hrs = th.cat([hr0, hr1, hr2, hr3], dim=-1)

        score = th.sum(hrs * tail, dim=-1)

        # print('score.shape:', score.shape)

        return {'score': score}

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail

        return fn

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                # print('heads.shape:', heads.shape)
                # print('relations.shape:', relations.shape)
                # print('tails.shape:', tails.shape)
                # print('num_chunks:', num_chunks)
                # print('chunk_size:', chunk_size)
                # print('neg_sample_size:', neg_sample_size)

                hidden_dim = heads.shape[1]
                rs = th.chunk(relations, 4, dim=-1)
                ts = th.chunk(tails, 4, dim=-1)

                rt0 = rs[0] * ts[0]
                rt1 = rs[1] * ts[1] + rs[2] * ts[3]
                rt2 = rs[0] * ts[2] + rs[2] * ts[3]
                rt3 = -rs[1] * ts[1] + rs[3] * ts[2]

                rts = th.cat([rt0, rt1, rt2, rt3], dim=-1)

                rts = rts.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = th.transpose(heads, 1, 2)
                score = th.bmm(rts, heads)

                # rts = rts.unsqueeze(1)
                # # print('rts.shape:', rts.shape)

                # heads = heads.reshape(-1, num_chunks * neg_sample_size, hidden_dim)
                # # print('heads.shape:', heads.shape)

                # score = th.sum(heads * rts, dim=-1)
                # # print('score.shape:', score.shape)

                return score

            return fn

        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                # heads [batch_size, 1, dim]
                # relations [batch_size, 1, dim]
                ##[batch_size, negative_sample_size, dim]
                # print('---heads.shape-----:', heads.shape)
                # print('relations.shape:', relations.shape)
                # print('tails.shape:', tails.shape)
                # print('num_chunks:', num_chunks)
                # print('chunk_size:', chunk_size)
                # print('neg_sample_size:', neg_sample_size)
                hidden_dim = heads.shape[1]

                hs = th.chunk(heads, 4, dim=-1)
                rs = th.chunk(relations, 4, dim=-1)
                hr0 = hs[0] * rs[0]
                hr1 = hs[1] * rs[1] - hs[3] * rs[1]
                hr2 = hs[2] * rs[0] + hs[3] * rs[3]
                hr3 = hs[1] * rs[2] + hs[2] * rs[2]
                hrs = th.cat([hr0, hr1, hr2, hr3], dim=-1)
                # print('hrs.shape:', hrs.shape)
                # print('tails.shape:', tails.shape)

                hrs2 = hrs.reshape(num_chunks, chunk_size, hidden_dim)
                tails2 = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails2 = th.transpose(tails2, 1, 2)
                # print('tails2.shape:',tails2.shape)
                # print('hrs2.shape:',hrs2.shape)
                # t = time.time()
                score = th.bmm(hrs2, tails2)
                # print('th.bmm(hrs2, tails2).shape:', th.bmm(hrs2, tails2).shape)
                # print('score2.shape:', score2.shape)
                # print('--neg spend time0:', time.time() - t)
                # # print('score2:', score2)

                # hrs = hrs.unsqueeze(1)
                # print('hrs.shape:', hrs.shape)
                # # tails = tails.reshape(-1, num_chunks * neg_sample_size, hidden_dim)
                # if neg_sample_size != 1001:
                #     tails = tails.reshape(-1, num_chunks * neg_sample_size, hidden_dim)
                # else:
                #     tails = tails.reshape(-1, neg_sample_size, hidden_dim)
                # print('tails.shape:', tails.shape)
                # # print('hrs * tails.shape:', (hrs * tails).shape)

                # t = time.time()
                # score = th.sum(hrs * tails, dim=-1)
                # print('score.shape:', score.shape)
                # # print('score:', score)
                # print('--neg spend time1:', time.time() - t)

                # tails = th.transpose(tails, 1, 2)
                # hrs = th.transpose(hrs, 0, 1)
                # t = time.time()
                # print('hrs3.shape:',hrs.shape)
                # print('tails3.shape:',tails.shape)
                # a = th.bmm(hrs, tails)
                # print('a.shape:', a.shape)
                # print('--neg spend time2:', time.time() - t)

                return score

            return fn


