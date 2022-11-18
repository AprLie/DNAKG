# -*- coding: utf-8 -*-
#
# general_models.py
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
"""
Graph Embedding Model
1. TransE
2. TransR
3. RESCAL
4. DistMult
5. ComplEx
6. RotatE
7. SimplE
"""
import os
import numpy as np
import math
import dgl.backend as F
import torch
import torch.nn as nn
from tqdm import tqdm, trange
backend = os.environ.get('DGLBACKEND', 'pytorch')
from .pytorch.tensor_models import logsigmoid
from .pytorch.tensor_models import abs
from .pytorch.tensor_models import masked_select
from .pytorch.tensor_models import get_device, get_dev
from .pytorch.tensor_models import norm
from .pytorch.tensor_models import get_scalar
from .pytorch.tensor_models import reshape
from .pytorch.tensor_models import cuda
from .pytorch.tensor_models import ExternalEmbedding, RelationExternalEmbedding
from .pytorch.tensor_models import InferEmbedding
from .pytorch.score_fun import *
from .pytorch.loss import LossGenerator

DEFAULT_INFER_BATCHSIZE = 2048

EMB_INIT_EPS = 2.0


class InferModel(object):
    def __init__(self, device, model_name, hidden_dim,
                 double_entity_emb=False, double_relation_emb=False,
                 gamma=0., batch_size=DEFAULT_INFER_BATCHSIZE):
        super(InferModel, self).__init__()

        self.device = device
        self.model_name = model_name
        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        self.entity_emb = InferEmbedding(device)
        self.relation_emb = InferEmbedding(device)
        self.batch_size = batch_size

        if model_name == 'TransE' or model_name == 'TransE_l2':
            self.score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            self.score_func = TransEScore(gamma, 'l1')
        elif model_name == 'TransR':
            assert False, 'Do not support inference of TransR model now.'
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore(relation_dim, entity_dim)
        elif model_name == 'RotatE':
            emb_init = (gamma + EMB_INIT_EPS) / hidden_dim
            self.score_func = RotatEScore(gamma, emb_init)
        elif model_name == 'SimplE':
            self.score_func = SimplEScore()


    def load_emb(self, path, dataset):
        """Load the model.
        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        self.entity_emb.load(path, dataset + '_' + self.model_name + '_entity')
        self.relation_emb.load(path, dataset + '_' + self.model_name + '_relation')
        self.score_func.load(path, dataset + '_' + self.model_name)

    def score(self, head, rel, tail, triplet_wise=False):
        head_emb = self.entity_emb(head)
        rel_emb = self.relation_emb(rel)
        tail_emb = self.entity_emb(tail)

        num_head = F.shape(head)[0]
        num_rel = F.shape(rel)[0]
        num_tail = F.shape(tail)[0]

        batch_size = self.batch_size
        score = []
        if triplet_wise:
            class FakeEdge(object):
                def __init__(self, head_emb, rel_emb, tail_emb):
                    self._hobj = {}
                    self._robj = {}
                    self._tobj = {}
                    self._hobj['emb'] = head_emb
                    self._robj['emb'] = rel_emb
                    self._tobj['emb'] = tail_emb

                @property
                def src(self):
                    return self._hobj

                @property
                def dst(self):
                    return self._tobj

                @property
                def data(self):
                    return self._robj

            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size: (i + 1) * batch_size \
                    if (i + 1) * batch_size < num_head \
                    else num_head]
                sr_emb = rel_emb[i * batch_size: (i + 1) * batch_size \
                    if (i + 1) * batch_size < num_head \
                    else num_head]
                st_emb = tail_emb[i * batch_size: (i + 1) * batch_size \
                    if (i + 1) * batch_size < num_head \
                    else num_head]
                edata = FakeEdge(sh_emb, sr_emb, st_emb)
                score.append(F.copy_to(self.score_func.edge_func(edata)['score'], F.cpu()))
            score = F.cat(score, dim=0)
            return score
        else:
            for i in range((num_head + batch_size - 1) // batch_size):
                sh_emb = head_emb[i * batch_size: (i + 1) * batch_size \
                    if (i + 1) * batch_size < num_head \
                    else num_head]
                s_score = []
                for j in range((num_tail + batch_size - 1) // batch_size):
                    st_emb = tail_emb[j * batch_size: (j + 1) * batch_size \
                        if (j + 1) * batch_size < num_tail \
                        else num_tail]

                    s_score.append(F.copy_to(self.score_func.infer(sh_emb, rel_emb, st_emb), F.cpu()))
                score.append(F.cat(s_score, dim=2))
            score = F.cat(score, dim=0)
            return F.reshape(score, (num_head * num_rel * num_tail,))

    @property
    def num_entity(self):
        return self.entity_emb.emb.shape[0]

    @property
    def num_rel(self):
        return self.relation_emb.emb.shape[0]


class MLP(torch.nn.Module):
    def __init__(self, input_entity_dim, entity_dim, input_relation_dim, relation_dim, args):
        super(MLP, self).__init__()
        self.args = args
        self.transform_e_net = torch.nn.Linear(input_entity_dim, entity_dim)
        self.transform_r_net = torch.nn.Linear(input_relation_dim, relation_dim)
        if args.use_2_layer_mlp:
            self.transform_e_net_v2 = torch.nn.Linear(entity_dim, entity_dim)
            self.transform_r_net_v2 = torch.nn.Linear(relation_dim, relation_dim)
            self.dropout = nn.Dropout(0.1)

        self.reset_parameters()

    def embed_entity(self, embeddings):
        feature = self.transform_e_net(embeddings)
        if self.args.use_2_layer_mlp:
            x = torch.nn.functional.leaky_relu(feature)
            x = self.dropout(x)
            score = self.transform_e_net_v2(feature)
        else:
            score = feature
        return score
        # return self.transform_e_net(embeddings)

    def embed_relation(self, embeddings):
        feature = self.transform_r_net(embeddings)
        if self.args.use_2_layer_mlp:
            x = torch.nn.functional.leaky_relu(feature)
            x = self.dropout(x)
            score = self.transform_r_net_v2(feature)
        else:
            score = feature

        return score
        # return self.transform_r_net(embeddings)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.transform_r_net.weight)
        nn.init.xavier_uniform_(self.transform_e_net.weight)
        if self.args.use_2_layer_mlp:
            # print('reset_parameters, use_2_layer_mlp')
            nn.init.xavier_uniform_(self.transform_r_net_v2.weight)
            nn.init.xavier_uniform_(self.transform_e_net_v2.weight)


class KEModel(object):
    """ DGL Knowledge Embedding Model.
    Parameters
    ----------
    args:
        Global configs.
    model_name : str
        Which KG model to use, including 'TransE_l1', 'TransE_l2', 'TransR',
        'RESCAL', 'DistMult', 'ComplEx', 'RotatE', 'SimplE'
    n_entities : int
        Num of entities.
    n_relations : int
        Num of relations.
    hidden_dim : int
        Dimension size of embedding.
    gamma : float
        Gamma for score function.
    double_entity_emb : bool
        If True, entity embedding size will be 2 * hidden_dim.
        Default: False
    double_relation_emb : bool
        If True, relation embedding size will be 2 * hidden_dim.
        Default: False
    """

    def __init__(self, args, model_name, n_entities, n_relations, hidden_dim, gamma,
                 double_entity_emb=False, double_relation_emb=False, ent_feat_dim=-1, rel_feat_dim=-1):
        super(KEModel, self).__init__()
        print('KEModel param: hidden_dim=', hidden_dim, 'ent_feat_dim=', ent_feat_dim, 'rel_feat_dim=', rel_feat_dim)
        self.args = args
        self.has_edge_importance = args.has_edge_importance
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / hidden_dim

        entity_dim = 2 * hidden_dim if double_entity_emb else hidden_dim
        relation_dim = 2 * hidden_dim if double_relation_emb else hidden_dim

        self.encoder_model_name = args.encoder_model_name
        if args.LRE:
            self.LRE = args.LRE
            self.LRE_rank = args.LRE_rank
            self.feat_hidden_dim = args.feat_hidden_dim
        device = get_device(args)

        if args.use_relation_weight:
            self.relation_weight = np.load(args.data_path + '/wikikg90m_kddcup2021/processed/relation_weight.npy')
            # print('--use_relation_weight--:', self.relation_weight.shape)
            # print('relation_weight:', self.relation_weight.tolist())

        self.loss_gen = LossGenerator(args, args.loss_genre, args.neg_adversarial_sampling,
                                      args.adversarial_temperature, args.pairwise)

        if self.encoder_model_name in ['shallow', 'concat', 'shallow_net']:
            if self.LRE:
                self.entity_emb_base = ExternalEmbedding(args, self.LRE_rank, entity_dim,
                                                         F.cpu() if args.mix_cpu_gpu else device)
                self.entity_emb_index = ExternalEmbedding(args, n_entities, self.LRE_rank,
                                                          F.cpu() if args.mix_cpu_gpu else device)
                self.entity_emb = None
            else:
                self.entity_emb = ExternalEmbedding(args, n_entities, entity_dim, F.cpu() if args.mix_cpu_gpu else device)

        if self.encoder_model_name in ['roberta', 'concat']:
            assert ent_feat_dim != -1 and rel_feat_dim != -1
            self.entity_feat = ExternalEmbedding(args, n_entities, ent_feat_dim,
                                                 F.cpu() if args.mix_cpu_gpu else device, is_feat=True)
        # For RESCAL, relation_emb = relation_dim * entity_dim
        if model_name == 'RESCAL':
            rel_dim = relation_dim * entity_dim
        else:
            rel_dim = relation_dim
        # if model_name == 'RotatE':
        #     rel_dim = rel_dim // 2
            # print('---RotatE, rel_dim: ', rel_dim)

        if model_name == 'PairRE':
            rel_dim = rel_dim * 2
            # print('---PairRE, rel_dim: ', rel_dim)

        self.use_mlp = self.encoder_model_name in ['concat', 'roberta', 'shallow_net']
        if model_name == "OTE":
            self.use_scale = True if args.scale_type > 0 else False
        else: self.use_scale = False
        if self.encoder_model_name == 'concat':
            if model_name == "OTE":
                self.transform_net = MLP(
                    entity_dim + ent_feat_dim, entity_dim,
                    (args.ote_size + int(self.use_scale)
                     ) * relation_dim + rel_feat_dim,
                    (args.ote_size + int(self.use_scale)) * relation_dim, args)
            else:
                self.transform_net = MLP(entity_dim + ent_feat_dim, entity_dim, relation_dim + rel_feat_dim, relation_dim, args)
            # self.transform_e_net = torch.nn.Linear(entity_dim, entity_dim)
            # self.transform_r_net = torch.nn.Linear(relation_dim, relation_dim)
        elif self.encoder_model_name == 'roberta':
            self.transform_net = MLP(ent_feat_dim, entity_dim, rel_feat_dim, relation_dim, args)

        elif self.encoder_model_name == 'shallow_net':
            self.transform_net = MLP(entity_dim, entity_dim, rel_dim, rel_dim, args)

        self.rel_dim = rel_dim
        self.entity_dim = entity_dim
        self.strict_rel_part = args.strict_rel_part
        self.soft_rel_part = args.soft_rel_part
        # print('strict_rel_part: ', self.strict_rel_part, "soft_rel_part: ", self.soft_rel_part)

        assert not self.strict_rel_part and not self.soft_rel_part

        if not self.strict_rel_part and not self.soft_rel_part:
            if self.encoder_model_name in ['shallow', 'concat', 'shallow_net']:
                # print('--use relation_emb--')
                if model_name == "OTE":
                    self.relation_emb = RelationExternalEmbedding(
                        args, n_relations, rel_dim,
                        F.cpu() if args.mix_cpu_gpu else device)
                else:
                    self.relation_emb = ExternalEmbedding(args, n_relations, rel_dim,
                                                      F.cpu() if args.mix_cpu_gpu else device)

            if self.encoder_model_name in ['roberta', 'concat']:
                # print('--use relation_feat--')
                self.relation_feat = ExternalEmbedding(args, n_relations, rel_feat_dim,
                                                       F.cpu() if args.mix_cpu_gpu else device, is_feat=True)
        else:
            # print('---global_relation_emb---')
            self.global_relation_emb = ExternalEmbedding(args, n_relations, rel_dim, F.cpu())

        if model_name == 'TransE' or model_name == 'TransE_l2':
            self.score_func = TransEScore(gamma, 'l2')
        elif model_name == 'TransE_l1':
            self.score_func = TransEScore(gamma, 'l1')
        elif model_name == 'TransR':
            projection_emb = ExternalEmbedding(args,
                                               n_relations,
                                               entity_dim * relation_dim,
                                               F.cpu() if args.mix_cpu_gpu else device)

            self.score_func = TransRScore(gamma, projection_emb, relation_dim, entity_dim)
        elif model_name == 'DistMult':
            self.score_func = DistMultScore()
        elif model_name == 'ComplEx':
            self.score_func = ComplExScore()
        elif model_name == 'RESCAL':
            self.score_func = RESCALScore(relation_dim, entity_dim)
        elif model_name == 'RotatE':
            self.score_func = RotatEScore(gamma, self.emb_init)
        elif model_name == 'SimplE':
            self.score_func = SimplEScore()

        elif model_name == 'AutoSF':
            # print('---use autoSF')
            self.score_func = AutoSFScore()

        elif model_name == 'PairRE':
            # print('---use PairRE')
            self.score_func = PairRE(gamma)
        elif model_name == "OTE":
            self.score_func = OTEScore(args.gamma, args.ote_size,
                                       args.scale_type)
        self.model_name = model_name

        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)

        self.head_neg_prepare = self.score_func.create_neg_prepare(True)
        self.tail_neg_prepare = self.score_func.create_neg_prepare(False)
        # if
        self.reset_parameters()

    def share_memory(self):
        """Use torch.tensor.share_memory_() to allow cross process embeddings access.
        """
        if self.encoder_model_name in ['concat', 'shallow']:
            if self.LRE:
                self.entity_emb_base.share_memory()
                self.entity_emb_index.share_memory()
            else:
                self.entity_emb.share_memory()
        if self.encoder_model_name in ['concat', 'roberta']:
            self.entity_feat.share_memory()
        if self.strict_rel_part or self.soft_rel_part:
            self.global_relation_emb.share_memory()
        else:
            if self.encoder_model_name in ['concat', 'shallow']:
                self.relation_emb.share_memory()
            if self.encoder_model_name in ['concat', 'roberta']:
                self.relation_feat.share_memory()

        if self.model_name == 'TransR':
            self.score_func.share_memory()

        if self.use_mlp:
            self.transform_net.share_memory()
    
    def save_useful_emb(self, path, dataset, dim):
        """Save real entity embedding and relation embedding.

        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """ 
        if self.encoder_model_name == 'concat':
            import gc
            relation_emb = self.transform_net.embed_relation(torch.cat([torch.from_numpy(self.relation_feat.emb), self.relation_emb.emb], -1)).cpu().detach().numpy()
            relation_save_file_name = os.path.join(path, dataset + '_' + self.model_name + '_relation_emb.npy')
            np.save(relation_save_file_name, relation_emb)
            del relation_emb
            gc.collect()
            num_node = 91230610
            num_file = 100
            node_per_file = int(num_node / num_file)
            entity_emb = np.zeros((num_node, dim), dtype=np.float32)
            
            for i in trange(num_file + 1):
                start = i * node_per_file
                end = min((i + 1) * node_per_file, num_node)
                if end == num_node:
                    start = end - node_per_file
                node_id = torch.arange(start, end)
                # node_id = torch.arange(0, 91229)
                entity_feat = self.transform_net.embed_entity(torch.cat([self.entity_feat(node_id, -1, False), self.entity_emb(node_id, -1, False)], -1)).cpu().detach().numpy()
                entity_emb[start:end, :] = entity_feat.astype(np.float32)
                del entity_feat
                gc.collect()
            entity_save_file_name = os.path.join(path, dataset + '_' + self.model_name + '_entity_emb.npy')
            np.save(entity_save_file_name, entity_emb)
        else:
            print("not consider!")
            return 

    def save_emb(self, path, dataset):
        """Save the model.
        Parameters
        ----------
        path : str
            Directory to save the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        if self.encoder_model_name in ['shallow', 'concat', 'shallow_net'] and self.args.save_entity_emb:
            if self.LRE:
                self.entity_emb_base.save(path, dataset+'_'+self.model_name+'_entity_emb_base')
                self.entity_emb_index.save(path, dataset+'_'+self.model_name+'_entity_emb_index')
            else:
                self.entity_emb.save(path, dataset + '_' + self.model_name + '_entity')

        if self.encoder_model_name in ['roberta', 'concat', 'shallow_net'] and self.args.save_mlp:
            torch.save({'transform_state_dict': self.transform_net.state_dict()},
                       os.path.join(path, dataset + "_" + self.model_name + "_mlp"))

        if self.strict_rel_part or self.soft_rel_part:
            if self.args.save_rel_emb:
                self.global_relation_emb.save(path, dataset + '_' + self.model_name + '_relation')
        else:
            if self.encoder_model_name in ['shallow', 'concat', 'shallow_net'] and self.args.save_rel_emb:
                self.relation_emb.save(path, dataset + '_' + self.model_name + '_relation')

        self.score_func.save(path, dataset + '_' + self.model_name)

    def load_emb(self, path, dataset):
        """Load the model.
        Parameters
        ----------
        path : str
            Directory to load the model.
        dataset : str
            Dataset name as prefix to the saved embeddings.
        """
        

        
        if self.encoder_model_name in ['roberta', 'concat', 'shallow_net']:
            print('---load_emb transform_net: ', path)
            if os.path.exists(os.path.join(path, dataset + "_" + self.model_name + "_mlp")):
                mlp_net = torch.load(os.path.join(path, dataset + "_" + self.model_name + "_mlp"))
                self.transform_net.load_state_dict(mlp_net['transform_state_dict'])
            else:
                print('*******transform_net is not exists, reset_parameters ************')
                self.transform_net.reset_parameters()

        if self.args.dtype == 16:
            print('---load_emb float16 path: ', path)
            self.relation_emb.load(path, dataset + '_' + self.model_name + '_relation_float16')
            self.entity_emb.load(path, dataset + '_' + self.model_name + '_entity_float16')

        else:
            print('---load_emb float32 path: ', path)
            if self.LRE:
                self.entity_emb_base.load(path, dataset+'_'+self.model_name+'_entity_emb_base')
                self.entity_emb_index.load(path, dataset+'_'+self.model_name+'_entity_emb_index')
            else: 
                self.entity_emb.load(path, dataset + '_' + self.model_name + '_entity')
            self.relation_emb.load(path, dataset + '_' + self.model_name + '_relation')
        self.score_func.load(path, dataset + '_' + self.model_name)

    def reset_parameters(self):
        """Re-initialize the model.
        """
        if self.encoder_model_name in ['shallow', 'concat', 'shallow_net'] and not self.args.is_eval:
            print('---entity_emb.init')
            if self.LRE:
                self.entity_emb_base.init(1.0)
                self.entity_emb_index.init(self.emb_init)
            else:
                self.entity_emb.init(self.emb_init)

        self.score_func.reset_parameters()

        if (not self.strict_rel_part) and (not self.soft_rel_part):
            if self.encoder_model_name in ['shallow', 'concat', 'shallow_net'] and not self.args.is_eval:
                print('---relation_emb.init')
                self.relation_emb.init(self.emb_init)

        elif not self.args.is_eval:
            print('---global_relation_emb.init')
            self.global_relation_emb.init(self.emb_init)

        if self.use_mlp and not self.args.is_eval:
            print('---use_mlp.init')
            self.transform_net.reset_parameters()

    def predict_score(self, g):
        """Predict the positive score.
        Parameters
        ----------
        g : DGLGraph
            Graph holding positive edges.
        Returns
        -------
        tensor
            The positive score
        """
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g, to_device=None, gpu_id=-1, trace=False, neg_deg_sample=False):
        """Calculate the negative score.
        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        to_device : func
            Function to move data into device.
        gpu_id : int
            Which gpu to move data to.
        trace : bool
            If True, trace the computation. This is required in training.
            If False, do not trace the computation.
            Default: False
        neg_deg_sample : bool
            If True, we use the head and tail nodes of the positive edges to
            construct negative edges.
            Default: False
        Returns
        -------
        tensor
            The negative score
        """
        num_chunks = neg_g.num_chunks
        chunk_size = neg_g.chunk_size
        neg_sample_size = neg_g.neg_sample_size

        mask = F.ones((num_chunks, chunk_size * (neg_sample_size + chunk_size)), dtype=F.float32,
                      ctx=F.context(pos_g.ndata['emb']))

        if neg_g.neg_head:
            neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
            if self.encoder_model_name == 'roberta':
                neg_head = self.transform_net.embed_entity(self.entity_feat(neg_head_ids, gpu_id, False))
            elif self.encoder_model_name == 'shallow_net':
                neg_head = self.transform_net.embed_entity(self.entity_emb(neg_head_ids, gpu_id, trace))
            elif self.encoder_model_name == 'shallow':
                if self.LRE:
                    emb_index = self.entity_emb_index(neg_head_ids, gpu_id, trace)
                    emb_base = self.entity_emb_base(None, gpu_id, trace)
                    neg_head = torch.mm(emb_index, emb_base)
                else:
                    neg_head = self.entity_emb(neg_head_ids, gpu_id, trace)
            elif self.encoder_model_name == 'concat':
                if self.LRE:
                    emb_index = self.entity_emb_index(neg_head_ids, gpu_id, trace)
                    emb_base = self.entity_emb_base(None, gpu_id, trace)
                    neg_head = torch.mm(emb_index, emb_base)
                else:
                    neg_head = self.entity_emb(neg_head_ids, gpu_id, trace)
                neg_head = self.transform_net.embed_entity(torch.cat(
                    [self.entity_feat(neg_head_ids, gpu_id, False), neg_head], -1))

            if neg_head.dtype == torch.float16:
                neg_head = neg_head.type(torch.float32)

            head_ids, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0:
                tail_ids = to_device(tail_ids, gpu_id)

            tail = pos_g.ndata['emb'][tail_ids]
            rel = pos_g.edata['emb']

            # When we train a batch, we could use the head nodes of the positive edges to
            # construct negative edges. We construct a negative edge between a positive head
            # node and every positive tail node.
            # When we construct negative edges like this, we know there is one positive
            # edge for a positive head node among the negative edges. We need to mask
            # them.
            if neg_deg_sample:
                head = pos_g.ndata['emb'][head_ids]
                head = head.reshape(num_chunks, chunk_size, -1)
                neg_head = neg_head.reshape(num_chunks, neg_sample_size, -1)
                neg_head = F.cat([head, neg_head], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:, 0::(neg_sample_size + 1)] = 0
            neg_head = neg_head.reshape(num_chunks * neg_sample_size, -1)

            neg_head, tail = self.head_neg_prepare(pos_g.edata['id'], num_chunks, neg_head, tail, gpu_id, trace)
            neg_score = self.head_neg_score(neg_head, rel, tail, num_chunks, chunk_size, neg_sample_size)

        else:
            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            # print('--neg_tail_ids.shape:', neg_tail_ids.shape)
            if self.encoder_model_name == 'roberta':
                neg_tail = self.transform_net.embed_entity(self.entity_feat(neg_tail_ids, gpu_id, False))

            elif self.encoder_model_name == 'shallow_net':
                neg_tail = self.transform_net.embed_entity(self.entity_emb(neg_tail_ids, gpu_id, trace))

            elif self.encoder_model_name == 'shallow':
                if self.LRE:
                    neg_tail = torch.mm(
                        self.entity_emb_index(neg_tail_ids, gpu_id, trace),
                        self.entity_emb_base(None, gpu_id, trace))
                else:
                    neg_tail = self.entity_emb(neg_tail_ids, gpu_id, trace)

            elif self.encoder_model_name == 'concat':
                if self.LRE:
                    neg_tail = torch.mm(
                        self.entity_emb_index(neg_tail_ids, gpu_id, trace),
                        self.entity_emb_base(None, gpu_id, trace))
                else:
                    neg_tail = self.entity_emb(neg_tail_ids, gpu_id, trace)
                neg_tail = self.transform_net.embed_entity(torch.cat(
                    [self.entity_feat(neg_tail_ids, gpu_id, False), neg_tail], -1))

            if neg_tail.dtype == torch.float16:
                neg_tail = neg_tail.type(torch.float32)

            head_ids, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None and gpu_id >= 0: head_ids = to_device(head_ids, gpu_id)

            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']

            # This is negative edge construction similar to the above.
            if neg_deg_sample:
                tail = pos_g.ndata['emb'][tail_ids]
                tail = tail.reshape(num_chunks, chunk_size, -1)
                neg_tail = neg_tail.reshape(num_chunks, neg_sample_size, -1)
                neg_tail = F.cat([tail, neg_tail], 1)
                neg_sample_size = chunk_size + neg_sample_size
                mask[:, 0::(neg_sample_size + 1)] = 0
            neg_tail = neg_tail.reshape(num_chunks * neg_sample_size, -1)
            head, neg_tail = self.tail_neg_prepare(
                pos_g.edata['id'], num_chunks, head, neg_tail, gpu_id, trace)
            neg_score = self.tail_neg_score(head, rel, neg_tail, 
                        num_chunks, chunk_size, neg_sample_size)

        if neg_deg_sample:
            neg_g.neg_sample_size = neg_sample_size
            mask = mask.reshape(num_chunks, chunk_size, neg_sample_size)
            return neg_score * mask
        else:
            return neg_score

    def forward_test_wikikg(self, query, ans, candidate, mode, gpu_id=-1):
        """Do the forward and generate ranking results.
        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        logs : List
            Where to put results in.
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        # 将-1的candidate转成编号为0的样本，同时提前记录值为-1的位置，其score应该强制减少
        neg_bias = th.where(candidate == -1, -1e6, 0.)
        candidate_copy = candidate.clone().to(candidate.device)
        candidate = th.where(candidate == -1, 0, candidate)
        scores = self.predict_score_wikikg(query, candidate, mode, to_device=cuda, gpu_id=gpu_id, trace=False)
        scores += neg_bias.to(scores.device)
        argsort = F.argsort(scores, dim=1, descending=True).to(candidate.device)
        # print('--scores.shape: ', scores.shape)
        # print('--argsort.shape: ', argsort.shape)
        return torch.gather(candidate_copy, 1, argsort[:, 0:10]).cpu(), scores.cpu()
        # wikikg90m-v1
        # return argsort[:, :10].cpu(), scores.cpu()

    def predict_score_wikikg(self, query, candidate, mode, to_device=None, gpu_id=-1, trace=False):
        num_chunks = len(query)
        chunk_size = 1
        neg_sample_size = candidate.shape[1]
        if mode == 'h,r->t':
            if self.encoder_model_name == 'roberta':
                neg_tail = self.transform_net.embed_entity(self.entity_feat(candidate.view(-1), gpu_id, False))
                head = self.transform_net.embed_entity(self.entity_feat(query[:, 0], gpu_id, False))
                rel = self.transform_net.embed_relation(self.relation_feat(query[:, 1], gpu_id, False))

            elif self.encoder_model_name == 'shallow_net':
                neg_tail = self.transform_net.embed_entity(self.entity_emb(candidate.view(-1), gpu_id, False))
                head = self.transform_net.embed_entity(self.entity_emb(query[:, 0], gpu_id, False))
                rel = self.transform_net.embed_relation(self.relation_emb(query[:, 1], gpu_id, False))

            elif self.encoder_model_name == 'shallow':
                if self.LRE:
                    neg_tail_index = self.entity_emb_index(candidate.view(-1), gpu_id, False)
                    head_index = self.entity_emb_index(query[:, 0], gpu_id, False)
                    entity_base = self.entity_emb_base(None, gpu_id, False)
                    neg_tail = torch.mm(neg_tail_index, entity_base)
                    head = torch.mm(head_index, entity_base)
                else:
                    neg_tail = self.entity_emb(candidate.view(-1), gpu_id, False)
                    head = self.entity_emb(query[:, 0], gpu_id, False)
                rel = self.relation_emb(query[:, 1], gpu_id, False)

            elif self.encoder_model_name == 'concat':
                if self.LRE:
                    neg_tail_index = self.entity_emb_index(candidate.view(-1), gpu_id, False)
                    head_index = self.entity_emb_index(query[:, 0], gpu_id, False)
                    entity_base = self.entity_emb_base(None, gpu_id, False)
                    neg_tail = torch.mm(neg_tail_index, entity_base)
                    head = torch.mm(head_index, entity_base)
                else:
                    neg_tail = self.entity_emb(candidate.view(-1), gpu_id, False)
                    head = self.entity_emb(query[:, 0], gpu_id, False)

                neg_tail = self.transform_net.embed_entity(torch.cat([self.entity_feat(
                    candidate.view(-1), gpu_id, False), neg_tail], -1))
                head = self.transform_net.embed_entity(torch.cat(
                    [self.entity_feat(query[:, 0], gpu_id, False), head], -1))
                rel = self.transform_net.embed_relation(torch.cat(
                    [self.relation_feat(query[:, 1], gpu_id, False), self.relation_emb(query[:, 1], gpu_id, False)],
                    -1))

            if head.dtype == torch.float16:
                head = head.type(torch.float32)
                rel = rel.type(torch.float32)
                neg_tail = neg_tail.type(torch.float32)

            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)
        else:
            assert False

        return neg_score.squeeze(dim=1)

    def get_loss_weigth(self, pos_g, neg_g, gpu_id=-1):
        """
        """
        # head_ids, tail_ids = pos_g.all_edges(order='eid')
        rel_ids = pos_g.edata['id']
        # print('rel_ids.shape:', rel_ids.shape)
        weigth = self.relation_weight[rel_ids]
        # print('weigth.shape:', weigth.shape)
        w = torch.from_numpy(weigth)
        if int(gpu_id) >=0:
            w = w.to(torch.device('cuda:' + str(gpu_id)))
        # print('w.shape:', w.shape)
        return w

    # @profile
    def forward(self, pos_g, neg_g, gpu_id=-1, rank=0):
        """Do the forward.
        Parameters
        ----------
        pos_g : DGLGraph
            Graph holding positive edges.
        neg_g : DGLGraph
            Graph holding negative edges.
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        Returns
        -------
        tensor
            loss value
        dict
            loss info
        """
        if self.encoder_model_name == 'roberta':
            pos_g.ndata['emb'] = self.transform_net.embed_entity(self.entity_feat(pos_g.ndata['id'], gpu_id, False))
            pos_g.edata['emb'] = self.transform_net.embed_relation(self.relation_feat(pos_g.edata['id'], gpu_id, False))

        if self.encoder_model_name == 'shallow_net':
            pos_g.ndata['emb'] = self.transform_net.embed_entity(self.entity_emb(pos_g.ndata['id'], gpu_id, True))
            pos_g.edata['emb'] = self.transform_net.embed_relation(self.relation_emb(pos_g.edata['id'], gpu_id, True))

        elif self.encoder_model_name == 'shallow':
            if self.LRE:
                emb_base = self.entity_emb_base(None, gpu_id, True)
                emb_index = self.entity_emb_index(pos_g.ndata['id'], gpu_id, True)
                pos_g.ndata['emb'] = torch.mm(emb_index, emb_base)
            else:
                pos_g.ndata['emb'] = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
            pos_g.edata['emb'] = self.relation_emb(pos_g.edata['id'], gpu_id, True)

        elif self.encoder_model_name == 'concat':
            # print("self.entity_feat(pos_g.ndata['id'], gpu_id, False):", self.entity_feat(pos_g.ndata['id'], gpu_id, False).shape)
            # print("self.entity_emb(pos_g.ndata['id'], gpu_id, False):", self.entity_emb(pos_g.ndata['id'], gpu_id, False).shape)
            # print("xx:", torch.cat([self.entity_feat(pos_g.ndata['id'], gpu_id, False), self.entity_emb(pos_g.ndata['id'], gpu_id, True)], -1).shape)
            if self.LRE:
                emb_base = self.entity_emb_base(None, gpu_id, True)
                emb_index = self.entity_emb_index(pos_g.ndata['id'], gpu_id, True)
                pos_g_emb = torch.mm(emb_index, emb_base)
            else:
                pos_g_emb = self.entity_emb(pos_g.ndata['id'], gpu_id, True)
            pos_g.ndata['emb'] = self.transform_net.embed_entity(torch.cat([self.entity_feat(
                pos_g.ndata['id'], gpu_id, False), pos_g_emb], -1)) # 这里需要注意要不要更新!!!!!!!!

            # print("self.relation_feat(pos_g.ndata['id'], gpu_id, False):", self.relation_feat(pos_g.ndata['id'], gpu_id, False).shape)
            # print("self.relation_emb(pos_g.ndata['id'], gpu_id, False):", self.relation_emb(pos_g.ndata['id'], gpu_id, False).shape)
            pos_g.edata['emb'] = self.transform_net.embed_relation(
                torch.cat([
                    self.relation_feat(pos_g.edata['id'], gpu_id, False),
                    self.relation_emb(pos_g.edata['id'], gpu_id, True)  # 这里需要注意要不要更新!!!!!!!!
                ], -1))

        if pos_g.ndata['emb'].dtype == torch.float16:
            pos_g.ndata['emb'] = pos_g.ndata['emb'].type(torch.float32)
            pos_g.edata['emb'] = pos_g.edata['emb'].type(torch.float32)

        self.score_func.prepare(pos_g, gpu_id, True)

        pos_score = self.predict_score(pos_g)

        if gpu_id >= 0:
            neg_score = self.predict_neg_score(pos_g, neg_g, to_device=cuda,
                                               gpu_id=gpu_id, trace=True,  # 这里需要注意要不要更新!!!!!!!!
                                               neg_deg_sample=self.args.neg_deg_sample)
        else:
            neg_score = self.predict_neg_score(pos_g, neg_g, trace=True,  # 这里需要注意要不要更新!!!!!!!!
                                               neg_deg_sample=self.args.neg_deg_sample)

        neg_score = reshape(neg_score, -1, neg_g.neg_sample_size)
        # subsampling weight
        # TODO: add subsampling to new sampler
        # if self.args.non_uni_weight:
        #    subsampling_weight = pos_g.edata['weight']
        #    pos_score = (pos_score * subsampling_weight).sum() / subsampling_weight.sum()
        #    neg_score = (neg_score * subsampling_weight).sum() / subsampling_weight.sum()
        # else:
        # edge_weight = F.copy_to(pos_g.edata['impts'], get_dev(gpu_id)) if self.has_edge_importance else None
        edge_weight = self.get_loss_weigth(pos_g, neg_g, gpu_id) if self.args.use_relation_weight else None
        # print('edge_weight:', edge_weight)
        # print('pos_score.shape:', pos_score.shape)
        # print('neg_score.shape:', neg_score.shape)
        loss, log = self.loss_gen.get_total_loss(pos_score, neg_score, edge_weight)
        # regularization: TODO(zihao)
        # TODO: only reg ent&rel embeddings. other params to be added.
        if self.args.regularization_coef > 0.0 and self.args.regularization_norm > 0 and \
                self.encoder_model_name in ['concat', 'shallow', 'shallow_net']:
            coef, nm = self.args.regularization_coef, self.args.regularization_norm
            reg = coef * norm(self.relation_emb.curr_emb().type(torch.float32), nm)
                          
            if self.LRE:
                emb_index = self.entity_emb_index.curr_emb()
                emb_base = self.entity_emb_base(None, gpu_id, False)
                entity_embeddings = torch.mm(
                    emb_index,
                    emb_base
                )
                reg += coef * norm(entity_embeddings, nm)
            else:
                reg += coef * norm(self.entity_emb.curr_emb().type(torch.float32), nm)
            log['regularization'] = get_scalar(reg)
            loss = loss + reg

        return loss, log

    def update(self, gpu_id=-1):
        """ Update the embeddings in the model
        gpu_id : int
            Which gpu to accelerate the calculation. if -1 is provided, cpu is used.
        """
        if self.encoder_model_name in ['shallow', 'concat', 'shallow_net']:
            if self.LRE:
                self.entity_emb_index.update(gpu_id)
                self.entity_emb_base.update(gpu_id)
            else:
                self.entity_emb.update(gpu_id)
            self.relation_emb.update(gpu_id)
            self.score_func.update(gpu_id)

    def prepare_relation(self, device=None):
        """ Prepare relation embeddings in multi-process multi-gpu training model.
        device : th.device
            Which device (GPU) to put relation embeddings in.
        """
        # print("prepare relation")
        self.relation_emb = ExternalEmbedding(self.args, self.n_relations, self.rel_dim, device)
        self.relation_emb.init(self.emb_init)
        if self.model_name == 'TransR':
            local_projection_emb = ExternalEmbedding(self.args, self.n_relations,
                                                     self.entity_dim * self.rel_dim, device)
            self.score_func.prepare_local_emb(local_projection_emb)
            self.score_func.reset_parameters()

    def prepare_cross_rels(self, cross_rels):
        self.relation_emb.setup_cross_rels(cross_rels, self.global_relation_emb)
        if self.model_name == 'TransR':
            self.score_func.prepare_cross_rels(cross_rels)

    def writeback_relation(self, rank=0, rel_parts=None):
        """ Writeback relation embeddings in a specific process to global relation embedding.
        Used in multi-process multi-gpu training model.
        rank : int
            Process id.
        rel_parts : List of tensor
            List of tensor stroing edge types of each partition.
        """
        idx = rel_parts[rank]
        if self.soft_rel_part:
            idx = self.relation_emb.get_noncross_idx(idx)
        self.global_relation_emb.emb[idx] = F.copy_to(self.relation_emb.emb, F.cpu())[idx]
        if self.model_name == 'TransR':
            self.score_func.writeback_local_emb(idx)

    def load_relation(self, device=None):
        """ Sync global relation embeddings into local relation embeddings.
        Used in multi-process multi-gpu training model.
        device : th.device
            Which device (GPU) to put relation embeddings in.
        """
        self.relation_emb = ExternalEmbedding(self.args, self.n_relations, self.rel_dim, device)
        self.relation_emb.emb = F.copy_to(self.global_relation_emb.emb, device)
        if self.model_name == 'TransR':
            local_projection_emb = ExternalEmbedding(self.args, self.n_relations,
                                                     self.entity_dim * self.rel_dim, device)
            self.score_func.load_local_emb(local_projection_emb)

    def create_async_update(self):
        """Set up the async update for entity embedding.
        """
        if self.encoder_model_name in ['shallow', 'concat', 'shallow_net']:
            if self.LRE:
                self.entity_emb_index.create_async_update()
                self.entity_emb_base.create_async_update()
            else:
                self.entity_emb.create_async_update()

    def finish_async_update(self):
        """Terminate the async update for entity embedding.
        """
        if self.encoder_model_name in ['shallow', 'concat', 'shallow_net']:
            if self.LRE:
                self.entity_emb_index.finish_async_update()
                self.entity_emb_base.finish_async_update()
            else:
                self.entity_emb.finish_async_update()

    def pull_model(self, client, pos_g, neg_g):
        with torch.no_grad():
            entity_id = F.cat(seq=[pos_g.ndata['id'], neg_g.ndata['id']], dim=0)
            relation_id = pos_g.edata['id']
            entity_id = F.tensor(np.unique(F.asnumpy(entity_id)))
            relation_id = F.tensor(np.unique(F.asnumpy(relation_id)))

            l2g = client.get_local2global()
            global_entity_id = l2g[entity_id]

            relation_data = client.pull(name='relation_emb', id_tensor=relation_id)
            self.relation_emb.emb[relation_id] = relation_data

            if self.LRE and self.encoder_model_name == 'shallow':
                entity_emb_index = client.pull(name='entity_emb_index', id_tensor=global_entity_id)
                self.entity_emb_index.emb[entity_id] = entity_emb_index

                entity_emb_base = client.pull(name='entity_emb_base', id_tensor=l2g)
                self.entity_emb_base.emb = entity_emb_base
            else:
                entity_data = client.pull(name='entity_emb', id_tensor=global_entity_id)
                self.entity_emb.emb[entity_id] = entity_data

    def push_gradient(self, client):
        with torch.no_grad():
            l2g = client.get_local2global()
            if self.LRE and self.encoder_model_name == 'shallow':
                for entity_id, entity_data in self.entity_emb_index.trace:
                    grad = entity_data.grad.data
                    global_entity_id = l2g[entity_id]
                    client.push(name='entity_emb_index',
                                id_tensor=global_entity_id, data_tensor=grad)

                for base_id, base_data in self.entity_emb_base.trace:
                    grad = base_data.grad.data
                    client.push(name='entity_emb_base', id_tensor=base_id, data_tensor=grad)

            else:
                for entity_id, entity_data in self.entity_emb.trace:
                    grad = entity_data.grad.data
                    global_entity_id = l2g[entity_id]
                    client.push(name='entity_emb', id_tensor=global_entity_id, data_tensor=grad)

                for relation_id, relation_data in self.relation_emb.trace:
                    grad = relation_data.grad.data
                    client.push(name='relation_emb', id_tensor=relation_id, data_tensor=grad)

        self.entity_emb.trace = []
        self.relation_emb.trace = []
