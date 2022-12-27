import sys

sys.path.insert(0, '..')

import numpy as np
import torch
from torch import nn
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import *
from deepctr_torch.layers import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer


class my_DIN(nn.Module):
    def __init__(self, din_feature_columns, history_feature_list, embedding_dict, feature_index, hist_act_pre = 'hist_',att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=True, init_std=0.0001, device='cpu', gpus=None):
        super(my_DIN, self).__init__()
        self.embedding_dict = embedding_dict
        self.feature_index = feature_index

        self.din_feature_columns = din_feature_columns

        # self.history_feature_list = history_feature_list

        # self.history_fc_names = list(map(lambda x: hist_act_pre + x, history_feature_list))

        self.history_fc_names =[i.name for i in din_feature_columns]

        att_emb_dim = self._compute_interest_dim()
        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)
        self.to(device)

    def forward(self, X):
        query_emb_list,keys_emb_list=[],[]
        seq_embed_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.din_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        
        for embed in seq_embed_list:
            query_emb_list.append(embed[:,0:1,:])
            keys_emb_list.append(embed[:,1:,:])
        
        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        hist = None
        keys_length_feature_name = [feat.length_name for feat in self.din_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]
        return hist, query_emb_list


    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.din_feature_columns:
            interest_dim += feat.embedding_dim
        return interest_dim



def get_xy_fd():
    feature_columns = [SparseFeat('user', 3, embedding_dim=8), SparseFeat('gender', 2, embedding_dim=8),
                       SparseFeat('item', 3 + 1, embedding_dim=8), SparseFeat('item_gender', 2 + 1, embedding_dim=8),
                       DenseFeat('score', 1)]

    feature_columns += [VarLenSparseFeat(SparseFeat('hist_item', 3 + 1, embedding_dim=8), 4, length_name="seq_length"),
                        VarLenSparseFeat(SparseFeat('hist_item_gender', 2 + 1, embedding_dim=8), 4, length_name="seq_length")]
    behavior_feature_list = ["item", "item_gender"]
    uid = np.array([0, 1, 2])
    ugender = np.array([0, 1, 0])
    iid = np.array([1, 2, 3])  # 0 is mask value
    igender = np.array([1, 2, 1])  # 0 is mask value
    score = np.array([0.1, 0.2, 0.3])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    behavior_length = np.array([3, 3, 2])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
                    "seq_length": behavior_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    return x, y, feature_columns, behavior_feature_list


if __name__ == "__main__":
    x, y, feature_columns, behavior_feature_list = get_xy_fd()
    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DIN(feature_columns, behavior_feature_list, device=device, att_weight_normalization=True)
    model.compile('adagrad', 'binary_crossentropy',
                  metrics=['binary_crossentropy'])
    history = model.fit(x, y, batch_size=3, epochs=10, verbose=2, validation_split=0.0)