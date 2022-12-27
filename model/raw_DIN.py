import sys

sys.path.insert(0, '..')

import numpy as np
import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.basemodel import BaseModel
from deepctr_torch.inputs import *
from deepctr_torch.layers import *
from deepctr_torch.layers.sequence import AttentionSequencePoolingLayer


class DIN(BaseModel):
    """Instantiates the Deep Interest Network architecture.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of deep net
    :param dnn_activation: Activation function to use in deep net
    :param att_hidden_size: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool. Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return:  A PyTorch model instance.

    """

    def __init__(self, dnn_feature_columns, history_feature_list, dnn_use_bn=False,
                 dnn_hidden_units=(256, 128), dnn_activation='relu', att_hidden_size=(64, 16),
                 att_activation='Dice', att_weight_normalization=False, l2_reg_dnn=0.0,
                 l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001,
                 seed=1024, task='binary', device='cpu', gpus=None):
        super(DIN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                  init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.sparse_feature_columns = list(
            filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)) if dnn_feature_columns else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)) if dnn_feature_columns else []

        self.history_feature_list = history_feature_list

        self.history_feature_columns = []
        self.sparse_varlen_feature_columns = []
        self.history_fc_names = list(map(lambda x: "hist_" + x, history_feature_list))

        for fc in self.varlen_sparse_feature_columns:
            feature_name = fc.name
            if feature_name in self.history_fc_names:
                self.history_feature_columns.append(fc)
            else:
                self.sparse_varlen_feature_columns.append(fc)

        att_emb_dim = self._compute_interest_dim()

        self.attention = AttentionSequencePoolingLayer(att_hidden_units=att_hidden_size,
                                                       embedding_dim=att_emb_dim,
                                                       att_activation=att_activation,
                                                       return_score=False,
                                                       supports_masking=False,
                                                       weight_normalization=att_weight_normalization)

        self.dnn = DNN(inputs_dim=self.compute_input_dim(dnn_feature_columns),
                       hidden_units=dnn_hidden_units,
                       activation=dnn_activation,
                       dropout_rate=dnn_dropout,
                       l2_reg=l2_reg_dnn,
                       use_bn=dnn_use_bn)
        self.dnn_linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False).to(device)
        self.to(device)


    def forward(self, X):
        # _, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns, self.embedding_dict)

        # sequence pooling part
        query_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                          return_feat_list=self.history_feature_list, to_list=True)
        keys_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.history_feature_columns,
                                         return_feat_list=self.history_fc_names, to_list=True)
        dnn_input_emb_list = embedding_lookup(X, self.embedding_dict, self.feature_index, self.sparse_feature_columns,
                                              to_list=True)

        sequence_embed_dict = varlen_embedding_lookup(X, self.embedding_dict, self.feature_index,
                                                      self.sparse_varlen_feature_columns)

        sequence_embed_list = get_varlen_pooling_list(sequence_embed_dict, X, self.feature_index,
                                                      self.sparse_varlen_feature_columns, self.device)

        dnn_input_emb_list += sequence_embed_list
        deep_input_emb = torch.cat(dnn_input_emb_list, dim=-1)

        # concatenate
        query_emb = torch.cat(query_emb_list, dim=-1)                     # [B, 1, E]
        keys_emb = torch.cat(keys_emb_list, dim=-1)                       # [B, T, E]

        keys_length_feature_name = [feat.length_name for feat in self.varlen_sparse_feature_columns if
                                    feat.length_name is not None]
        keys_length = torch.squeeze(maxlen_lookup(X, self.feature_index, keys_length_feature_name), 1)  # [B, 1]

        hist = self.attention(query_emb, keys_emb, keys_length)           # [B, 1, E]

        import pdb; pdb.set_trace()

        # deep part
        deep_input_emb = torch.cat((deep_input_emb, hist), dim=-1)
        deep_input_emb = deep_input_emb.view(deep_input_emb.size(0), -1)

        dnn_input = combined_dnn_input([deep_input_emb], dense_value_list)
        dnn_output = self.dnn(dnn_input)
        dnn_logit = self.dnn_linear(dnn_output)

        y_pred = self.out(dnn_logit)

        return y_pred


    def _compute_interest_dim(self):
        interest_dim = 0
        for feat in self.sparse_feature_columns:
            if feat.name in self.history_feature_list:
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

    # hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [1, 2, 0, 0]])
    # hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [2, 1, 0, 0]])
    # behavior_length = np.array([3, 3, 2])

    hist_iid = np.array([[1, 2, 3, 0], [1, 2, 3, 0], [0,0,0,0]])
    hist_igender = np.array([[1, 1, 2, 0], [2, 1, 1, 0], [0,0,0,0]])
    behavior_length = np.array([3, 3, 0])

    feature_dict = {'user': uid, 'gender': ugender, 'item': iid, 'item_gender': igender,
                    'hist_item': hist_iid, 'hist_item_gender': hist_igender, 'score': score,
                    "seq_length": behavior_length}
    x = {name: feature_dict[name] for name in get_feature_names(feature_columns)}
    y = np.array([1, 0, 1])

    import pdb; pdb.set_trace()

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