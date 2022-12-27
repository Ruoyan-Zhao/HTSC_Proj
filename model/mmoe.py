# from torchsampler import ImbalancedDatasetSampler
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
# from deepctr_torch.models.deepfm import *
from deepctr_torch.models.basemodel import *
from deepctr_torch.models.deepfm import *
from deepctr_torch.layers import BiInteractionPooling
from deepctr_torch.layers import activation_layer

from collections import defaultdict
from multiprocessing import Pool

import os,sys
sys.path.append('../')
from utils import uAUC
import copy
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def create_embedding_matrix(feature_columns, init_std=0.0001, linear=False, sparse=False, device='cpu'):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    for tensor in embedding_dict.values():
        nn.init.normal_(tensor.weight, mean=0, std=init_std)
        # nn.init.kaiming_uniform_(tensor.weight, mode='fan_in', nonlinearity='relu')

    return embedding_dict.to(device)


class DNN(nn.Module):
    """The Multi Layer Percetron

      Input shape
        - nD tensor with shape: ``(batch_size, ..., input_dim)``. The most common situation would be a 2D input with shape ``(batch_size, input_dim)``.

      Output shape
        - nD tensor with shape: ``(batch_size, ..., hidden_size[-1])``. For instance, for a 2D input with shape ``(batch_size, input_dim)``, the output would have shape ``(batch_size, hidden_size[-1])``.

      Arguments
        - **inputs_dim**: input feature dimension.

        - **hidden_units**:list of positive integer, the layer number and units in each layer.

        - **activation**: Activation function to use.

        - **l2_reg**: float between 0 and 1. L2 regularizer strength applied to the kernel weights matrix.

        - **dropout_rate**: float in [0,1). Fraction of the units to dropout.

        - **use_bn**: bool. Whether use BatchNormalization before activation or not.

        - **seed**: A Python integer to use as random seed.
    """

    def __init__(self, inputs_dim, hidden_units, activation='relu', l2_reg=0, dropout_rate=0, use_bn=False,
                 init_std=0.0001, dice_dim=3, seed=1024, device='cpu'):
        super(DNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.seed = seed
        self.l2_reg = l2_reg
        self.use_bn = use_bn
        if len(hidden_units) == 0:
            raise ValueError("hidden_units is empty!!")
        hidden_units = [inputs_dim] + list(hidden_units)

        self.linears = nn.ModuleList(
            [nn.Linear(hidden_units[i], hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        if self.use_bn:
            self.bn = nn.ModuleList(
                [nn.BatchNorm1d(hidden_units[i + 1]) for i in range(len(hidden_units) - 1)])

        self.activation_layers = nn.ModuleList(
            [activation_layer(activation, hidden_units[i + 1], dice_dim) for i in range(len(hidden_units) - 1)])

        for name, tensor in self.linears.named_parameters():
            if 'weight' in name:
                # nn.init.normal_(tensor, mean=0, std=init_std)
                # nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
                nn.init.kaiming_normal_(tensor, mode='fan_in', nonlinearity='relu')

        self.to(device)

    def forward(self, inputs):
        deep_input = inputs

        for i in range(len(self.linears)):

            fc = self.linears[i](deep_input)

            if self.use_bn:
                fc = self.bn[i](fc)

            fc = self.activation_layers[i](fc)
            fc = self.dropout(fc)
            
            deep_input = fc
        return deep_input



def create_embedding_from_pretrained(feature_columns, init_std=0.0001, linear=False, sparse=False, 
                                     device='cpu', pretrained_embedding_dict=None, frozen_pretrained=False):
    # Return nn.ModuleDict: for sparse features, {embedding_name: nn.Embedding}
    # for varlen sparse features, {embedding_name: nn.EmbeddingBag}
    sparse_feature_columns = list(
        filter(lambda x: isinstance(x, SparseFeat), feature_columns)) if len(feature_columns) else []

    varlen_sparse_feature_columns = list(
        filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if len(feature_columns) else []

    embedding_dict = nn.ModuleDict(
        {feat.embedding_name: nn.Embedding(feat.vocabulary_size, feat.embedding_dim if not linear else 1, sparse=sparse)
         for feat in
         sparse_feature_columns + varlen_sparse_feature_columns}
    )

    # for feat in varlen_sparse_feature_columns:
    #     embedding_dict[feat.embedding_name] = nn.EmbeddingBag(
    #         feat.dimension, embedding_size, sparse=sparse, mode=feat.combiner)

    for name, tensor in embedding_dict.items():
        if (pretrained_embedding_dict is not None) and  (name in pretrained_embedding_dict):
            tensor.weight.data.copy_(torch.from_numpy(pretrained_embedding_dict[name]))
            if frozen_pretrained:
                tensor.weight.requires_grad = False
        else:
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict.to(device)


class MyBaseModel(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns, l2_reg_linear=1e-5, l2_reg_embedding=1e-5,
                 init_std=0.0001, seed=1024, task='binary', device='cpu', gpus=None, 
                 pretrained_embedding_dict=None, frozen_pretrained=False, num_tasks=4):

        super(BaseModel, self).__init__()
        torch.manual_seed(seed)
        self.dnn_feature_columns = dnn_feature_columns

        self.reg_loss = torch.zeros((1,), device=device)
        self.aux_loss = torch.zeros((1,), device=device)
        self.device = device
        self.gpus = gpus
        if gpus and str(self.gpus[0]) not in self.device:
            raise ValueError(
                "`gpus[0]` should be the same gpu with `device`")

        self.feature_index = build_input_features(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, init_std, sparse=False, device=device)
        # self.embedding_dict = create_embedding_from_pretrained(dnn_feature_columns, init_std, sparse=False, 
        #                                                       device=device, pretrained_embedding_dict=pretrained_embedding_dict,
        #                                                       frozen_pretrained=frozen_pretrained)

        self.regularization_weight = []
        self.add_regularization_weight(self.embedding_dict.parameters(), l2=l2_reg_embedding)

        self.to(device)

        # parameters of callbacks
        self._is_graph_network = True  # used for ModelCheckpoint
        self.stop_training = False  # used for EarlyStopping
        self.history = History()

    def fit(self, train_loader, epochs=1, verbose=1, initial_epoch=0, validation_split=0.,
            validation_data=None, callbacks=None, num_warm_epochs=1, 
            lr_scheduler=True, scheduler_epochs=5, reduction='sum', task_weight=None, task_dict=None,
            scheduler_method='cos', early_stop_uauc=0.55, label_smoothing=0.2):
        """
        :param x: Numpy array of training data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).If input layers in the model are named, you can also pass a
            dictionary mapping input names to Numpy arrays.
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per gradient update. If unspecified, `batch_size` will default to 256.
        :param epochs: Integer. Number of epochs to train the model. An epoch is an iteration over the entire `x` and `y` data provided. Note that in conjunction with `initial_epoch`, `epochs` is to be understood as "final epoch". The model is not trained for a number of iterations given by `epochs`, but merely until the epoch of index `epochs` is reached.
        :param verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
        :param initial_epoch: Integer. Epoch at which to start training (useful for resuming a previous training run).
        :param validation_split: Float between 0 and 1. Fraction of the training data to be used as validation data. The model will set apart this fraction of the training data, will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. The validation data is selected from the last samples in the `x` and `y` data provided, before shuffling.
        :param validation_data: tuple `(x_val, y_val)` or tuple `(x_val, y_val, val_sample_weights)` on which to evaluate the loss and any model metrics at the end of each epoch. The model will not be trained on this data. `validation_data` will override `validation_split`.
        :param shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of each epoch.
        :param callbacks: List of `deepctr_torch.callbacks.Callback` instances. List of callbacks to apply during training and validation (if ). See [callbacks](https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks). Now available: `EarlyStopping` , `ModelCheckpoint`

        :return: A `History` object. Its `History.history` attribute is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
        """
        do_validation = False
        if validation_data and len(validation_data)==3:
            do_validation = True
            val_x_loader, val_y, val_userid_list = validation_data
        else:
            val_y = []

        model = self.train()
        loss_func = self.loss_func
        optim = self.optim
        if lr_scheduler:
            if scheduler_method=='linear':
                scheduler = get_linear_schedule_with_warmup(
                   optim,
                   num_warmup_steps=int(len(train_loader)*num_warm_epochs),
                   num_training_steps=int(len(train_loader)*(scheduler_epochs)))
            else:
                scheduler = get_cosine_schedule_with_warmup(
                   optim,
                   num_warmup_steps=int(len(train_loader)*num_warm_epochs),
                   num_training_steps=int(len(train_loader)*(scheduler_epochs)))
        # print('0000 - - reload')
        if self.gpus:
            print('parallel running on these gpus:', self.gpus)
            model = torch.nn.DataParallel(model, device_ids=self.gpus)
            batch_size *= len(self.gpus)  # input `batch_size` is batch_size per gpu
        else:
            pass

        steps_per_epoch = len(train_loader)

        # configure callbacks
        callbacks = (callbacks or []) + [self.history]  # add history callback
        callbacks = CallbackList(callbacks)
        callbacks.on_train_begin()
        callbacks.set_model(self)
        if not hasattr(callbacks, 'model'):
            callbacks.__setattr__('model', self)
        callbacks.model.stop_training = False

        # Train
        best_metric = -1
        early_stopping_flag = False
        result_logs = {}
        logging.info("Train on {0} samples, validate on {1} samples, {2} steps per epoch".format(
            len(train_loader)*train_loader.batch_size, len(val_y), steps_per_epoch))
        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            epoch_logs = {}
            start_time = time.time()
            loss_epoch = 0
            total_loss_epoch = 0
            sample_num = 0
            train_result = {}
            try:
                with tqdm(enumerate(train_loader), disable=verbose != 1) as t:
                    for _, (x_train, y_train) in t:
                        x = x_train.to(self.device).float()
                        y = y_train.to(self.device).float()
                        sample_num += x.shape[0]
                        y_pred = model(x)#.squeeze()

                        optim.zero_grad()
                        reg_loss = self.get_regularization_loss()
                        total_loss = reg_loss + self.aux_loss
                        for i in range(self.num_tasks):
                            loss = loss_func(y_pred[:,i], y[:,i], reduction=reduction)  # y.squeeze()
                            total_loss += loss*task_weight[i]
                            loss_epoch += loss.item()
                            # 输出日志用
                            loss_name = task_dict[i]+'_loss'
                            if loss_name not in train_result:
                                train_result[loss_name] = []
                            train_result[loss_name].append(loss.item())
                        
                        total_loss_epoch += total_loss.item()
                        total_loss.backward()
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        #torch.nn.utils.clip_grad_value_(model.parameters(), 10.)
                        optim.step()
                        if lr_scheduler:
                            scheduler.step() 
                        
                        # 为节约时间，训练集只输出loss，不做评估
                        if False: # verbose>0:
                            for name, metric_fun in self.metrics.items():
                                # 训练集不做auc和uauc的评估
                                if (name=='uauc'): continue
                                if (name=='auc'): continue
                                # 每个评估函数都要评估多个任务
                                for i in range(model.num_tasks):
                                    task_name = task_dict[i]+'_'+name
                                    if task_name not in train_result:
                                        train_result[task_name] = []
                                    try:
                                        train_result[task_name].append(metric_fun(
                                            y[:,i].squeeze().cpu().data.numpy(), 
                                            y_pred[i].cpu().data.numpy().astype('float64')))
                                    except:
                                        pass

            except KeyboardInterrupt:
                t.close()
                raise
            t.close()

            # Add epoch_logs, training logs
            epoch_logs["loss"] = total_loss_epoch / sample_num
            for name, result in train_result.items():
                epoch_logs[name] = np.sum(result) / sample_num  # len(result)

            if do_validation:
                eval_result = self.evaluate(val_x_loader, val_y, val_userid_list, task_dict)
                for name, result in eval_result.items():
                    epoch_logs["val_" + name] = result
                    
            # verbose
            if verbose > 0:
                epoch_time = int(time.time() - start_time)
                eval_str = 'Epoch {0}/{1}'.format(epoch + 1, epochs)

                eval_str += " {0}s - loss: {1: .4f}".format(epoch_time, epoch_logs["loss"])
                
                # 输出训练集的评估结果
                # for name in self.metrics:
                #     if name=='uauc': continue
                #     for i in range(self.num_tasks):
                #         task_name = [i]+'_'+name
                #         eval_str += " - " + task_name + ": {0: .4f}".format(epoch_logs[task_name])
                for name, result in train_result.items():
                    eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])
                
                # 输出验证集的评估结果
                if do_validation:
                    for name in self.metrics:
                        for i in range(self.num_tasks):
                            task_name = task_dict[i]+'_'+name
                            eval_str += " - " + "val_" + task_name + \
                                    ": {0: .4f}".format(epoch_logs["val_" + task_name])

                    best_metric = max(best_metric, eval_result['UAUC'])
                    if best_metric<=early_stop_uauc:
                        # 验证集uauc<=0.55时，提前终止训练，以节约时间
                        early_stopping_flag = True
                    epoch_logs['val_UAUC'] = eval_result['UAUC']
                    eval_str += ' - val_UAUC: {0: .5f}'.format(eval_result['UAUC'])
                logging.info(eval_str)
                
            result_logs['epoch'+str(epoch+1)] = epoch_logs
            # 验证集uauc==0.5时，提前终止训练，以节约时间
            if early_stopping_flag:
                callbacks.on_epoch_end(epoch, epoch_logs)
                callbacks.on_train_end()
                return self.history, best_metric, result_logs
                
            callbacks.on_epoch_end(epoch, epoch_logs)
            if self.stop_training:
                break

        callbacks.on_train_end()

        return self.history, best_metric, result_logs

    def evaluate(self, x, y, userid_list, task_dict=None):
        """
        :param x: Numpy array of test data (if the model has a single input), or list of Numpy arrays (if the model has multiple inputs).
        :param y: Numpy array of target (label) data (if the model has a single output), or list of Numpy arrays (if the model has multiple outputs).
        :param batch_size: Integer or `None`. Number of samples per evaluation step. If unspecified, `batch_size` will default to 256.
        :return: Dict contains metric names and metric values.
        """
        pred_ans = self.predict(x)
        eval_result = {}
        eval_result['UAUC'] = 0
        
        # 外层多个评估函数
        for name, metric_fun in self.metrics.items():
            # 内层每个评估函数都要对多个任务进行评估
            # 生成参数
            if name=='uauc':
                uauc, task_uaucs = metric_fun(y, pred_ans, userid_list) 
                eval_result['UAUC'] = uauc
                for i in range(self.num_tasks):
                    eval_result[task_dict[i]+'_'+name] = task_uaucs[i]
            else:
                for i in range(self.num_tasks):
                    eval_result[task_dict[i]+'_'+name] = metric_fun(y[:,i], pred_ans[:,i])
            
        return eval_result

    def predict(self, test_loader):
        """
        :param x: The input data, as a Numpy array (or list of Numpy arrays if the model has multiple inputs).
        :param batch_size: Integer. If unspecified, it will default to 256.
        :return: Numpy array(s) of predictions.
        """
        model = self.eval()

        pred_ans = np.empty([0, self.num_tasks])
        with torch.no_grad():
            for _, (x_test) in enumerate(test_loader):
                x = x_test[0].to(self.device).float()
                y_pred = model(x).cpu().numpy()  # .squeeze()
                pred_ans = np.vstack([pred_ans, y_pred])

        return pred_ans

    def add_regularization_weight(self, weight_list, l1=0.0, l2=0.0):
        # For a Parameter, put it in a list to keep Compatible with get_regularization_loss()
        if isinstance(weight_list, torch.nn.parameter.Parameter):
            weight_list = [weight_list]
        # For generators, filters and ParameterLists, convert them to a list of tensors to avoid bugs.
        # e.g., we can't pickle generator objects when we save the model.
        else:
            weight_list = list(weight_list)
        self.regularization_weight.append((weight_list, l1, l2))

    def get_regularization_loss(self, ):
        total_reg_loss = torch.zeros((1,), device=self.device)
        for weight_list, l1, l2 in self.regularization_weight:
            for w in weight_list:
                if isinstance(w, tuple):
                    parameter = w[1]  # named_parameters
                else:
                    parameter = w
                if l1 > 0:
                    total_reg_loss += torch.sum(l1 * torch.abs(parameter))
                if l2 > 0:
                    try:
                        total_reg_loss += torch.sum(l2 * torch.square(parameter))
                    except AttributeError:
                        total_reg_loss += torch.sum(l2 * parameter * parameter)

        return total_reg_loss

    def add_auxiliary_loss(self, aux_loss, alpha):
        self.aux_loss = aux_loss * alpha

    def compile(self, optimizer,
                learning_rate=0.01,
                loss=None,
                metrics=None,
                ):
        """
        :param optimizer: String (name of optimizer) or optimizer instance. See [optimizers](https://pytorch.org/docs/stable/optim.html).
        :param loss: String (name of objective function) or objective function. See [losses](https://pytorch.org/docs/stable/nn.functional.html#loss-functions).
        :param metrics: List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']`.
        """
        self.metrics_names = ["loss"]
        self.optim = self._get_optim(optimizer, learning_rate)
        self.loss_func = self._get_loss_func(loss)
        self.metrics = self._get_metrics(metrics)

    def _get_optim(self, optimizer, learning_rate):
        if isinstance(optimizer, str):
            if optimizer == "sgd":
                optim = torch.optim.SGD(self.parameters(), lr=learning_rate)
            elif optimizer == "adam":
                optim = torch.optim.Adam(self.parameters(), lr=learning_rate)  # 0.001
            elif optimizer == "adagrad":
                optim = torch.optim.Adagrad(self.parameters(), lr=learning_rate)  # 0.01
            elif optimizer == "rmsprop":
                optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
            elif optimizer == 'adamw':
                optim = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=0.)
            elif optimizer == 'adamax':
                optim = torch.optim.Adamax(self.parameters(), lr=learning_rate, weight_decay=0.)
            elif optimizer == 'momentum':
                optim = torch.optim.SGD(self.parameters(),lr=learning_rate,momentum=0.9,nesterov=True)
            else:
                raise NotImplementedError
        else:
            optim = optimizer
        return optim
    
    def _get_metrics(self, metrics, set_eps=False):
        metrics_ = {}
        if metrics:
            for metric in metrics:
                if metric == "binary_crossentropy" or metric == "logloss":
                    if set_eps:
                        metrics_[metric] = self._log_loss
                    else:
                        metrics_[metric] = log_loss
                if metric == "auc":
                    metrics_[metric] = roc_auc_score
                if metric == "mse":
                    metrics_[metric] = mean_squared_error
                if metric == "accuracy" or metric == "acc":
                    metrics_[metric] = lambda y_true, y_pred: accuracy_score(
                        y_true, np.where(y_pred > 0.5, 1, 0))
                
                # 添加uauc metric
                if metric == 'uauc':
                    metrics_[metric] = uAUC
                self.metrics_names.append(metric)
        return metrics_

    def _get_loss_func(self, loss):
        if isinstance(loss, str):
            if loss == "binary_crossentropy":
                loss_func = F.binary_cross_entropy
            elif loss == "mse":
                loss_func = F.mse_loss
            elif loss == "mae":
                loss_func = F.l1_loss
            else:
                raise NotImplementedError
        else:
            loss_func = loss
        return loss_func

    def _log_loss(self, y_true, y_pred, eps=1e-7, normalize=True, sample_weight=None, labels=None):
        # change eps to improve calculation accuracy
        return log_loss(y_true,
                        y_pred,
                        eps,
                        normalize,
                        sample_weight,
                        labels)


class PretrainedEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim, init_weight):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.emb.weight.data.copy_(torch.from_numpy(init_weight))
        self.emb.weight.requires_grad = False

    def forward(self, x):
        return self.emb(x)


class MMOE(MyBaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, use_fm=False, use_nfm=False,
                 dnn_hidden_units=(256, 128), l2_reg_linear=0.00001, l2_reg_embedding=0.00001, 
                 l2_reg_dnn=0, init_std=0.0001, seed=1024, dnn_dropout=0, bi_dropout=0,
                 dnn_activation='relu', dnn_use_bn=True, task='binary', device='cpu', gpus=None,
                 pretrained_embedding_dict=None, frozen_pretrained=False, num_tasks=4, 
                 pretrained_user_emb_weight=None, pretrained_author_emb_weight=None,
                 pretrained_feed_emb_weight=None, pretrained_bgm_song_emb_weight=None,
                 pretrained_bgm_singer_emb_weight=None):

        super(MMOE, self).__init__(linear_feature_columns, dnn_feature_columns, l2_reg_linear=l2_reg_linear,
                                       l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                       device=device, gpus=gpus, pretrained_embedding_dict=pretrained_embedding_dict, 
                                       frozen_pretrained=frozen_pretrained, num_tasks=num_tasks)

        self.num_tasks = num_tasks
        self.use_fm = use_fm
        self.use_nfm = use_nfm
        
        self.pretrained_user = pretrained_user_emb_weight is not None
        self.pretrained_feed = pretrained_feed_emb_weight is not None
        self.pretrained_author = pretrained_author_emb_weight is not None
        self.user_emb = []
        self.feed_emb = []
        self.author_emb = []

        self.pretrained_emb_dim = 0
        # import pdb; pdb.set_trace()
        # 载入预训练的的Embedding矩阵
        if self.pretrained_user:
            for emb_w in pretrained_user_emb_weight:
                self.user_emb.append(PretrainedEmbedding(emb_w.shape[0],
                                                         emb_w.shape[1],
                                                         emb_w).to(device))
                self.pretrained_emb_dim += emb_w.shape[1]
            self.user_emb = nn.ModuleList(self.user_emb)
        
        if self.pretrained_author:
            for emb_w in pretrained_author_emb_weight:
                self.author_emb.append(PretrainedEmbedding(emb_w.shape[0],
                                                emb_w.shape[1],
                                                emb_w).to(device))
                self.pretrained_emb_dim += emb_w.shape[1]
            self.author_emb = nn.ModuleList(self.author_emb)

        if self.pretrained_feed:
            for emb_w in pretrained_feed_emb_weight:
                self.feed_emb.append(PretrainedEmbedding(emb_w.shape[0],
                                            emb_w.shape[1],
                                            emb_w).to(device))
                self.pretrained_emb_dim += emb_w.shape[1]
            self.feed_emb = nn.ModuleList(self.feed_emb)
        
        if use_fm:
            self.fm = FM()
        
        dnn_input_dim = self.compute_input_dim(dnn_feature_columns)
        dnn_input_dim += self.pretrained_emb_dim

        
        Num_Expert = 2
        # self.expert_dnn = nn.ModuleList([DNN(dnn_input_dim, dnn_hidden_units,
        #                activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
        #                init_std=init_std, device=device) for i in range(Num_Expert)])
        # self.mmoe_dnn = MultiGateDNN(dnn_input_dim, self.expert_dnn, num_experts=Num_Expert, num_tasks = self.num_tasks, device=device)

        # dnn_1 = 784
        # self.mmoe_dnn = MultiGateDNN(dnn_input_dim, dnn_input_dim, num_experts=Num_Expert, num_tasks = self.num_tasks, device=device)

        # dnn_hidden_units = dnn_hidden_units[1:]
        self.dnn = nn.ModuleList([DNN(dnn_input_dim,dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=dnn_use_bn,
                       init_std=init_std, device=device) for i in range(self.num_tasks)])
        


        # 相关性较强的几个任务, dnn最后一层的输出可以和相关任务输入合并输入最后一层 !!!!!!!!!!!!
        # INDEX :[read_comment, like, click_avatar, forward, favorite, comment, follow, finish]
        # read_comment & comment / comment = 0.5585950692333671 =>  read_comment可以作为comment的输入
        # click_avatar & follow / follow = 0.9833238582527951   => click_avatar,like, 可以作为 follow 的输入
        dnn_linear = []
        for i in range(self.num_tasks):
            if (i==5):
                dnn_linear.append(nn.Linear(dnn_hidden_units[-1]*2, 1, bias=False))
            elif (i==6) :
                dnn_linear.append(nn.Linear(dnn_hidden_units[-1]*3, 1, bias=False))
            else:
                dnn_linear.append(nn.Linear(dnn_hidden_units[-1], 1, bias=False))
        
        # for i in range(self.num_tasks):
        #     dnn_linear.append(nn.Linear(dnn_hidden_units[i][-1], 1, bias=False))

        self.dnn_linear = nn.ModuleList(dnn_linear).to(device)

        for dnn_ in self.dnn:
            self.add_regularization_weight(
                filter(lambda x: 'weight' in x[0] and 'bn' not in x[0], dnn_.named_parameters()), l2=l2_reg_dnn)
        for task_out_linear in self.dnn_linear:
            self.add_regularization_weight(task_out_linear.weight, l2=l2_reg_dnn)
                
        self.out = nn.ModuleList([PredictionLayer(task, ) for i in range(self.num_tasks)])
        self.to(device)


    def forward(self, X):
        sparse_embedding_list, dense_value_list = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                                  self.embedding_dict)

        pretrained_emb = []
        # 预训练Embedding
        if self.pretrained_user:
            for emb in self.user_emb:
                pretrained_emb.append(emb(X[:,self.feature_index['userid'][0]].long()))
        if self.pretrained_author:
            for emb in self.author_emb:
                pretrained_emb.append(emb(X[:,self.feature_index['authorid'][0]].long()))
        if self.pretrained_feed:
            for emb in self.feed_emb:
                pretrained_emb.append(emb(X[:,self.feature_index['feedid'][0]].long()))


        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        # import pdb; pdb.set_trace()

        dnn_input = torch.cat([dnn_input]+pretrained_emb, dim=1)
        
        # mmoe_outputs = self.mmoe_dnn(dnn_input)
        # 
        dnn_outputs = []
        for i, task_dnn in enumerate(self.dnn):
            # dnn_output = task_dnn(mmoe_outputs[i])
            dnn_output = task_dnn(dnn_input)
            dnn_outputs.append(dnn_output)
        
        # 相关性较强的几个任务, dnn最后一层的输出可以和相关任务输入合并输入最后一层 !!!!!!!!!!!!
        # INDEX :[read_comment, like, click_avatar, forward, favorite, comment, follow, finish]
        # read_comment & comment / comment = 0.5585950692333671 =>  read_comment可以作为comment的输入
        # click_avatar & follow / follow = 0.9833238582527951   => click_avatar,like,[finish] 可以作为 follow的输入
        dnn_outputs[5] = torch.cat([dnn_outputs[0], dnn_outputs[5]], dim=1)
        dnn_outputs[6] = torch.cat([dnn_outputs[1],dnn_outputs[2], dnn_outputs[6]], dim=1)

        # import pdb; pdb.set_trace()
        
        logit = []
        i = 0
        for task_dnn_linear, task_out in zip(self.dnn_linear, self.out):
            dnn_logit = task_dnn_linear(dnn_outputs[i])
            logit.append(task_out(dnn_logit))
            i += 1
        # print(torch.cat(logit, dim=1).shape)
        return torch.cat(logit, dim=1)


class MultiGateDNN(nn.Module):
    # def __init__(self,input_dim, expert_dnn, num_experts, num_tasks, device):
    def __init__(self,input_dim, hid_units, num_experts, num_tasks, device):
        super(MultiGateDNN, self).__init__()

        w = torch.empty(input_dim, hid_units, num_experts,device=device)
        self.expert_kernels = torch.nn.Parameter(torch.nn.init.xavier_normal_(w),
                                                 requires_grad=True)
        self.expert_kernels_bias = torch.nn.Parameter(torch.zeros(hid_units, num_experts, device=device),
                                                      requires_grad=True)

        # self.expert_dnn = nn.ModuleList([copy.deepcopy(expert_dnn) for i in range(num_experts)])
        # self.expert_dnn = expert_dnn
        

        w = torch.empty(input_dim, num_experts, device=device)
        self.gate_kernels = torch.nn.ParameterList(
            [nn.Parameter(torch.nn.init.xavier_normal_(w), requires_grad=True) for i in
             range(num_tasks)])

        
        self.gate_kernels_bias = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(num_experts, device=device), requires_grad=True) for i in range(num_tasks)])

        # self.output_layer = nn.ModuleList([nn.Sequential(
        #     nn.Linear(config.units,config.hidden_units),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_units,unit),
        # )
        #     for unit in config.label_dict
        # ])

        self.expert_activation = nn.ReLU()
        self.gate_activation = nn.Softmax(dim=-1)

        # self.embedding_layer = nn.Embedding(config.num_feature,config.embed_size)

    def forward(self,x):
        gate_outputs = []
        final_outputs = []
        expert_outputs = torch.einsum("ab,bcd->acd", (x, self.expert_kernels))
        expert_outputs += self.expert_kernels_bias
        expert_outputs = self.expert_activation(expert_outputs)
        # expert_outputs = []
        # for dnn_ in self.expert_dnn:
        #     expert_out = dnn_(x)
        #     expert_outputs.append(expert_out)
        # expert_outputs = torch.stack(expert_outputs, dim=-1)

        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = torch.einsum("ab,bc->ac", (x, gate_kernel))
            gate_output += self.gate_kernels_bias[index]
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        for gate_output in gate_outputs:
            expanded_gate_output = torch.unsqueeze(gate_output, 1)
            weighted_expert_output = expert_outputs * expanded_gate_output.expand_as(expert_outputs)
            final_outputs.append(torch.sum(weighted_expert_output, 2))

        # output_layers = []
        # for i,output in enumerate(final_outputs):
        #     output_layers.append(torch.sigmoid(self.output_layer[i](output)))

        return final_outputs