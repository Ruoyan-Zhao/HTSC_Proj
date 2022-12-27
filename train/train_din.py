import os, sys, gc, pickle
import preprocess
sys.path.append('../')
# from model.moe import MOE
# from model.mmoe import MMOE
from model.moe_DIN import MOE_DIN
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, VarLenSparseFeat, get_feature_names
import random

import logging
# logger = logging.getLogger(__name__)


def train_my_model(args, np_rd_seed=2345, rd_seed=2345, torch_seed=1233):
    np.random.seed(np_rd_seed)
    random.seed(rd_seed)
    model = MOE_DIN( linear_feature_columns=args['linear_feature_columns'],history_feature_list = ['feedid','authorid'],
              dnn_feature_columns=args['dnn_feature_columns'],task='binary',dnn_dropout=args['dropout'],dnn_hidden_units=args['hidden_units'],
              l2_reg_embedding=args['l2_reg_embedding'], l2_reg_dnn=args['l2_reg_dnn'],
              device=device, seed=torch_seed, num_tasks=args['num_tasks'],
            # 预训练embedding !!!!!!!!!!!!
              pretrained_user_emb_weight=[user_emb_weight],
              pretrained_author_emb_weight=[author_emb_weight],
              pretrained_feed_emb_weight=[feed_emb_weight, official_feed_weight],
              )
    
    # model = MOE( linear_feature_columns=args['linear_feature_columns'],
    #           dnn_feature_columns=args['dnn_feature_columns'],task='binary',dnn_dropout=args['dropout'],dnn_hidden_units=args['hidden_units'],
    #           l2_reg_embedding=args['l2_reg_embedding'], l2_reg_dnn=args['l2_reg_dnn'],
    #           device=device, seed=torch_seed, num_tasks=args['num_tasks'],
    #         # 预训练embedding !!!!!!!!!!!!
    #         #   pretrained_user_emb_weight=[user_emb_weight],
    #         #   pretrained_author_emb_weight=[author_emb_weight],
    #         #   pretrained_feed_emb_weight=[feed_emb_weight, official_feed_weight],
    #           )

    model.compile(optimizer=args['optimizer'], learning_rate=args['learning_rate'], 
                loss="binary_crossentropy", 
                metrics=['uauc'])
                # metrics=["binary_crossentropy",'auc','uauc'])

    metric = model.fit(train_loader, validation_data=[val_x_loader, val_y, val_userid_list],
                       epochs=args['epochs'], lr_scheduler=args['lr_scheduler'], scheduler_epochs=args['scheduler_epochs'],
                       scheduler_method=args['scheduler_method'], num_warm_epochs=args['num_warm_epochs'],
                       reduction=args['reduction'],
                       task_dict=args['task_dict'], task_weight=args['task_weight'],verbose=2,
                       early_stop_uauc=0.55)
    torch.save(model.state_dict(), f'{MODEL_SAVE_PATH}/npseed{np_rd_seed}_rdseed{rd_seed}_torchseed{torch_seed}')
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__=='__main__':
    global MODEL_SAVE_PATH
    DATA_PATH = '/home/zju/RecSys/dataset/my_feed_data/'
    # TRAIN_DATA_PATH = DATA_PATH + '/training_data/'
    # TRAIN_DATA_PATH = DATA_PATH + '/training_data_With_finish/'
    TRAIN_DATA_PATH = DATA_PATH + '/training_data_finish0.9_history/'
    MODEL_SAVE_PATH = '/home/zju/RecSys/our_feed_rec/exp/model/'
    pretrained_models = {
        'sg_ns_64_epoch30':{
            # 'official_feed': f'{DATA_PATH}/official_feed_emb.d512.pkl',
            # 'feedid': f'{DATA_PATH}/feedid_w7_iter10.64d.filled_cold.pkl',
            # 'authorid': f'{DATA_PATH}/authorid_w7_iter10.64d.filled_cold.pkl',
            # 'userid_by_feed': f'{DATA_PATH}/userid_by_feedid_w10_iter10.64d.pkl',
            'official_feed': f'{DATA_PATH}/official_feed_emb_pca.d64.pkl',
            'feedid': f'{DATA_PATH}/w2v_models_sg_ns_64_epoch30/feedid_w7_iter10.64d.pkl',
            'authorid': f'{DATA_PATH}/w2v_models_sg_ns_64_epoch30/authorid_w7_iter10.64d.pkl',
            'userid_by_feed': f'{DATA_PATH}/w2v_models_sg_ns_64_epoch30/userid_by_feedid_w10_iter10.64d.pkl',
        }
    }

    # USED_FEATURES = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na',
    #                  'videoplayseconds','tag_manu_machine_corr']+\
    #                 ['feed_machine_tag_tfidf_cls_32','feed_machine_kw_tfidf_cls_17',
    #                  'author_machine_tag_tfidf_cls_21','author_machine_kw_tfidf_cls_18']
    USED_FEATURES = ['userid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na',
                     'videoplayseconds','tag_manu_machine_corr']+\
                     ['hist_feedid','hist_authorid','seq_length']+\
                     ['feed_machine_tag_tfidf_cls_32','feed_machine_kw_tfidf_cls_17','author_machine_tag_tfidf_cls_21','author_machine_kw_tfidf_cls_18']
                     

    args = {}
    args['USED_FEATURES'] = USED_FEATURES
    args['DATA_PATH'] = DATA_PATH

    global hidden_units
    hidden_units = (784,512,128)
    # hidden_units = (512,128)
    args['hidden_units'] = hidden_units
    # args['batch_size'] = 40000
    args['batch_size'] = 10000
    # args['emb_dim'] = 128
    args['emb_dim'] = 64
    args['learning_rate'] = 0.02
    args['lr_scheduler'] = True
    args['epochs'] = 3
    args['scheduler_epochs'] = 3
    args['num_warm_epochs'] = 0
    args['scheduler_method'] = 'cos'
    args['use_bn'] = True
    args['reduction'] = 'sum'
    args['optimizer'] = 'adagrad'
    args['num_tasks'] = 8
    args['early_stop_uauc'] = 0.689
    args['num_workers'] = 7
    args['dropout'] =  0.0
    args['l2_reg_dnn'] = 0.001
    args['l2_reg_embedding'] = 0.01
    args['task_dict'] = {
            0: 'read_comment',
            1: 'like',
            2: 'click_avatar',
            3: 'forward',
            4: 'favorite',
            5: 'comment',
            6: 'follow',
            7: 'finish'
    }
    args['task_weight'] = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
            4: 1,
            5: 1,
            6: 1,
            7: 1
    }

    args['pretrained_model'] = pretrained_models['sg_ns_64_epoch30']
    

    ############################################ Set Log！！！！！！！！！！！！！！
    LOG_PATH = os.path.join('/home/zju/RecSys/our_feed_rec/exp/log/', 'debug_AllFeature.log')
    # LOG_PATH = os.path.join('/home/zju/RecSys/our_feed_rec/exp/log/', 'MOE_finish0.9w1_MultiTask56.log')
    if os.path.exists(LOG_PATH) and (not LOG_PATH.endswith('debug.log')):
        print("!!!!!!!!!!!!!!!! {} Already Exists".format(LOG_PATH))
        sys.exit()
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(LOG_PATH)
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info("Parameters: ")
    logging.info(args)

    
    # 全部特征
    linear_feature_columns = pickle.load(open(TRAIN_DATA_PATH+'/linear_feature.pkl','rb'))
    dnn_feature_columns = pickle.load(open(TRAIN_DATA_PATH+'/dnn_feature.pkl','rb'))
    
    # print('raw:')
    # print(dnn_feature_columns)
    # 使用其中部分特征
    linear_feature_columns = [f for f in linear_feature_columns if f.name in USED_FEATURES]
    dnn_feature_columns = [f for f in dnn_feature_columns if f.name in USED_FEATURES]
    # import pdb; pdb.set_trace()
    features = []
    for f in dnn_feature_columns:
        if isinstance(f, SparseFeat):
            features.append(SparseFeat(f.name, f.vocabulary_size, args['emb_dim']))
        elif isinstance(f, VarLenSparseFeat):
            features.append(VarLenSparseFeat(SparseFeat(f.name, f.vocabulary_size,embedding_dim=args['emb_dim']), 40, length_name=f.length_name))
        else:
            features.append(f)
    dnn_feature_columns = features
    
    lbe_dict = preprocess.LBE_MODEL

    train_X = pickle.load(open(TRAIN_DATA_PATH+'/data_train_x.pkl','rb'))
    train_y = pickle.load(open(TRAIN_DATA_PATH+'/data_train_y.pkl','rb'))

    val_X = pickle.load(open(TRAIN_DATA_PATH+'/data_val_x.pkl','rb'))
    val_y = pickle.load(open(TRAIN_DATA_PATH+'/data_val_y.pkl','rb'))

    test_X = pickle.load(open(TRAIN_DATA_PATH+'/data_test_x.pkl','rb'))
    test_y = pickle.load(open(TRAIN_DATA_PATH+'/data_test_y.pkl','rb'))
    
    val_userid_list = val_X['userid'].tolist()
    test_userid_list = test_X['userid'].tolist()

    # 从数据集中选取部分特征
    Seq_Length = [train_X['seq_length'], val_X['seq_length'],test_X['seq_length']]
    train_X = {f.name:train_X[f.name] for f in dnn_feature_columns}
    val_X = {f.name:val_X[f.name] for f in dnn_feature_columns}
    test_X = {f.name:test_X[f.name] for f in dnn_feature_columns}
    train_X['seq_length'] = Seq_Length[0]
    val_X['seq_length'] = Seq_Length[1]
    test_X['seq_length'] = Seq_Length[2]

    train_X['hist_feedid_pre1'] = train_X['hist_feedid']
    val_X['hist_feedid_pre1'] = val_X['hist_feedid']
    test_X['hist_feedid_pre1'] = test_X['hist_feedid']

    train_X['hist_feedid_pre2'] = train_X['hist_feedid']
    val_X['hist_feedid_pre2'] = val_X['hist_feedid']
    test_X['hist_feedid_pre2'] = test_X['hist_feedid']

    # train_X['hist_authorid_pre1'] = train_X['hist_authorid']
    # val_X['hist_authorid_pre1'] = val_X['hist_authorid']
    # test_X['hist_authorid_pre1'] = test_X['hist_authorid']

    for i in dnn_feature_columns:
        if i.name == 'hist_feedid':
            col = i
    dnn_feature_columns.append(VarLenSparseFeat(SparseFeat('hist_feedid_pre1', col.vocabulary_size,embedding_dim=args['emb_dim']), 40, length_name=col.length_name))
    dnn_feature_columns.append(VarLenSparseFeat(SparseFeat('hist_feedid_pre2', col.vocabulary_size,embedding_dim=args['emb_dim']), 40, length_name=col.length_name))
    # for i in dnn_feature_columns:
    #     if i.name == 'hist_authorid':
    #         col = i
    # dnn_feature_columns.append(VarLenSparseFeat(SparseFeat('hist_authorid_pre1', col.vocabulary_size,embedding_dim=args['emb_dim']), 41, length_name=col.length_name))

    args['linear_feature_columns'] = linear_feature_columns
    args['dnn_feature_columns'] = dnn_feature_columns
    
    # 载入label encoder模型
    LBE_MODEL_PATH = f'{DATA_PATH}/label_encoder_models/lbe_dic_all.pkl'
    lbe_dict = pickle.load(open(LBE_MODEL_PATH, 'rb'))
    
    global user_emb_weight, author_emb_weight, feed_emb_weight, official_feed_weight
    # # 载入预训练Embedding weight matrix # 预训练embedding !!!!!!!!!!!!
    user_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['userid'], 
                                                        args['pretrained_model']['userid_by_feed'], padding=True)
    # # user_by_author_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['userid'], 
    # #                                                     args['pretrained_model']['userid_by_author'], padding=True)

    author_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['authorid'], 
                                                        args['pretrained_model']['authorid'], padding=True)
    feed_emb_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
                                                        args['pretrained_model']['feedid'], padding=True)
    # # feed_emb_weight_eges = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
    # #                                                     '../my_data/eges/feedid_eges0_emb.pkl', padding=True)
    official_feed_weight = preprocess.load_feature_pretrained_embedding(lbe_dict['feedid'], 
                                                        args['pretrained_model']['official_feed'], padding=True)

    logging.info('All used features:')
    logging.info(train_X.keys())

    device = 'gpu'
    if device=='gpu' and torch.cuda.is_available():
        # print('cuda ready...')
        device = 'cuda:0'
    else:
        device = 'cpu'

    _model = MOE_DIN(dnn_hidden_units=args['hidden_units'], linear_feature_columns=linear_feature_columns,
              dnn_feature_columns=dnn_feature_columns, history_feature_list = ['feedid'],task='binary', dnn_dropout=0.,
              l2_reg_embedding=0., l2_reg_dnn=0.,
              l2_reg_linear=0., device=device, seed=1233, num_tasks=args['num_tasks'],
              pretrained_user_emb_weight=None,
              pretrained_author_emb_weight=None,
              pretrained_feed_emb_weight=None,
              )

    train_loader = preprocess.get_dataloader(train_X, _model, y=train_y, batch_size=args['batch_size'],num_workers=2,shuffle=True)
    val_x_loader = preprocess.get_dataloader(val_X, _model, batch_size=args['batch_size'],num_workers=2,shuffle=False)
    test_x_loader = preprocess.get_dataloader(test_X, _model, batch_size=args['batch_size'],num_workers=2,shuffle=False)
    del _model
    gc.collect()
    
    # 测试
    train_my_model(args, np_rd_seed=2345, rd_seed=2345, torch_seed=1233)
    for _ in range(3):
        seed1 = random.randint(1, 100000)
        seed2 = random.randint(1, 100000)
        seed3 = random.randint(1, 100000)
        logging.info("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        logging.info("np random seed = " +str(seed1))
        logging.info("random seed = " +str(seed2))
        logging.info("torch random seed = " +str(seed3))
        train_my_model(args, np_rd_seed=seed1, rd_seed=seed2, torch_seed=seed3)
        logging.info("\n")
        logging.info("\n")
    
    
    


