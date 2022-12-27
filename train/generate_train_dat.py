# encoding: utf-8 
import os, sys, gc, pickle
sys.path.append('../')
from model.moe import MOE
import preprocess

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from tqdm import tqdm
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names

import logging
logger = logging.getLogger(__name__)

LBE_MODEL_PATH = '/home/zju/RecSys/dataset/my_feed_data/label_encoder_models/lbe_dic_all.pkl'


def generate_label_encode_model(df_all, sparse_col, out_path=LBE_MODEL_PATH):
    if os.path.exists(out_path):
        print('!!!!!!!!file already exists!!!!!!!!!! using exists: ', out_path)
        LBE_MODEL = pickle.load(open(out_path, 'rb'))
        return LBE_MODEL
    LBE_MODEL={}
    for fea in sparse_col:
        fea_cls = list(set(df_all[fea].tolist()))
        fea_cls.sort()
        lbe = LabelEncoder()
        lbe.fit(fea_cls)
        LBE_MODEL[fea] = lbe
        # import pdb; pdb.set_trace()
    pickle.dump(LBE_MODEL, open(out_path, 'wb'))
    return LBE_MODEL
#  test_df 是需转换的；test_df_1 是转换后的
#  值是转置后的， 列 变 索引，索引变列

    
def process_pipe(feed_path, user_act_path, used_columns, used_sparse_cols, used_dense_cols,
                 emb_dim=16, is_training=True, test_data=False):
    data, history_cols_dict = preprocess.preprocess(feed_path, user_act_path)
    used_columns +=( history_cols_dict['feedid'] + history_cols_dict['feedid_neg'] + history_cols_dict['authorid'] +['seq_length','seq_length_neg'] )

    generate_label_encode_model(data, used_sparse_cols, LBE_MODEL_PATH)

    data_ds = preprocess.down_sample(data, used_columns, sample_method=None, 
                          neg2pos_ratio=300, user_samp='random', 
                          by_date=None, is_training=is_training)
    
    train_data = data_ds.query('date_<=12')
    val_data = data_ds.query('date_==13')
    test_data = data_ds.query('date_==14')

    X_dic_train, y_arr_train, linear_feats, dnn_feats, lbe_dict = preprocess.process_features(
                        train_data, used_sparse_cols, used_dense_cols, history_cols_dict,
                        actions=ACTIONS, emb_dim=emb_dim, use_tag_text=None, use_kw_text=None, 
                        feed_history=None, author_history=None,  use_din=False, 
                        max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                        )
    X_dic_val, y_arr_val, linear_feats, dnn_feats, lbe_dict = preprocess.process_features(
                        val_data, used_sparse_cols, used_dense_cols, history_cols_dict,
                        actions=ACTIONS, emb_dim=emb_dim, use_tag_text=None, use_kw_text=None, 
                        feed_history=None, author_history=None,  use_din=False, 
                        max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                        )
    # import pdb; pdb.set_trace()
    X_dic_test, y_arr_test, linear_feats, dnn_feats, lbe_dict = preprocess.process_features(
                        test_data, used_sparse_cols, used_dense_cols, history_cols_dict,
                        actions=ACTIONS, emb_dim=emb_dim, use_tag_text=None, use_kw_text=None, 
                        feed_history=None, author_history=None,  use_din=False, 
                        max_seq_length=128, behavior_feature_list=['feedid','authorid'],
                        )

    return [(X_dic_train, y_arr_train, linear_feats, dnn_feats, lbe_dict),
            (X_dic_val, y_arr_val, linear_feats, dnn_feats, lbe_dict),
            (X_dic_test, y_arr_test, linear_feats, dnn_feats, lbe_dict)]


if __name__=='__main__':
    CLS_COLS = ['feed_manu_tag_tfidf_cls_32', 'feed_machine_tag_tfidf_cls_32', 'feed_manu_kw_tfidf_cls_22', 
                'feed_machine_kw_tfidf_cls_17', 'feed_description_tfidf_cls_18', 'author_manu_tag_tfidf_cls_19', 
                'author_machine_tag_tfidf_cls_21', 'author_manu_kw_tfidf_cls_18', 'author_machine_kw_tfidf_cls_18', 
                'author_description_tfidf_cls_18']

    TOPIC_COLS = ['feed_manu_tag_topic_class', 'feed_machine_tag_topic_class', 'feed_manu_kw_topic_class', 
                  'feed_machine_kw_topic_class', 'feed_description_topic_class', 'author_description_topic_class', 
                  'author_manu_kw_topic_class', 'author_machine_kw_topic_class', 'author_manu_tag_topic_class', 
                  'author_machine_tag_topic_class']

    SPARSE_COLS = ['userid','feedid','authorid','bgm_song_id','bgm_singer_id','videoplayseconds_bin','bgm_na']+\
        CLS_COLS+TOPIC_COLS
    DENSE_COLS = ['videoplayseconds','tag_manu_machine_corr']
    ACTIONS = ["read_comment","like","click_avatar","forward",'favorite','comment','follow','finish']


    USED_COLUMNS = SPARSE_COLS + DENSE_COLS + ACTIONS

    DATA_PATH = '/home/zju/RecSys/dataset/my_feed_data/'
    # OUTPATH = DATA_PATH+'/training_data/'
    # OUTPATH = DATA_PATH+'/training_data_finish0.9_history/'
    OUTPATH = DATA_PATH+'/training_data_finish0.9_hist2/'
    
    if not os.path.exists(OUTPATH):
        os.makedirs(OUTPATH)

    # user_act_path = '/home/zju/RecSys/dataset/wechat_data_raw/user_action.csv'
    user_act_path = '/home/zju/RecSys/dataset/my_feed_data/my_user_action_finish0.9.csv' # With Finish 1.0
    feed_path = DATA_PATH + '/feedid_text_features/feed_author_text_features_fillna_by_author_clusters.pkl'
    
    data_train, data_val, data_test = process_pipe(
        feed_path, user_act_path, USED_COLUMNS, SPARSE_COLS, DENSE_COLS)

    pickle.dump(data_train[2], open(f'{OUTPATH}/linear_feature.pkl','wb'))
    pickle.dump(data_train[3], open(f'{OUTPATH}/dnn_feature.pkl', 'wb'))
    # import pdb; pdb.set_trace()

    pickle.dump(data_train[0], open(f'{OUTPATH}/data_train_x.pkl', 'wb'))
    pickle.dump(data_train[1], open(f'{OUTPATH}/data_train_y.pkl', 'wb'))

    pickle.dump(data_val[0], open(f'{OUTPATH}/data_val_x.pkl', 'wb'))
    pickle.dump(data_val[1], open(f'{OUTPATH}/data_val_y.pkl', 'wb'))

    pickle.dump(data_test[0], open(f'{OUTPATH}/data_test_x.pkl', 'wb'))
    pickle.dump(data_test[1], open(f'{OUTPATH}/data_test_y.pkl', 'wb'))

    