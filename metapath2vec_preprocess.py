import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from torch import nn
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
def my_KNN(x, y, k=5, split_list=[0.2,0.4,0.6,0.8], time=10, show_train=True, shuffle=True):
    x = np.array(x)
    x = np.squeeze(x)
    y = np.array(y)
    #f = open("lambdalog", "a")
    if len(y.shape) > 1:
        y = np.argmax(y, axis=1)
    for split in split_list:
        ss = split
        split = int(x.shape[0] * split)
        micro_list = []
        macro_list = []
        if time:
            for i in range(time):
                if shuffle:
                    permutation = np.random.permutation(x.shape[0])
                    x = x[permutation, :]
                    y = y[permutation]
                # x_true = np.array(x_true)
                train_x = x[:split, :]
                test_x = x[split:, :]

                train_y = y[:split]
                test_y = y[split:]

                #estimator = KNeighborsClassifier(n_neighbors=k)
                estimator = LogisticRegression()
                estimator.fit(train_x, train_y)
                y_pred = estimator.predict(test_x)
                f1_macro = f1_score(test_y, y_pred, average='macro')
                f1_micro = f1_score(test_y, y_pred, average='micro')
                macro_list.append(f1_macro)
                micro_list.append(f1_micro)
            msg='KNN({}avg, split:{}, k={}) f1_macro: {:.4f}, f1_micro: {:.4f}'.format(
                time, ss, k, sum(macro_list) / len(macro_list), sum(micro_list) / len(micro_list))
            print(msg)
            #f.write(msg+'\n')
            print("macro_std",np.std(macro_list),"micro_std",np.std(micro_list))
def load_dblp_raw():
    # url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/LabDBLP.mat'
    fea_path = get_download_dir() + '/DBLP_feat'
    # download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    labels = data['Aut_lab'][:, 1] - 1
    selected_aut_id = data['Aut_lab'][:, 0]
    id2index = {v: k for k, v in enumerate(data['AutID'][:, 0])}
    selected_aut_index = np.array([id2index[id] for id in selected_aut_id])
    selected_paper_index = np.where(data['PA'][:, selected_aut_index].sum(1))[0]
    selected_term_index = np.where(data['PT'][selected_paper_index, :].sum(0))[1]

    p_vs_a = data['PA'][selected_paper_index, :][:, selected_aut_index]  # paper-author
    p_vs_t = data['PT'][selected_paper_index, :][:, selected_term_index]  # paper-term, bag of words
    p_vs_c = data['PC'][selected_paper_index, :]  # paper-conference, labels come from that
    p_num=p_vs_a.shape[0]
    a_num=p_vs_a.shape[1]
    t_num=p_vs_t.shape[1]
    c_num=p_vs_c.shape[1]
    f_1=open("./mp2v_data/dblp_paper","w")
    f_2=open("./mp2v_data/dblp_author","w")
    f_3=open("./mp2v_data/dblp_conf","w")
    f_4=open("./mp2v_data/dblp_paper_author","w")
    f_5=open("./mp2v_data/dblp_paper_conf","w")
    pa=p_vs_a.tocoo()
    pa_row_col=[pa.row,[i+p_num for i in  pa.col]]
    ##pac
    #p
    #a
    #c
    for i in range(p_num):
        line=str(i)+' p'+str(i)+'\n'
        f_1.write(line)
    for i in range(p_num,p_num+a_num):
        line = str(i) + ' a' + str(i) + '\n'
        f_2.write(line)
    for i in range(p_num+a_num, p_num + a_num+c_num):
        line = str(i) + ' v' + str(i) + '\n'
        f_3.write(line)
    for i in range(len(pa_row_col[0])):
        line1=str(pa_row_col[0][i])+' '+str(pa_row_col[1][i])+'\n'
        f_4.write(line1)

    pc=p_vs_c.tocoo()
    pc_row_col=[pc.row,[i+p_num+a_num for i in pc.col]]
    for i in range(len(pc_row_col[0])):
        line1=str(pc_row_col[0][i])+' '+str(pc_row_col[1][i])+'\n'
        f_5.write(line1)
    f_1.close()
    f_2.close()
    f_3.close()
    f_4.close()
    f_5.close()
def load_acm_raw():
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field?
    p_vs_a = data['PvsA']  # paper-author
    p_vs_t = data['PvsT']  # paper-term, bag of words
    p_vs_c = data['PvsC']  # paper-conference, labels come from that
    p_vs_p = data['PvsP']
    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]
    p_num=p_vs_a.shape[0]
    a_num=p_vs_a.shape[1]
    l_num=p_vs_l.shape[1]
    f_1 = open("./mp2v_data/acm_paper", "w")
    f_2 = open("./mp2v_data/acm_author", "w")
    f_3 = open("./mp2v_data/acm_field", "w")
    f_4 = open("./mp2v_data/acm_paper_author", "w")
    f_5 = open("./mp2v_data/acm_paper_field", "w")
    pa = p_vs_a.tocoo()
    pa_row_col = [pa.row, [i + p_num for i in pa.col]]
    ##pal
    # p
    # a
    # l
    for i in range(p_num):
        line = str(i) + ' i' + str(i) + '\n'
        f_1.write(line)
    for i in range(p_num, p_num + a_num):
        line = str(i) + ' a' + str(i) + '\n'
        f_2.write(line)
    for i in range(p_num + a_num, p_num + a_num + l_num):
        line = str(i) + ' f' + str(i) + '\n'
        f_3.write(line)
    for i in range(len(pa_row_col[0])):
        line1 = str(pa_row_col[0][i]) + ' ' + str(pa_row_col[1][i]) + '\n'
        f_4.write(line1)

    pl = p_vs_l.tocoo()
    pl_row_col = [pl.row, [i + p_num + a_num for i in pl.col]]
    for i in range(len(pl_row_col[0])):
        line1 = str(pl_row_col[0][i]) + ' ' + str(pl_row_col[1][i]) + '\n'
        f_5.write(line1)
    f_1.close()
    f_2.close()
    f_3.close()
    f_4.close()
    f_5.close()
def load_imdb_raw():
    # url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/imdb_3_class.pkl'
    # download(_get_dgl_url(url), path=data_path)
    f = open(data_path, mode="rb")
    data = pickle.load(f)

    m_vs_d = data['md']  # movie-director
    m_vs_a = data['ma']  # movie-actor

    m_num=m_vs_d.shape[0]
    d_num=m_vs_d.shape[1]
    a_num=m_vs_a.shape[1]
    f_1 = open("./mp2v_data/imdb_movie", "w")
    f_2 = open("./mp2v_data/imdb_director", "w")
    f_4 = open("./mp2v_data/imdb_movie_director", "w")
    f_3 = open("./mp2v_data/imdb_movie_actor", "w")
    md = m_vs_d.tocoo()
    ma = m_vs_a.tocoo()
    md_row_col = [md.row, [i + m_num for i in md.col]]
    ma_row_col = [ma.row, [i + m_num+d_num for i in ma.col]]
    ##mda
    # m
    # d
    # a
    for i in range(m_num):
        line = str(i) + ' i' + str(i) + '\n'
        f_1.write(line)
    for i in range(m_num, m_num+ d_num):
        line = str(i) + ' a' + str(i) + '\n'
        f_2.write(line)
    for i in range(len(ma_row_col[0])):
        line1 = str(ma_row_col[0][i]) + ' ' + str(ma_row_col[1][i]) + '\n'
        f_3.write(line1)
    for i in range(len(md_row_col[0])):
        line1 = str(md_row_col[0][i]) + ' ' + str(md_row_col[1][i]) + '\n'
        f_4.write(line1)
    f_1.close()
    f_2.close()
    f_4.close()
def eval_dblp():
    f = open("/home/xuyou/dgl/examples/pytorch/han/mp2v_data/dblp_output_emb40.txt", "r")
    data_path = get_download_dir() + '/LabDBLP.mat'
    a_num = 4057
    p_num = 14328
    c_num = 20
    t_num = 8898
    labels_arr = np.zeros((4057, 4))
    features = np.zeros((4057, 64))
    data = sio.loadmat(data_path)
    labels = data['Aut_lab'][:, 1] - 1
    for line in f:
        nums = line.split()
        if len(nums) == 2:
            continue
        if nums[0][0]!='a':
            continue
        id = int(nums[0][1:])
        fea = [float(i) for i in nums[1:]]
        if id not in range(p_num, p_num + a_num):
            continue
        features[id - p_num] = fea
        lab = labels[id - p_num]
        labels_arr[id - p_num][lab] = 1
    my_KNN(features[:2500], labels_arr[:2500])
def eval_dblp_mg2v():
    f = open("/home/xuyou/deepwalk/MetaGraph2Vec/RandomWalk2Vec/dblp_g2v", "r")
    data_path = get_download_dir() + '/LabDBLP.mat'
    a_num = 4057
    p_num = 14328
    c_num = 20
    t_num = 8898
    labels_arr = np.zeros((4057, 4))
    features = np.zeros((4057, 64))
    data = sio.loadmat(data_path)
    labels = data['Aut_lab'][:, 1] - 1
    for line in f:
        nums = line.split()
        if len(nums) == 2:
            continue
        if nums[0][0]!='a':
            continue
        id = int(nums[0][1:])
        fea = [float(i) for i in nums[1:]]
        if id not in range(p_num, p_num + a_num):
            continue
        features[id - p_num] = fea
        lab = labels[id - p_num]
        labels_arr[id - p_num][lab] = 1
    my_KNN(features[:2500], labels_arr[:2500])
def eval_acm():
    f=open("/home/xuyou/dgl/examples/pytorch/han/mp2v_data/acm_output_emb40.txt","r")
    data_path = get_download_dir() + '/ACM.mat'
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field?
    p_vs_a = data['PvsA']  # paper-author
    p_vs_t = data['PvsT']  # paper-term, bag of words
    p_vs_c = data['PvsC']
    p_vs_c_filter = p_vs_c[:, conf_ids]

    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]

    p_vs_c = p_vs_c[p_selected]
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    p_num=4025
    a_num=17431
    features=np.zeros((p_num,64))
    labels_arr=np.zeros((p_num,3))
    for line in f:
        nums=line.split()
        if len(nums)==2:
            continue
        if nums[0][0]!='i':
            continue
        id=int(nums[0][1:])
        fea=[float(i) for i in nums[1:]]
        if id not in range(0,p_num):
            continue
        features[id]=fea
        lab=labels[id]
        labels_arr[id][lab]=1
    my_KNN(features[:2500],labels_arr[:2500])
def eval_imdb():
    m_num=4183
    a_num=5084
    d_num=2004
    data_path = get_download_dir() + '/imdb_3_class.pkl'
    # download(_get_dgl_url(url), path=data_path)
    f = open(data_path, mode="rb")
    data = pickle.load(f)
    labels = np.array(data['labels'])
    features = np.zeros((m_num, 64))
    labels_arr = np.zeros((m_num, 3))
    ff=open("/home/xuyou/dgl/examples/pytorch/han/mp2v_data/acm_output_emb40.txt","r")
    for line in ff:
        nums = line.split()
        if len(nums) == 2 or nums[0][0]!='i':
            continue
        id = int(nums[0][1:])
        fea = [float(i) for i in nums[1:]]
        if id not in range(0, m_num):
            continue
        features[id] = fea
        lab = labels[id]
        labels_arr[id][lab] = 1
    ids=list(range(m_num))
    np.random.shuffle(ids)
    my_KNN(features[ids[:2500]], labels_arr[ids[:2500]])
def eval_acm_mg2v():
    f=open("/home/xuyou/deepwalk/MetaGraph2Vec/RandomWalk2Vec/acm_g2v","r")
    data_path = get_download_dir() + '/ACM.mat'
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]
    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field?
    p_vs_a = data['PvsA']  # paper-author
    p_vs_t = data['PvsT']  # paper-term, bag of words
    p_vs_c = data['PvsC']
    p_vs_c_filter = p_vs_c[:, conf_ids]

    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]

    p_vs_c = p_vs_c[p_selected]
    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    p_num=4025
    a_num=17431
    features=np.zeros((p_num,64))
    labels_arr=np.zeros((p_num,3))
    for line in f:
        nums=line.split()
        if len(nums)==2:
            continue
        if nums[0][0]!='p':
            continue
        id=int(nums[0][1:])
        fea=[float(i) for i in nums[1:]]
        if id not in range(0,p_num):
            continue
        features[id]=fea
        lab=labels[id]
        labels_arr[id][lab]=1
    my_KNN(features[:2500],labels_arr[:2500])
def eval_imdb_mg2v():
    m_num=4183
    a_num=5084
    d_num=2004
    data_path = get_download_dir() + '/imdb_3_class.pkl'
    # download(_get_dgl_url(url), path=data_path)
    f = open(data_path, mode="rb")
    data = pickle.load(f)
    labels = np.array(data['labels'])
    features = np.zeros((m_num, 64))
    labels_arr = np.zeros((m_num, 3))
    ff=open("/home/xuyou/deepwalk/MetaGraph2Vec/RandomWalk2Vec/imdb_g2v","r")
    for line in ff:
        nums = line.split()
        if len(nums) == 2 or nums[0][0]!='p':
            continue
        id = int(nums[0][1:])
        fea = [float(i) for i in nums[1:]]
        if id not in range(0, m_num):
            continue
        features[id] = fea
        lab = labels[id]
        labels_arr[id][lab] = 1
    ids=list(range(m_num))
    np.random.shuffle(ids)
    my_KNN(features[ids[:2500]], labels_arr[ids[:2500]])
if __name__=="__main__":
    #load_dblp_raw()
    #load_acm_raw()
    #load_imdb_raw()
    #eval_dblp()
    #eval_acm()
    #eval_imdb()
    #eval_dblp_mg2v()
    #eval_acm_mg2v()
    eval_imdb_mg2v()