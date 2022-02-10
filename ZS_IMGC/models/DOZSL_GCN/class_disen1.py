import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio







def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            class_list.append(line)
    finally:
        wnids.close()
    # print(len(class_list))
    return class_list

def load_class():
    seen = readTxt(seen_file)
    unseen = readTxt(unseen_file)
    return seen, unseen

###########################


def softmax(x):
    max = np.max(x, axis=1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

if __name__ == '__main__':

    # datadir = '../../data'
    datadir = '/home/gyx/ZSL2021/ZS_IMGC/data'

    # dataset = 'AwA2'
    dataset = 'ImageNet/ImNet_A'



    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'KG_file')


    seen_file = os.path.join(DATASET_DIR, 'seen.txt')
    unseen_file = os.path.join(DATASET_DIR, 'unseen.txt')
    seen, unseen = load_class()
    classes = seen + unseen

    embed_file = os.path.join(datadir, dataset, 'KG_file', 'embeddings', 'kge_H_A_65000.mat')

    matcontent = scio.loadmat(embed_file)
    embed = matcontent['embeddings']
    nodes = matcontent['wnids']

    nodes = [node.tolist()[0][0] for node in nodes]
    feat_list = list()
    for cls in classes:
        if cls in nodes:
            feat = embed[nodes.index(cls)]
            feat_list.append(feat)
        else:
            print(cls)

    feats = np.array(feat_list)



    # compute cosine similarity
    num = np.dot(feats, feats.T)  # 向量点乘


    denom = np.linalg.norm(feats, axis=1).reshape(-1, 1) * np.linalg.norm(feats, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0

    res = 0.5 + 0.5 * res

    # row, col = np.diag_indices_from(res)
    # res[row, col] = 0

    # print(res)

    # res = softmax(res)

    # print(res[0])

    # sim_matrix = res>=0.95
    # print(sim_matrix.astype(int))
    # print(sim_matrix[0])

    cccc = 0
    for i in range(80):
        count = sum(res[i] >= 0.85)
        if count > 1:
            cccc += 1
    print(cccc)













