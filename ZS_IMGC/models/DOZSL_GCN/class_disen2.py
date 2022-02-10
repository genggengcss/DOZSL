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

def count(feats, sim_value):
    # compute cosine similarity
    num = np.dot(feats, feats.T)  # 向量点乘
    denom = np.linalg.norm(feats, axis=1).reshape(-1, 1) * np.linalg.norm(feats, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    res = 0.5 + 0.5 * res

    cccc = 0
    for i in range(class_num):
        count = sum(res[i] >= sim_value)

        if count > 1:
            cccc += 1
    print(cccc)


if __name__ == '__main__':

    # datadir = '../../data'
    datadir = '/home/gyx/ZSL2021/ZS_IMGC/data'

    # dataset = 'AwA2'
    dataset = 'ImageNet/ImNet_A'



    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'KG_file')

    embed_path = '/home/gyx/ZSL2021/DisenSemEncoder/data'

    # ImNet-A
    if dataset == 'ImageNet/ImNet_A':
        embed_file = os.path.join(embed_path, 'KG_ImNet_A/DisenKAGT_TransE_mult_K4_D100_ImNet_A',
                                  '2200_2191_ent_embeddings.npy')
        entity_file = os.path.join(embed_path, 'KG_ImNet_A/ent2id.txt')
        namespace = 'ImNet-A:'
        class_num = 80

    if dataset == 'ImageNet/ImNet_O':
        embed_file = os.path.join(embed_path, 'KG_ImNet_O/DisenKAGT_TransE_mult_K5_D100_ImNet_O',
                                  '2000_1753_ent_embeddings.npy')
        entity_file = os.path.join(embed_path, 'KG_ImNet_O/ent2id.txt')
        namespace = 'ImNet-O:'
        class_num = 35

    if dataset == 'AwA2':
        embed_file = os.path.join(embed_path, 'KG_AwA/DisenKAGT_TransE_mult_K5_D100_AwA',
                                  '5200_5125_ent_embeddings.npy')
        entity_file = os.path.join(embed_path, 'KG_AwA/ent2id.txt')
        namespace = 'AwA:'
        class_num = 50

    ent2id = json.load(open(entity_file))


    if dataset == 'AwA2':
        split = json.load(open(os.path.join(DATASET_DIR, 'class.json')))
        seens, unseens = split['seen'], split['unseen']
        seen_wnids, unseen_wnids = list(seens.keys()), list(unseens.keys())
        classes = seen_wnids + unseen_wnids
    else:
        seen_file = os.path.join(DATASET_DIR, 'seen.txt')
        unseen_file = os.path.join(DATASET_DIR, 'unseen.txt')
        seen, unseen = load_class()
        classes = seen + unseen




    embeds = np.load(embed_file)

    feat_list1, feat_list2, feat_list3, feat_list4, feat_list5 = list(), list(), list(), list(), list()
    for cls in classes:
        cls = namespace+ cls
        cls = cls.lower()


        if cls in ent2id:
            # vectors[i] = embeds[ent2id[wnid]][[0,1]].reshape(-1, embed_size)
            vector = embeds[ent2id[cls]]
            feat_list1.append(vector[0])
            feat_list2.append(vector[1])
            feat_list3.append(vector[2])
            feat_list4.append(vector[3])
            if dataset == 'AwA2':
                feat_list5.append(vector[4])
        else:
            print(cls)

    feats1 = np.array(feat_list1)
    feats2 = np.array(feat_list2)
    feats3 = np.array(feat_list3)
    feats4 = np.array(feat_list4)



    sim_value = 0.95
    count(feats1, sim_value)
    count(feats2, sim_value)
    count(feats3, sim_value)
    count(feats4, sim_value)

    if dataset == 'AwA2':
        count(np.array(feat_list5), sim_value)


















