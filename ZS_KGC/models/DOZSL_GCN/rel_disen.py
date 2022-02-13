import json
import os
import numpy as np

DATA_DIR = '/home/gyx/ZSL2021/ZS_KGC/data'
DATASET = 'Wiki'


if __name__ == '__main__':

    if DATASET == 'NELL':
        embed_file = 'rela_matrix_rdfs_55000.npz'
        # embed_file = 'kge_DisenKGAT/RGAT_K1_D200_3400_3360.npz'

    if DATASET == 'Wiki':
        # embed_file = 'rela_matrix_rdfs_65000.npz'
        embed_file = 'kge_DisenKGAT/RGAT_K1_D200_6400_6357.npz'

    rel2id = json.load(open(os.path.join(DATA_DIR, DATASET, 'relation2ids')))

    rela_matrix = np.load(os.path.join(DATA_DIR, DATASET, 'KG_file', 'embeddings', embed_file))['relaM']

    train_tasks = json.load(open(os.path.join(DATA_DIR, DATASET, 'datasplit', 'train_tasks.json')))
    test_tasks = json.load(open(os.path.join(DATA_DIR, DATASET, 'datasplit', "test_tasks.json")))
    val_tasks = json.load(open(os.path.join(DATA_DIR, DATASET, 'datasplit', "dev_tasks.json")))

    seen_rels = sorted(train_tasks.keys())
    val_rels = sorted(val_tasks.keys())
    unseen_rels = sorted(test_tasks.keys())
    all_rels = seen_rels + val_rels + unseen_rels


    feat_list = []
    for rel in all_rels:

        if rel in rel2id:
            vector = rela_matrix[rel2id[rel]]
            feat_list.append(vector)

        else:
            print(rel)

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
    for i in range(537):

        count = sum(res[i] >= 0.98)

        if count > 1:
            cccc += 1

    print(cccc)