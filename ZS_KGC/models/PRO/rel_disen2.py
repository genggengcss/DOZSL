import json
import os
import numpy as np

DATA_DIR = '/home/gyx/ZSL2021/ZS_KGC/data'
# DATASET = 'NELL'
DATASET = 'Wiki'



def count(feats, sim):
    # compute cosine similarity
    num = np.dot(feats, feats.T)  # 向量点乘

    denom = np.linalg.norm(feats, axis=1).reshape(-1, 1) * np.linalg.norm(feats, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0

    res = 0.5 + 0.5 * res

    cccc = 0
    if DATASET == 'NELL':
        number_rel = 181
    if DATASET == 'Wiki':
        number_rel = 537

    for i in range(number_rel):

        count = sum(res[i] >= sim)

        if count > 1:
            cccc += 1

    print(cccc)
if __name__ == '__main__':

    rel2id = json.load(open(os.path.join(DATA_DIR, DATASET, 'relation2ids')))


    train_tasks = json.load(open(os.path.join(DATA_DIR, DATASET, 'datasplit', 'train_tasks.json')))
    test_tasks = json.load(open(os.path.join(DATA_DIR, DATASET, 'datasplit', "test_tasks.json")))
    val_tasks = json.load(open(os.path.join(DATA_DIR, DATASET, 'datasplit', "dev_tasks.json")))

    seen_rels = sorted(train_tasks.keys())
    val_rels = sorted(val_tasks.keys())
    unseen_rels = sorted(test_tasks.keys())
    all_rels = seen_rels + val_rels + unseen_rels

    # print(len(all_rels))

    # embed_path = '/home/xuyajing/zsl2021/ZSL2021/DisenSemEncoder/data'
    embed_path = '/home/gyx/ZSL2021/DisenSemEncoder/data'


    if DATASET == 'NELL':
        embed_file = os.path.join(embed_path, 'Onto_NELL/DisenE_TransE_K4_D200_NELL',
                                  '6400_6278_ent_embeddings.npy')
        entity_file = os.path.join(embed_path, 'Onto_NELL/ent2id.txt')
    if DATASET == 'Wiki':
        # embed_file = os.path.join(embed_path, 'Onto_Wiki/DisenE_TransE_K4_D200_Wiki',
        #                           '14400_14058_ent_embeddings.npy')
        embed_file = os.path.join(embed_path, 'Onto_Wiki/DisenKAGT_TransE_mult_K4_D200_Wiki',
                                  '4200_4084_ent_embeddings.npy')
        entity_file = os.path.join(embed_path, 'Onto_Wiki/ent2id.txt')

    ent2id = json.load(open(entity_file))

    embeds = np.load(embed_file)
    feat1_list, feat2_list, feat3_list, feat4_list = [], [], [], []
    for rel in all_rels:
        if DATASET == 'NELL':
            rel = rel.replace('concept:', 'NELL:')
        if DATASET == 'Wiki':
            rel = 'Wikidata:' + rel
        rel = rel.lower()
        if rel in ent2id:
            vector = embeds[ent2id[rel]]
            feat1_list.append(vector[0])
            feat2_list.append(vector[1])
            feat3_list.append(vector[2])
            feat4_list.append(vector[3])
        else:
            print(rel)



    feats1 = np.array(feat1_list)
    feats2 = np.array(feat2_list)

    feats3 = np.array(feat3_list)

    feats4 = np.array(feat4_list)

    sim_value = 0.97
    count(feats1, sim_value)
    count(feats2, sim_value)
    count(feats3, sim_value)
    count(feats4, sim_value)


