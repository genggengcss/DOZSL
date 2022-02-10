import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio


from collections import defaultdict

import torch
from torch.nn.init import xavier_normal_
from torch.nn import Parameter



def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    xavier_normal_(param.data)
    return param


###########################

def loadDict(file_name):
    entities = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            index, cls = line.split('\t')
            entities.append(cls)
    finally:
        wnids.close()
    print(len(entities))
    return entities



if __name__ == '__main__':


    datadir = '/home/gyx/ZSL2021/ZS_KGC/data'
    #
    # dataset = 'NELL'
    dataset = 'Wiki'

    DATASET_DIR = os.path.join(datadir, dataset)
    DATA_DIR = os.path.join(datadir, dataset, 'KG_file')

    embed_path = os.path.join('../../data/', 'Onto_' + dataset)

    # load entity dict
    entity_file = os.path.join(embed_path, 'ent2id.txt')
    ent2id = json.load(open(entity_file))

    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K8_D200_NELL', '2600_2559_ent_embeddings.npy')
    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K8_D200_NELL', '2400_2398_ent_embeddings.npy')

    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K8_D200_g9_NELL', '2000_1802_ent_embeddings.npy')

    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K9_D200_NELL', '2000_1911_ent_embeddings.npy')


    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K8_D200_Wiki', '2600_2536_ent_embeddings.npy')
    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K8_D200_g9_Wiki', '2800_2799_ent_embeddings.npy')

    embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K9_D200_Wiki', '2400_2253_ent_embeddings.npy')
    if dataset == 'NELL':
        rel2id = json.load(open(os.path.join(DATASET_DIR, 'relation2ids')))
    if dataset == 'Wiki':
        rel2id = json.load(open(os.path.join(DATASET_DIR, 'relation2ids_1')))
    id2rel = {v: k for k, v in rel2id.items()}
    id2rel = {k: id2rel[k] for k in sorted(id2rel.keys())}


    w_dim = 1800
    all_feats = np.zeros((len(rel2id.keys()), w_dim), dtype=np.float)
    ent_embeds = np.load(embed_file)

    print(ent_embeds.shape)
    for i, rel in id2rel.items():
        if dataset == 'NELL':
            rel = rel.replace('concept:', 'NELL:')
        if dataset == 'Wiki':
            rel = 'Wikidata:' + rel

        rel = rel.lower()
        # load embeddings
        if rel in ent2id:
            rel_embed = ent_embeds[ent2id[rel]].astype('float32')
            all_feats[i] = rel_embed.reshape(-1, w_dim)
            # all_feats[i] = rel_embed[[1, 2]].reshape(-1, w_dim)
        else:
            print('not found:', rel)


    all_feats = np.array(all_feats)  # 229, 51

    save_name = 'DisenKGAT_K9_D200_2400_2253.npz'
    np.savez(os.path.join(DATA_DIR, 'embeddings', 'kge_DisenKGAT', save_name), relaM=all_feats)









