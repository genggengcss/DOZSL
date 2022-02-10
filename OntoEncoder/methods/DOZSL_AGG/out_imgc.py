import os
import json
import sys
import pickle as pkl
import numpy as np
import scipy.io as scio




# AwA dataset split (id & name)

train_wnid = ["n02071294", "n02363005", "n02110341", "n02123394", "n02106662", "n02123597", "n02445715", "n01889520", "n02129604", "n02398521", "n02128385", "n02493793", "n02503517", "n02480855", "n02403003", "n02481823", "n02342885", "n02118333", "n02355227", "n02324045", "n02114100", "n02085620", "n02441942", "n02444819", "n02410702", "n02391049", "n02510455", "n02395406", "n02129165", "n02134084", "n02106030", "n02403454", "n02430045", "n02330245", "n02065726", "n02419796", "n02132580", "n02391994", "n02508021", "n02432983"]
test_wnid = ["n02411705", "n02068974", "n02139199", "n02076196", "n02064816", "n02331046", "n02374451", "n02081571", "n02439033", "n02127482"]

train_name = ["killer+whale", "beaver", "dalmatian", "persian+cat", "german+shepherd", "siamese+cat", "skunk", "mole", "tiger", "hippopotamus", "leopard", "spider+monkey", "elephant", "gorilla", "ox", "chimpanzee", "hamster", "fox", "squirrel", "rabbit", "wolf", "chihuahua", "weasel", "otter", "buffalo", "zebra", "giant+panda", "pig", "lion", "polar+bear", "collie", "cow", "deer", "mouse", "humpback+whale", "antelope", "grizzly+bear", "rhinoceros", "raccoon", "moose"]
test_name = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat", "horse", "walrus", "giraffe", "bobcat"]





def readTxt(file_name):
    class_list = list()
    wnids = open(file_name, 'rU')
    try:
        for line in wnids:
            line = line[:-1]
            class_list.append(line)
    finally:
        wnids.close()
    print(len(class_list))
    return class_list

def load_class():
    seen = readTxt(seen_file)
    unseen = readTxt(unseen_file)
    return seen, unseen

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


def save_embed_awa(filename, wnids, names):


    # load embeddings
    embeds = np.load(filename)
    # save to .mat file
    matcontent = scio.loadmat(os.path.join(DATASET_DIR, 'att_splits.mat'))
    all_names = matcontent['allclasses_names'].squeeze().tolist()

    # embed_size = embeds.shape[1]
    vectors = np.zeros((len(all_names), embed_size), dtype=np.float)
    for i in range(len(all_names)):
        name = all_names[i][0]
        wnid = wnids[names.index(name)]
        wnid = 'AwA:' + wnid
        wnid = wnid.lower()
        if wnid in ent2id:
            vector = embeds[ent2id[wnid]]
            # print(vector.shape)

            vectors[i] = embeds[ent2id[wnid]].reshape(-1, embed_size)
            # vectors[i] = embeds[ent2id[wnid]][[0, 1]].reshape(-1, embed_size)
        else:
            print('not found:', wnid)
        # vectors[i] = embeds[ent2id[wnid]].reshape(-1, embed_size)

    print(vectors.shape)

    embed_file = os.path.join(DATA_DIR, 'embeddings', 'kge_DisenKGAT', save_file)
    scio.savemat(embed_file, {'embeddings': vectors})

def save_embed(filename, classes):

    # load embeddings
    embeds = np.load(filename)
    # save to .mat file
    matcontent = scio.loadmat(os.path.join(datadir, 'ImageNet', 'w2v.mat'))
    wnids = matcontent['wnids'].squeeze().tolist()
    wnids = wnids[:2549]

    vectors = np.zeros((len(wnids), embed_size), dtype=np.float)

    print(vectors.shape)
    for i, wnid in enumerate(wnids):
        wnid = wnid[0]
        if wnid in classes:
            if dataset == 'ImNet_A':
                wnid = 'ImNet-A:'+wnid
            if dataset == 'ImNet_O':
                wnid = 'ImNet-O:' + wnid
            wnid = wnid.lower()
            if wnid in ent2id:
                vectors[i] = embeds[ent2id[wnid]].reshape(-1, embed_size)
                # vectors[i] = embeds[ent2id[wnid]][[0, 1]].reshape(-1, embed_size)
            else:
                print('not found:', wnid)
        else:
            continue
    # save wnids together
    wnids_cell = np.empty((len(wnids), 1), dtype=np.object)
    for i in range(len(wnids)):
        wnids_cell[i][0] = np.array(wnids[i])

    embed_file = os.path.join(DATA_DIR, 'embeddings', 'kge_DisenKGAT', save_file)
    scio.savemat(embed_file, {'embeddings': vectors, 'wnids': wnids_cell})




if __name__ == '__main__':

    datadir = '/home/gyx/ZSL2021/ZS_IMGC/data'

    dataset = 'AwA2'
    # dataset = 'ImNet_O'

    if dataset == 'AwA2':
        DATASET_DIR = os.path.join(datadir, dataset)
        embed_path = os.path.join('../../data/', 'KG_AwA')
    else:
        DATASET_DIR = os.path.join(datadir, 'ImageNet', dataset)
        embed_path = os.path.join('../../data/', 'KG_' + dataset)

    DATA_DIR = os.path.join(DATASET_DIR, 'KG_file')



    # load entity dict
    entity_file = os.path.join(embed_path, 'ent2id.txt')
    ent2id = json.load(open(entity_file))


    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K4_D100_ImNet_A', '2400_2353_ent_embeddings.npy')
    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K4_D100_ImNet_A', '2200_2191_ent_embeddings.npy')
    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K5_D100_ImNet_A', '2200_2010_ent_embeddings.npy')

    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K4_D100_ImNet_O', '2000_1868_ent_embeddings.npy')
    # embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K5_D100_ImNet_O', '2000_1753_ent_embeddings.npy')

    embed_file = os.path.join(embed_path, 'DisenKAGT_TransE_mult_K5_D100_AwA', '5000_4957_ent_embeddings.npy')

    embed_size = 500  # K*D

    save_file = 'DisenKGAT_mult_K5_D100_5000_4957.mat'

    if dataset == 'AwA2':
        class_file = os.path.join(DATASET_DIR, 'class.json')
        classes = json.load(open(class_file, 'r'))
        wnids = list()
        names = list()
        for wnid, name in classes['seen'].items():
            wnids.append(wnid)
            names.append(name)
        for wnid, name in classes['unseen'].items():
            wnids.append(wnid)
            names.append(name)

        save_embed_awa(embed_file, wnids, names)

    else:
        seen_file = os.path.join(DATASET_DIR, 'seen.txt')
        unseen_file = os.path.join(DATASET_DIR, 'unseen.txt')
        seen, unseen = load_class()
        classes = seen + unseen

        save_embed(embed_file, classes)












