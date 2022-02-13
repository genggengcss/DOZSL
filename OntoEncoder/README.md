# DOZSL
Code and Data for the paper: "Disentangled Ontology Embedding for Zero-shot Learning".
In this work, we focus on ontologies for augmenting ZSL, and propose to learn disentangled ontology embeddings to capture and utilize more fine-grained class relationships in different aspects.
We also contribute a new ZSL framework named DOZSL, which contains two new ZSL solutions based on generative models and graph propagation models, respectively,
for effectively utilizing the disentangled ontology embeddings for the zero-shot learning problems in image classification (i.e., ZS-IMGC) and KG completion (i.e., ZS-KGC).

### Requirements
- `python 3.5`
- `PyTorch >= 1.5.0`

### Dataset Preparation

#### AwA2
Download public pre-trained image features and dataset split for [AwA2](http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip), uncompress it and put the files in **AWA2** folder to our folder `ZS_IMGC/data/AwA2/`.


#### ImageNet (ImNet-A, ImNet-O)

Download pre-trained image features of ImageNet classes as well as their splits from [here](https://drive.google.com/drive/folders/1An6nLXRRvlKSCbJoKKlqTNDvgN7PyvvW?usp=sharing) and put them to the folder `ZS_IMGC/data/ImageNet/`.


#### NELL-ZS & Wiki-ZS
Download from [here](https://github.com/Panda0406/Zero-shot-knowledge-graph-relational-learning) and put them to the corresponding data folder.

### Basic Training and Testing

The first thing you need to do is to train the ontology encoder using the code in the folder `OntoEncoder`, you can get more details at [OntoEncoder/README.md](OntoEncoder/README.md).

Secondly, with well-trained ontology embedding, you can take it as the input of the generative model (see codes in the folder `ZS_IMGC/models/DOZSL_GAN` or `ZS_KGC/models/DOZSL_GAN` for ZS-IMGC and ZS-KGC tasks, respectively) or the graph propagation model (see codes in the folder `ZS_IMGC/models/DOZSL_GCN` or `ZS_KGC/models/DOZSL_GCN` for ZS-IMGC and ZS-KGC tasks, respectively).

*Note: you can skip the first step if you just want to use the ontology embedding we learned, the files are provided in the corresponding directories*.


#### Ontology Encoder

### Steps:

1.  Running the scripts in each method folder to pre-train the KGs and ontological schemas

2. Selecting the targte class embeddings from the trained concept embeddings by running `python out_imgc.py` for ZS-IMGC tasks, and `python out_kgc.py` for ZS-KGC tasks
3. Using the selected embeddings to perform the downstream ZSL methods, including `DOZSL_GAN` (for generation based) and `DOZSL_GCN` (for propagation based)

