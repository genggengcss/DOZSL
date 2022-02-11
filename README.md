The code and data is for the paper xx

## Ontology Encoder

### Steps:

1.  Running the scripts in each method folder to pre-train the KGs and ontological schemas

2. Selecting the targte class embeddings from the trained concept embeddings by running `python out_imgc.py` for ZS-IMGC tasks, and `python out_kgc.py` for ZS-KGC tasks
3. Using the selected embeddings to perform the downstream ZSL methods, including `DOZSL_GAN` (for generation based) and `DOZSL_GCN` (for propagation based)

