
## Running Command

### Default Ontology Encoder
Running the script `pretrain_struc.py` to pretrain the structural representation of Ontological Schema.

**For AwA & ImNet_A/O**
```
python run.py --dataset AwA --hidden_dim 100 --save_name TransE_D100_AwA
python run.py --dataset ImNet_A --hidden_dim 100 --save_name ImNet_A_D100_AwA
python run.py --dataset ImNet_O --hidden_dim 100 --save_name ImNet_O_D100_AwA
```
**For NELL-ZS & Wiki-ZS**
```
python run.py --dataset NELL --hidden_dim 200 --save_name TransE_D200_NELL
python run.py --dataset Wiki --hidden_dim 200 --save_name TransE_D200_Wiki
```
