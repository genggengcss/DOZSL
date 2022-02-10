### running command

#### for ImNet-A

python run.py --dataset ImNet_A --save_name DOZSL_AGG_sub_K4_D100_ImNet_A --num_factors 4

### for ImNet-O
python run.py --dataset ImNet_O --save_name DOZSL_AGG_sub_K4_D100_ImNet_O --num_factors 4

### for AwA
python run.py --dataset AwA --save_name DOZSL_AGG_sub_K4_D100_AwA --num_factors 4


### for NELL
python run.py --dataset NELL --save_name DOZSL_AGG_sub_K8_D200_NELL --num_factors 8 --init_dim 200 --gcn_dim 200 --embed_dim 200



### for Wiki
python run.py --dataset Wiki --save_name DOZSL_AGG_sub_K8_D200_Wiki --num_factors 8 --init_dim 200 --gcn_dim 200 --embed_dim 200


