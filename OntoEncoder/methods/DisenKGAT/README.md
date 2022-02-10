### running command

#### for IMGC

python run_mult.py --dataset ImNet_A --save_name DisenKAGT_TransE_mult_K2_D100_ImNet_A --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 2
python run_mult.py --dataset ImNet_A --save_name DisenKAGT_TransE_mult_K4_D100_ImNet_A --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 4
python run_mult.py --dataset ImNet_A --save_name DisenKAGT_TransE_mult_K5_D100_ImNet_A --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 5


python run_mult.py --dataset ImNet_O --save_name DisenKAGT_TransE_mult_K2_D100_ImNet_O --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 2
python run_mult.py --dataset ImNet_O --save_name DisenKAGT_TransE_mult_K4_D100_ImNet_O --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 4
python run_mult.py --dataset ImNet_O --save_name DisenKAGT_TransE_mult_K5_D100_ImNet_O --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 5

python run_mult.py --dataset AwA --save_name DisenKAGT_TransE_mult_K2_D100_AwA --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 2
python run_mult.py --dataset AwA --save_name DisenKAGT_TransE_mult_K4_D100_AwA --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 4
python run_mult.py --dataset AwA --save_name DisenKAGT_TransE_mult_K5_D100_AwA --init_dim 100 --gcn_dim 100 --embed_dim 100 --num_factors 5


### for NELL

python run_mult.py --dataset NELL --save_name DisenKAGT_TransE_mult_K2_D200_NELL --num_factors 2
python run_mult.py --dataset NELL --save_name DisenKAGT_TransE_mult_K4_D200_NELL --num_factors 4
python run_mult.py --dataset NELL --save_name DisenKAGT_TransE_mult_K9_D200_NELL --num_factors 9


### for Wiki

python run_mult.py --dataset Wiki --save_name DisenKAGT_TransE_mult_K2_D200_Wiki --num_factors 2
python run_mult.py --dataset Wiki --save_name DisenKAGT_TransE_mult_K4_D200_Wiki --num_factors 4
python run_mult.py --dataset Wiki --save_name DisenKAGT_TransE_mult_K9_D200_Wiki --num_factors 9

