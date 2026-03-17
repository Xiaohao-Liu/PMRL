CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/vast/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir model_weights/VAST \
--output_dir outputs/pretrain_pmrl \
--model_type pmrl \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl 
