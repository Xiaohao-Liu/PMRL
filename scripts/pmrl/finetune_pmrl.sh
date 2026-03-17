CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/vast/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir model_weights/VAST \
--checkpoint model_weights/PMRL/base.pt \
--output_dir outputs/finetune_pmrl_msrvtt \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name finetune_pmrl_msrvtt 

CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/vast/finetune_cfg/retrieval-didemo.json \
--pretrain_dir model_weights/VAST \
--checkpoint model_weights/PMRL/base.pt \
--output_dir outputs/finetune_pmrl_didemo \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name finetune_pmrl_didemo


CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/vast/finetune_cfg/retrieval-activitynet.json \
--pretrain_dir model_weights/VAST \
--checkpoint model_weights/PMRL/base.pt \
--output_dir outputs/finetune_pmrl_activitynet \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name finetune_pmrl_activitynet


CUDA_VISIBLE_DEVICES=0,3,4,5 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/vast/finetune_cfg/retrieval-vatex.json \
--pretrain_dir model_weights/VAST \
--checkpoint model_weights/PMRL/base.pt \
--output_dir outputs/finetune_pmrl_vatex \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name finetune_pmrl_vatex

