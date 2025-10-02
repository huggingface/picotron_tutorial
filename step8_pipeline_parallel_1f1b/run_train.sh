torchrun --nproc_per_node 2 train.py --pp_size 2 --pp_engine 1f1b --micro_batch_size 4 --gradient_accumulation_steps 2 --seq_len 128 --max_tokens 204800 --num_proc 16 --run_name pp_1f1b
# debugpy-run -m torch.distributed.run -- --nproc_per_node 2 train.py --pp_size 2 --pp_engine 1f1b --micro_batch_size 4 --gradient_accumulation_steps 2 --seq_len 128 --max_tokens 204800 --num_proc 16 --run_name pp_1f1b

# DP = 2 / PP = 2
# torchrun --nproc_per_node 4 train.py --dp_size 2 --pp_size 2 --pp_engine 1f1b --micro_batch_size 4 --gradient_accumulation_steps 2 --seq_len 128 --max_tokens 204800 --num_proc 16 --run_name pp_1f1b


# DP=2 / TP=2 / PP=2
# torchrun --nproc_per_node 8 train.py --dp_size 2 --tp_size 2 --pp_size 2 --pp_engine 1f1b --micro_batch_size 4 --gradient_accumulation_steps 2 --seq_len 128 --max_tokens 204800 --num_proc 16 --run_name pp_1f1b