# 1. Full pipeline at 2048 (RAGTAG + both fine-tunes)
./run_experiment.sh \
  --datasets "issues3k.csv" \
  --models "unsloth/Llama-3.2-3B-Instruct:50,unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-14B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-32B-Instruct-bnb-4bit:50" \
  --test_sizes "issues3k.csv:0.5" \
  --top_ks "1,3,5,9,15" \
  --max_seq_length 2048 \
  --ft_max_seq_length 2048 \
  --inference_batch_size 1 \
  --results_dir results/issues3k_ctx2048 \
  --nrp

# 2. RAGTAG-only at 4096
./run_experiment.sh \
  --datasets "issues3k.csv" \
  --models "unsloth/Llama-3.2-3B-Instruct:50,unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-14B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-32B-Instruct-bnb-4bit:50" \
  --test_sizes "issues3k.csv:0.5" \
  --top_ks "1,3,5,9,15" \
  --max_seq_length 4096 \
  --inference_batch_size 1 \
  --skip_flawed_ft \
  --skip_fixed_ft \
  --results_dir results/issues3k_ctx4096 \
  --nrp

# 3. RAGTAG-only at 8192
./run_experiment.sh \
  --datasets "issues3k.csv" \
  --models "unsloth/Llama-3.2-3B-Instruct:50,unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-14B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-32B-Instruct-bnb-4bit:50" \
  --test_sizes "issues3k.csv:0.5" \
  --top_ks "1,3,5,9,15" \
  --max_seq_length 8192 \
  --inference_batch_size 1 \
  --skip_flawed_ft \
  --skip_fixed_ft \
  --results_dir results/issues3k_ctx8192 \
  --nrp

# 4. RAGTAG-only at 16384
./run_experiment.sh \
  --datasets "issues3k.csv" \
  --models "unsloth/Llama-3.2-3B-Instruct:50,unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-14B-Instruct-bnb-4bit:50,unsloth/Qwen2.5-32B-Instruct-bnb-4bit:50" \
  --test_sizes "issues3k.csv:0.5" \
  --top_ks "1,3,5,9,15" \
  --max_seq_length 16384 \
  --inference_batch_size 1 \
  --skip_flawed_ft \
  --skip_fixed_ft \
  --results_dir results/issues3k_ctx16384 \
  --nrp