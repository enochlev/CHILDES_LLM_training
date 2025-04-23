#!/bin/bash
#SBATCH -J grpo_trainer                # Job name
#SBATCH -o slurm_gpu_%j.out            # Standard output and error log
#SBATCH -p gpu-l40                    # Partition for short GPU jobs (≤7 days)
#SBATCH -t 24:00:00                    # Time limit of 24 hours
#SBATCH --nodes=1                      # Use one node
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=16              # Minimal CPU allocation
#SBATCH --gres=gpu:1                   # Request one GPU

echo "Job starting at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "GPUs allocated: $SLURM_JOB_GPUS"


. ~/.bashrc
conda activate childes
cd ~/CHILDES_LLM_training/GRPO_trainer
#python GRPO_trainer.py 1500 "HuggingFaceTB/SmolLM2-360M-Instruct" "llm-grpo-toddler-small-15" 3
python GRPO_trainer.py 5000 "/bsuhome/enochlevandovsky/scratch/checkpoints/smollm2-102M/out" "llm-grpo-toddler-tiny-16" 3

#python GRPO_trainer.py 2500 "HuggingFaceTB/SmolLM2-135M-Instruct" "llm-grpo-toddler-tiny-6" 3
#{ "steps": 7500, "base_model_name": "HuggingFaceTB/SmolLM2-135M-Instruct", "model_output_name": "llm-grpo-toddler-tiny-1", "downscalling": 4 },
#{ "steps": 5000, "base_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct", "model_output_name": "llm-grpo-toddler-small-2" , "downscalling": 3 },
#{ "steps": 2500, "base_model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "model_output_name": "llm-grpo-toddler-large-1", "downscalling": 4 },
#CUDA_VISIBLE_DEVICES=3 python GRPO_trainer.py 1000 "HuggingFaceTB/SmolLM2-360M-Instruct" "llm-grpo-toddler-small-0" 3

#llm-grpo-toddler-small-3 is complex coherence reward with ,.?!:; giving lower reward.
#llm-grpo-toddler-tiny-3 is complex coherence reward with ,.?!:; giving lower reward.
#llm-grpo-toddler-small-4 is easy coherence reward with ,.?!:; giving lower reward.
#llm-grpo-toddler-small-5 is upgraded complex coherence reward with ,.?!:; giving lower reward.
#llm-grpo-toddler-tiny-5 is upgraded complex coherence reward with ,.?!:; giving lower reward.
#llm-grpo-toddler-small-6 is easy upgraded coherence reward with ,.?!:; giving lower reward.
#llm-grpo-toddler-tiny-6 is easy upgraded coherence reward with ,.?!:; giving lower reward.

echo "Job finished at: $(date)"
