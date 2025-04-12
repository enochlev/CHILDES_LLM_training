#do OS python calls
#/home/enochlev/.local/share/virtualenvs/unlsoth--v9BSFsR/bin/python testing.py 4 llm-grpo-toddler-small-2 HuggingFaceTB/SmolLM2-135M-Instruct
python_path = "python"
itterations = [
#{ "steps": 7500, "base_model_name": "HuggingFaceTB/SmolLM2-135M-Instruct", "model_output_name": "llm-grpo-toddler-tiny-1", "downscalling": 4 },
{ "steps": 5000, "base_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct", "model_output_name": "llm-grpo-toddler-small-2" , "downscalling": 4 },
#{ "steps": 2500, "base_model_name": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "model_output_name": "llm-grpo-toddler-large-1", "downscalling": 4 },
]

#vllm serve HuggingFaceTB/SmolLM2-135M-Instruct --gpu-memory-utilization .1


import os
for it in itterations:
    os.system(f"{python_path} GRPO_trainer.py {it['steps']} {it['base_model_name']} {it['model_output_name']} {it['downscalling']}")
    #print(f"Would run {python_path} GRPO_trainer/GRPO_trainer.py {it['steps']} {it['base_model_name']} {it['model_output_name']} {it['downscalling']}")
    import time
    print("Sleeping for 3 seconds<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    time.sleep(3)


#pip install --upgrade --no-cache-dir unsloth unsloth_zoo

#pip install --upgrade --no-cache-dir "unsloth[cu124-ampere-torch251] @ git+https://github.com/unslothai/unsloth.git"# "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo.git"
#pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5
