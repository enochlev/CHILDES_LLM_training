
import re
from unsloth import FastLanguageModel
import torch
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import random
from trl import GRPOConfig, GRPOTrainer
import pandas as pd
from unsloth.chat_templates import apply_chat_template

from length_bayesian import BayesianSentenceLengthSkewModel

#cd to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))
#set visiable devies to 3

import sys
#default{ "steps": 5000, "base_model_name": "HuggingFaceTB/SmolLM2-360M-Instruct", "model_output_name": "llm-grpo-toddler-small-2" , "downscalling": 3 },
steps = int(sys.argv[1])
base_model_name = sys.argv[2]
model_output_name  = sys.argv[3]
downscalling = float(sys.argv[4])


use_vllm = False

print(f"Steps: {steps}")
print(f"Base Model Name: {base_model_name}")
print(f"Model Output Name: {model_output_name}")

# steps = 4
# base_model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
# model_output_name = "llm-grpo-toddler-small-2"


#HuggingFaceTB/SmolLM2-135M-Instruct
#HuggingFaceTB/SmolLM2-360M-Instruct
#HuggingFaceTB/SmolLM2-1.7B-Instruct

max_prompt_length = 96
max_completion_length = 96
max_seq_length = max_prompt_length + max_completion_length
lora_rank = 128



model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_name,##"meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    load_in_8bit = False,
    fast_inference = use_vllm, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.65, # Reduce if out of memory
    #float8_kv_cache = True, # Enable 8bit cache for key and value
)


model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)





# Optional extra params for vLLM
if use_vllm:
    from unsloth import vLLMSamplingParams
    vllm_sampling_params = vLLMSamplingParams(
        #stop = [". ", "? "],
        #bad_words = ["\n"],
        include_stop_str_in_output = False,
        min_tokens = 4,
        #min_p = 0.1,
        #seed = 3407,
    )

training_args = GRPOConfig(
    learning_rate = 5e-6,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    per_device_train_batch_size = 256,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_prompt_length,
    num_train_epochs = 4, # Set to 1 for a full training run
    max_steps = steps,
    report_to = "none", # Can use Weights & Biases
    vllm_sampling_params = vllm_sampling_params if use_vllm else None, # Optional
    temperature = 1.0,
    #save_strategy = "steps",
    #save_steps = 249,
    #output_dir = "llm-grpo-toddler-medium-1",
    reward_weights = [1.0, 1.0/downscalling, .15],
)




def childish_reward(prompts, completions, **kwargs) -> list[float]:  
    #computes reward score of completions, to make sure it is the tone of the reward model
    if "<|im_start|>" in prompts[0]:#huggingface
        prompts = [prompt.split("<|im_start|>user\n",1)[-1].split("<|im_end|>",1)[0] for prompt in prompts]
    elif "<|start_header_id|>" in prompts[0]:#llama
        prompts = [prompt.split("<|start_header_id|>user<|end_header_id|>\n\n",1)[-1].split("<|eot_id|>",1)[0] for prompt in prompts]
    elif "<start_of_turn>" in prompts[0]:#gemma
        prompts = [prompt.split("\n\n",1)[-1].split("<end_of_turn>\n",1)[0] for prompt in prompts]
    else:
        assert False

    responses = [[completion] for completion in completions]
    
    scores = model_reward.predict(responses,batch_size=128).tolist()

    return scores


def coherence_reward(prompts, completions, **kwargs) -> list[float]:
    if "<|im_start|>" in prompts[0]:#huggingface
        prompts = [prompt.split("<|im_start|>user\n",1)[-1].split("<|im_end|>",1)[0] for prompt in prompts]
    elif "<|start_header_id|>" in prompts[0]:#llama
        prompts = [prompt.split("<|start_header_id|>user<|end_header_id|>\n\n",1)[-1].split("<|eot_id|>",1)[0] for prompt in prompts]
    elif "<start_of_turn>" in prompts[0]:#gemma
        prompts = [prompt.split("\n\n",1)[-1].split("<end_of_turn>\n",1)[0] for prompt in prompts]
    else:
        assert False

    #completion = [completion for completion in completions]
    pairs = list(zip(prompts,completions))
    score = model_coherence.predict(pairs)#coherence

    #tranformer the score to be between 0 and 1 using df_coherence_min and df_coherence_max
    score = [(s - df_coherence_min) / (df_coherence_max - df_coherence_min) for s in score]

    if random.random() < 0.02:
        print("Prompt:", prompts[0])
        print("Completion:", completions[0])
        print("Score:", score[0])
    #apply min max normalization so that it is between 0 and 1 and not between min and max
    return score





#predict cosin similairty of responce to answer
def coherence_reward_2(prompts, completions, **kwargs) -> list[float]:

    if "<|im_start|>" in prompts[0]:#huggingface
        prompts = [prompt.split("<|im_start|>user\n",1)[-1].split("<|im_end|>",1)[0] for prompt in prompts]
    elif "<|start_header_id|>" in prompts[0]:#llama
        prompts = [prompt.split("<|start_header_id|>user<|end_header_id|>\n\n",1)[-1].split("<|eot_id|>",1)[0] for prompt in prompts]
    elif "<start_of_turn>" in prompts[0]:#gemma
        prompts = [prompt.split("\n\n",1)[-1].split("<end_of_turn>\n",1)[0] for prompt in prompts]
    else:
        assert False

    #completion = [completion for completion in completions]
    pairs = list(zip(prompts,completions))
    scores_final = []
    score_1 = model_coherence.predict(pairs)#coherence
    score_2 = model_coherence2.predict(pairs)#similarity
    score_3 = model_coherence2_2.predict(pairs)#question similarity
    score_ = model_coherence3.predict(pairs)# coherence logic
    score_4 = score_[:,0]/5
    score_5 = score_[:,1]/5
    score_6 = score_[:,2]/5
    import math

    for i in range(len(pairs)):
        # Extract for clarity
        s1   = score_1[i]
        s2   = score_2[i]
        s3  = score_3[i]
        s4 = score_4[i]
        s5 = score_5[i]
        s6 = score_6[i]
        #part to modify
        # 1) Reward moderate to high s1 (coherence). Let’s just keep it as-is:
        t2 = 3.0 * (s2 - s2**2)
        t3 = 3.0 * (s3 - s3**2)   # previously 4.0
        t6 = 2.0 * (s6 - s6**2)
        t1 = 1.0 * s1            # linear in s1


        penalty_s4 = -1.0 * (s4**2)
        penalty_s5 = -0.5 * (s5**2)

        raw = t1 + t2 + t3 + t6 + penalty_s4 + penalty_s5

        if s3 > 0.7:
            raw -= 2.5 * (s3 - 0.7)**2   # bumped from 2.0 up to 2.5


        if (s2 > 0.75) and (s3 > 0.75):
            raw -= 0.8

        # 3) Keep the existing s4 checks for non‐coherence:
        if s4 > 0.5:
            raw -= 2.0 * (s4 - 0.5)
        if s4 < -0.5:
            raw -= 1.5 * (-0.5 - s4)


        avg = (raw + 2.0) / 5.0
        scores_final.append(avg)


    if random.random() < 0.01:
        print("Prompt:", prompts[0])
        print("Completion:", completions[0])
        print("Score:", scores_final[0])
    #apply min max normalization so that it is between 0 and 1 and not between min and max
    return scores_final

def length_reward(prompts, completions, **kwargs) -> list[float]:
    """
    length_reward_model = BayesianSentenceLengthSkewModel.load("../models/bayesian_sentence_length_model.model")
    length_reward_model.predict("This is a nice thing to do in summer.",temperature=4)
    """
    if "<|im_start|>" in prompts[0]:#huggingface
        prompts = [prompt.split("<|im_start|>user\n",1)[-1].split("<|im_end|>",1)[0] for prompt in prompts]
    elif "<|start_header_id|>" in prompts[0]:#llama
        prompts = [prompt.split("<|start_header_id|>user<|end_header_id|>\n\n",1)[-1].split("<|eot_id|>",1)[0] for prompt in prompts]
    elif "<start_of_turn>" in prompts[0]:#gemma
        prompts = [prompt.split("\n\n",1)[-1].split("<end_of_turn>\n",1)[0] for prompt in prompts]
    else:
        assert False

    scores = [model_length.predict(completion,temperature=1) for completion in completions]
    #also give a lower score for each number of punctuation marks. Were 1 is ok. use 1/x. but if there is 0 then also give 1.0 score
    scores = [score * (1/(max(1, len(re.findall(r'[.!?,;:]', completion))))) for score, completion in zip(scores, completions)]

    return scores



from transformers import TrainerCallback
class CustomLogger(TrainerCallback):
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, **kwargs):
        # Create an entry with the fields you expect.
        # Adjust the keys ('step', 'training_loss', etc.) if necessary.

        logs = kwargs["logs"]
        entry = {
            "Step": state.global_step,
            "Training Loss": logs.get("training_loss"),
            "reward": logs.get("reward"),
            "reward_std": logs.get("reward_std"),
            "completion_length": logs.get("completion_length"),
            "kl": logs.get("kl"),
            "rewards / coherence_reward": logs.get("rewards/coherence_reward"),
            "rewards / childish_reward": logs.get("rewards/childish_reward"),
            "rewards / length_reward": logs.get("rewards/length_reward"),
        }

        save_frequency = [500, 750, 1000,1250, 1500, 1750, 2000 ,3500, 3000,4500, 5000]
        if steps not in save_frequency:
            save_frequency.append(steps)

        self.history.append(entry)

        pd.DataFrame(self.history).to_csv(f"{model_output_name}.csv", index=False)


        if state.global_step in save_frequency:
            if os.path.exists(f"../../scratch/models/{model_output_name}") == False:
                os.mkdir(f"../../scratch/models/{model_output_name}")
           
            model.save_pretrained_merged(f"../../scratch/models/{model_output_name}/step-{state.global_step}", tokenizer, save_method = "merged_16bit",maximum_memory_usage=0.8)
            #model.save_pretrained(f"../../scratch/models/{model_output_name}/step-{state.global_step}")
            #tokenizer.save_pretrained(f"../../scratch/models/{model_output_name}/step-{state.global_step}")
    def on_train_begin(self, args, state, control, **kwargs):
        pass

custom_logger = CustomLogger()


from datasets import load_dataset, Dataset

df = pd.read_csv('../data/child_response_pairs.csv')
df_coherence_min = df['child_coherence_score_prediction'].min()
df_coherence_max = df['child_coherence_score_prediction'].max()
df = df.drop_duplicates(subset=['text'])
#df.text mus have question mark at the end
df.text = df.text.str.strip()
#sort by child_coherence_score_prediction descending so that we can get the best parent questions#get top 10%
df = df.sort_values(by='adult_coherence_score_prediction', ascending=False).head(df.shape[0]//10).sample(frac=1, random_state=42).reset_index(drop=True)
df = df.reset_index(drop=True)
df = df.rename(columns={'text': 'prompt'})
df = df[['prompt']]


prefix = "You are a toddler-llm. Respond in a childish manner."

if "llama" in base_model_name.lower():
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
    tokenizer.chat_template = tokenizer.chat_template.replace("""{{- "Cutting Knowledge Date: December 2023\n" }}\n{{- "Today Date: " + date_string + "\n\n" }}""","")
    #'<|start_header_id|>user<|end_header_id|>\n\naNswer the dialgue questions in a short and concise manner. Question: are you excited for your birthday?<|eot_id|>\n\n'
    #'<|start_header_id|>assistant<|end_header_id|>\nYes, I am excited for my birthday.<|eot_id|>\n'
    df['prompt'] = df['prompt'].apply(lambda x: tokenizer.apply_chat_template([{'role': 'system', 'content': prefix}, {'role': 'user', 'content': x}], tokenize=False) + "<|start_header_id|>assistant<|end_header_id|>\n\n")


elif "gemma" in base_model_name.lower():
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3"
        )
    #'<bos><start_of_turn>user\nAnswer the dialgue questions in a short and concise manner.Question: are you excited for your birthday?<end_of_turn>\n'
    #<start_of_turn>model\nYes, I am excited for my birthday.<end_of_turn>\n'
    df['prompt'] = df['prompt'].apply(lambda x: tokenizer.apply_chat_template([{'role': 'system', 'content': prefix}, {'role': 'user', 'content': x}], tokenize=False) + "<start_of_turn>model\n")


else:

    #'<|im_start|>user\nAnswer the dialgue questions in a short and concise manner. Question: are you excited for your birthday?<|im_end|>\n'
    #'<|im_start|>assistant\nYes, I am excited for my birthday.<|im_end|>\n'
    df['prompt'] = df['prompt'].apply(lambda x: tokenizer.apply_chat_template([{'role': 'system', 'content': prefix}, {'role': 'user', 'content': x}], tokenize=False) + "assistant\n")

dataset = Dataset.from_pandas(df)



trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        childish_reward,
        coherence_reward,
        length_reward,
    ],
    args = training_args,
    train_dataset = dataset,
)
trainer.add_callback(custom_logger)


from sentence_transformers import CrossEncoder

automodel_args = {"torch_dtype": torch.bfloat16}
model_reward = CrossEncoder("../models/childish_reward_model",device='cuda', max_length=256)#, automodel_args=automodel_args)
model_coherence = CrossEncoder("../models/child_coherence_model",device='cuda',max_length=356)#, automodel_args=automodel_args)
model_coherence2_2 = CrossEncoder("cross-encoder/stsb-roberta-large",device='cuda',max_length=356)#, automodel_args=automodel_args)
model_coherence2 = CrossEncoder("cross-encoder/quora-roberta-base",device='cuda',max_length=356)#, automodel_args=automodel_args)
model_coherence3 = CrossEncoder("cross-encoder/nli-deberta-v3-base",device='cuda',max_length=356)#use [:,1] after prediction and / 5 to normalize it
WEIGHT = [1.0,0.25,1.0]

model_length = BayesianSentenceLengthSkewModel.load("../models/bayesian_sentence_length_model.model")

results = trainer.train()


#model.save_pretrained_merged(model_output_name, tokenizer, save_method = "merged_16bit",maximum_memory_usage=0.7)
pd.DataFrame(custom_logger.history).to_csv(f"{model_output_name}.csv", index=False)
pd.DataFrame(custom_logger.history).to_csv(f"../../scratch/models/{model_output_name}/training_log.csv", index=False)