from openai import OpenAI
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dotenv import load_dotenv
from tqdm import tqdm
os.chdir(os.path.dirname(__file__))
os.chdir("..")
load_dotenv()
os.chdir(os.path.dirname(__file__))

# client = OpenAI(
#     api_key = os.getenv("GROQ_API_KEY"),
#     base_url= "https://api.groq.com/openai/v1"
# )
client = OpenAI()

system_prompt_batch = """You are an expert linguist and cognitive scientist. Your task is to critically evaluate short conversational exchanges between adult and child speakers. 

You will receive conversational pairs in a batch composed of two utterances:
1) Adult speaker utterance
2) Child speaker response

Your goal is to decide how semantically coherent and contextually sensible the conversation is, evaluating three specific dimensions:

1. Adult utterance coherence: Measures if the adult's utterance makes sense on its own, without external context. 
   - High score (0.8-1.0):
     • "What's your favorite color?" (clear, standalone question)
     • "Do you want to play with blocks?" (clear activity proposal)
     • "Tell me about your day at school." (clear prompt for information)
   - Medium score (0.4-0.7):
     • "The one over there." (somewhat comprehensible but lacks specificity)
     • "And then what happened?" (depends on prior conversation)
   - Low score (0.0-0.3): 
     • "And three of them" (incomprehensible without context)
     • "So when it does." (incomplete thought)
     • "Because the green." (fragmentary and unclear)

2. Child response coherence: Measures if the child's response reasonably relates to the adult's utterance.
   - High score (0.8-1.0):
     • Adult: "What's your favorite color?" Child: "Blue!" 
     • Adult: "Do you want a snack?" Child: "Yes, please!"
     • Adult: "How many fingers am I holding up?" Child: "Three!"
   - Medium score (0.4-0.7):
     • Adult: "How was school?" Child: "I drew a picture." (related but indirect)
     • Adult: "Would you like to read a book?" Child: "I'm hungry." (not directly answering)
   - Low score (0.0-0.3): 
     • Adult: "What's your favorite color?" Child: "Uh, dinosaur."
     • Adult: "Is it time for bed?" Child: "The moon is cheese."
     • Adult: "Do you want milk?" Child: "My truck is red."

3. Child response strict context coherence: Measures if the child's response accurately addresses established context within the conversation.
   - High score (0.8-1.0): (this is the most strict measure and rarely given. Give this scores to coherent, thought out, and contextually relevant responses.)
     • Adult: "I gave you 3 apples, how many do you have?" Child: "I have 3 apples."
     • Adult: "What do you want to do?" Child: "What can I do?" (appropriate request for clarification)
     • Adult: "Here's a red ball, what color is it?" Child: "It's red." (correctly answered to what was available in context)
     • Adult: "Look at these five cookies I baked. How many cookies are there?" Child: "Five cookies."
   - Medium score (0.4-0.7): (this is also rare, but slightly more common then the strict measure.)
     • Adult: "Did you see the dog at the park?" Child: "Dogs go woof." (related but not addressing the specific question)
     • Adult: "What letter does apple start with?" Child: "I like apples." (related but not addressing the specific question)
   - Low score (0.0-0.3): (common, even if child responce is coherent. Also use this for cases where the child is in context but responce is short, minimal, or child is answering a question that is not in context.)
     • Adult: "What do you want to do?" Child: "Catch the ball." (referencing a ball not established in context)
     • Adult: "How many apples do you have?" Child: "I have 3 apples." (claiming specific quantity without established context)
     • Adult: "What did you eat for lunch?" Child: "I ate a sandwich." (when the sandwich is not established in context)
     • Adult: "Should we go to the store?" Child: "I want the blue one." (nothing blue is in context)
     • Adult: "What did you do?" Child: "First I went to the park. Then I played with Ben. After I bought candy." (rambling, we don't know there is a park, Ben, or candy in context)
     • Adult: "Ok and how do you want to do that?" Child: "Then he goes up the lader and fight him and wins." (No context established by the parent, it should be "What do you me to do?" to get context)

Provide your evaluation strictly as a numeric coherence probability score between 0.0 and 1.0, formatted as JSON like this:

{"0": 
  {
    "adult_utterance_coherence": "<score between 0.0 and 1.0>",
    "child_response_coherence": "<score between 0.0 and 1.0>",
    "child_response_strict_context_coherence": "<score between 0.0 and 1.0>"
  },
 "1":
  {
    "adult_utterance_coherence": "<score between 0.0 and 1.0>",
    "child_response_coherence": "<score between 0.0 and 1.0>",
    "child_response_strict_context_coherence": "<score between 0.0 and 1.0>"
  }
}
...

Only respond with the JSON output as shown. Provide no additional explanation."""

def get_coherence_score_batch(adult_utterances, child_utterances, model="llama3-70b-8192"):
    """
    Process multiple adult-child conversation pairs in a single batch request.
    
    Parameters:
        adult_utterances (list): List of adult utterances
        child_utterances (list): List of child utterances
        
    Returns:
        dict: JSON response with coherence scores for each conversation pair
    """
    
    assert len(adult_utterances) == len(child_utterances), "Input lists must be of equal length"
    
    # Construct the batch prompt
    batch_prompt = ""
    for i in range(len(adult_utterances)):
        batch_prompt += f"""id: {i}
adult_utterance: "{adult_utterances[i]}"
child_utterance: "{child_utterances[i]}"

"""
    
    # Add retry logic - 3 attempts with 5 second delay
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt_batch},
                    {"role": "user", "content": batch_prompt},
                ],
                temperature=0.5,
                response_format={"type": "json_object"},
                n=3
            )
            #average of the 3
            #first but the numbers in a list, then averag them all
            final_json = {}
            #put in list inside of final_json
            for choice in response.choices:
                result = choice.message.content.strip()
                result_json = json.loads(result)
                for k in result_json:
                    if k not in final_json:
                        final_json[k] = {}
                    for sub_k in result_json[k]:
                        if sub_k not in final_json[k]:
                            final_json[k][sub_k] = []
                        try:
                            final_json[k][sub_k].append(float(result_json[k][sub_k]))
                        except:
                            pass

            #average the numbers
            for k in final_json:
                for sub_k in final_json[k]:
                    #mean
                    #m = sum(final_json[k][sub_k]) / len(final_json[k][sub_k])
                    #harmanic mean
                    hm = len(final_json[k][sub_k]) / sum([1/max(x,.05) for x in final_json[k][sub_k]])
                    #min
                    #m = min(final_json[k][sub_k])

                    #final_json[k][sub_k] = round((hm + m) / 2,2)
                    final_json[k][sub_k] = round(hm,2)
                    #final_json[k][sub_k] = round(m,2)
            
            return final_json
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error on attempt {attempt+1}: {str(e)}. Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed after {max_retries} attempts: {str(e)}")
                return None

models=[
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "llama-3.2-90b-vision-preview",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "meta-llama/llama-4-maverick-17b-128e-instruct"
]

models=["gpt-4o"]

# Create a round-robin model selector
class ModelSelector:
    def __init__(self, models):
        self.models = models
        self.index = 0
        self.lock = threading.Lock()
    
    def next_model(self):
        with self.lock:
            model = self.models[self.index]
            self.index = (self.index + 1) % len(self.models)
            return model

def process_batch(batch_df, model_selector, result_list,user_input=False):
    """Process a single batch with the next available model"""
    model_name = model_selector.next_model()
    
    adult_utterances = batch_df["text"].tolist()
    child_utterances = batch_df["CHI_response"].tolist()
    
    scores = get_coherence_score_batch(adult_utterances, child_utterances, model=model_name)
    
    if type(scores) == dict:
        scores = {str(k): v for k, v in scores.items()}
        
        batch_results = []
        for j, (_, row) in enumerate(batch_df.iterrows()):
            score_key = str(j)
            if score_key in scores:
                adult_coherence_score = None
                if "adult_utterance_coherence" in scores[score_key]:
                    adult_coherence_score = scores[score_key]["adult_utterance_coherence"]
                else:
                    for key in scores[score_key]:
                        if "adult" in key:
                            adult_coherence_score = scores[score_key][key]
                            break
                
                child_coherence_score = None
                if "child_responce_coherence" in scores[score_key]:
                    child_coherence_score = scores[score_key]["child_responce_coherence"]
                else:
                    for key in scores[score_key]:
                        if "child" in key:
                            child_coherence_score = scores[score_key][key]
                            break

                strict_child_coherence_score = None
                if "child_response_strict_context_coherence" in scores[score_key]:
                    strict_child_coherence_score = scores[score_key]["child_response_strict_context_coherence"]
                else:
                    for key in scores[score_key]:
                        if "strict" in key:
                            strict_child_coherence_score = scores[score_key][key]
                            break
                
                batch_results.append({
                    "folder_name": row["folder_name"],
                    "root_name": row["root_name"],
                    "child_name": row["child_name"],
                    "years": row["years"],
                    "months": row["months"],
                    "days": row["days"],
                    "speaker": row["speaker"],
                    "text": row["text"],
                    "CHI_response": row["CHI_response"],
                    "adult_coherence_score": adult_coherence_score,
                    "child_coherence_score": child_coherence_score,
                    "strict_child_coherence_score": strict_child_coherence_score,
                })
            else:
                print(f"Missing score for index {j} in batch")
        if user_input:
            # Display batch inputs/outputs and ask for approval
            print("\n===== BATCH REVIEW =====")
            print(f"Using model: {model_name}")
            for i, result in enumerate(batch_results):
                print(f"\nItem {i+1}/{len(batch_results)}")
                print(f"Parent: \"{result['text']}\"")
                print(f"Child: \"{result['CHI_response']}\"")
                print(f"Scores: Adult={result['adult_coherence_score']}, Child={result['child_coherence_score']}, Strict={result['strict_child_coherence_score']}")
            
            # Ask for approval
            user_input = input("\nApprove this batch? y/(n): ")
        else:
            user_input = 'y'
        
        with threading.Lock():
            if user_input.lower() == 'y':  # Default is no
                result_list.extend(batch_results)
                print(f"Batch approved - {len(batch_results)} records added")
                # Periodically save results
                if len(result_list) % (batch_size * 5) < batch_size:
                    temp_df = pd.DataFrame(result_list)
                    temp_df.to_csv("child_response_pairs_scored.csv", index=False)
                    print(f"Saved intermediate results: {len(result_list)} records")
                return len(batch_results)
            else:
                print("Batch rejected - no records added")
                return 0
    else:
        print(f"Failed to get scores with model {model_name}")
        return 0

def process_data(df, batch_size=1, max_workers=5, user_input=False):
    """Process data with concurrent workers and round-robin model selection"""    
    try:
        df_dataset = pd.read_csv("child_response_pairs_scored.csv")
        result_list = df_dataset.to_dict('records')
        print(f"Loaded {len(result_list)} existing records")
    except:
        result_list = []
        print("No existing records found, starting fresh")
    
    # Initialize model selector with all available models
    model_selector = ModelSelector(models)
    
    tasks = []
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i+batch_size]
        tasks.append(batch_df)
    
    # Process in batches with multiple workers
    processed_items = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch_df in tasks:
            futures.append(executor.submit(process_batch, batch_df, model_selector, result_list,user_input))
        
        # Single progress bar for all processing
        with tqdm(total=len(df), desc="Processing items") as pbar:
            for future in as_completed(futures):
                try:
                    processed = future.result()
                    processed_items += processed
                    pbar.update(processed)
                except Exception as e:
                    print(f"Error in worker: {str(e)}")
    
    # Final save
    df_dataset = pd.DataFrame(result_list)
    df_dataset.to_csv("child_response_pairs_scored.csv", index=False)
    return df_dataset

# Main execution code
import pandas as pd

# Replace the bottom section with this code
df = pd.read_csv("child_response_pairs.csv")

if "child_coherence_score_prediction" in df.columns:
    df = df[df["child_coherence_score_prediction"] > 0.3]

#df = df[(df["text"].str.strip().str.endswith("?"))]

#df = df[(df["text"].str.len() > 20) & (df["text"].str.len() < 1000)]
df = df[(~df["CHI_response"].str.contains("yeah.")) & (df["CHI_response"].str.len() < 1000)]

df = df.sample(128)

# Set your parameters here
batch_size = 64  # Batch size for processing
max_workers = 1  # Number of concurrent workers (adjust based on your system)

df_result = process_data(df, batch_size=batch_size, max_workers=max_workers, user_input=True)
print(f"Processed {len(df_result)} records successfully")