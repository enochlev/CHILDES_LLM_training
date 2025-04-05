from openai import OpenAI
import json
import os
import time
client = OpenAI(
    api_key = os.getenv("GROQ_API_KEY"),
    base_url= "https://api.groq.com/openai/v1"
)

system_prompt = """
You are an expert linguist and cognitive scientist. Your task is to critically evaluate short conversational exchanges between adult and child speakers. 

You will receive conversational pairs composed of two utterances:
1) Adult speaker's utterance
2) Child speaker's response

Your goal is to decide how semantically coherent and contextually sensible the entire conversation is, considering both the adult's and child's utterances. The conversation should make sense as a whole, with each speaker's utterance building on or responding to the other. Children's speech may be grammatically simple or contain typical developmental errors, but the conversation should pragmatically and semantically flow clearly.

Provide your evaluation strictly as a numeric coherence probability score between 0.0 and 1.0, formatted as JSON like this:

{
  "coherence_score": <score between 0.0 and 1.0>
}

Use this scoring guidance:
- Score of 0.0–0.3 indicates the conversation either doesn't make sense contextually or is unrelated.
- Score of 0.4–0.5 indicates ambiguous relevance or partial sense, indicating uncertainty or partial coherence.
- Score of 0.5–0.7 indicates ambiguous relevance or partial sense, maybe the adult made sense but the child did not.
- Score of 0.7–1.0 indicates that the conversation makes clear sense contextually to any listener with no context of the conversation or setting.

Only respond with the JSON output as shown. Provide no additional explanation.
"""



system_prompt_batch = """
You are an expert linguist and cognitive scientist. Your task is to critically evaluate short conversational exchanges between adult and child speakers. 

You will receive conversational pairs in a batch composed of two utterances:
1) Adult speaker utterance
2) Child speaker response

Your goal is to decide how semantically coherent and contextually sensible the conversation is, considering both the adult and child utterances. 
The adults utterence coherence score is a score on how much that question/phrase makes sence out of context. For examples, "What's your favorite color?" is a coherent question/pgrase. "and three of them" is not a coherent question/phrase because it makes no sence out of context.
The child's response coherence score is a score on how much the child's response makes sence in the context of the adult's utterance. For example, if the adult asks "What's your favorite color?" and the child responds "Blue!" that would be a coherent response. If the child responds "uh dinasour" that would not be a coherent response. Children's speech may be grammatically simple or contain typical developmental errors, but the conversation should pragmatically and semantically flow clearly.

Provide your evaluation strictly as a numeric coherence probability score between 0.0 and 1.0, formatted as JSON like this:

{"0": 
  {
    "adult_utterance_coherence": "<score between 0.0 and 1.0>",
    "child_responce_coherence": <score between 0.0 and 1.0>
  },
 "1":
  {
    "adult_utteran .... same as above for the 2nd batch

Use this scoring guidance:
- Score of 0.0–0.3 indicates the conversation either doesn't make sense contextually or is unrelated.
- Score of 0.4–0.7 indicates ambiguous relevance or partial sense, indicating uncertainty or partial coherence.
- Score of 0.7–1.0 indicates that the parents' question and the child's response are coherent and contextually sensible.

Note a parents' question may be coherent but the child's response may not be, and vice versa.

Only respond with the JSON output as shown. Provide no additional explanation.
"""



def get_coherence_score(adult_utterance, child_utterance, model="llama3-70b-8192"):
    user_prompt_2 = f"""
Adult: "{adult_utterance}"
Child: "{child_utterance}"    
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_2},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content.strip()
        result_json = json.loads(result)
        return result_json
    except Exception as e:
        print(f"Unexpected response format: {str(e)}")
        return None


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
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt_batch},
                {"role": "user", "content": batch_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        result = response.choices[0].message.content.strip()
        result_json = json.loads(result)
        return result_json
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        return None


def process_data(df, batch_size=1, model_name="llama3-70b-8192"):
    """Process data with either individual or batch API calls based on batch_size"""
    try:
        df_dataset = pd.read_csv("child_response_pairs_scored.csv")
    except:
        df_dataset = df.head(0)
    
    trys = 0
    RATE_LIMIT = 15  # Number of allowed calls per minute
    rate_limit = []
    
    if batch_size == 1:
        # Process one item at a time
        for _, row in df.iterrows():
            current_time = time.time()
            rate_limit = [t for t in rate_limit if current_time - t < 60]
            
            if len(rate_limit) >= RATE_LIMIT:
                time.sleep(2 * 60/RATE_LIMIT)
                print("Rate limit reached, waiting...")
            
            score = get_coherence_score(row["text"], row["CHI_response"], model=model_name)
            if score is not None:
                trys = 0
                rate_limit.append(current_time)
                df_dataset = pd.concat([df_dataset, pd.DataFrame([{
                    "folder_name": row["folder_name"],
                    "root_name": row["root_name"],
                    "child_name": row["child_name"],
                    "years": row["years"],
                    "months": row["months"],
                    "days": row["days"],
                    "speaker": row["speaker"],
                    "text": row["text"],
                    "CHI_response": row["CHI_response"],
                    "score": score["coherence_score"],
                }])])
                # Save periodically
                if len(df_dataset) % 10 == 0:
                    df_dataset.to_csv("child_response_pairs_scored.csv", index=False)
                    print(f"Progress: {len(df_dataset)} records processed")
            else:
                df_dataset.to_csv("child_response_pairs_scored.csv", index=False)
                trys += 1
                if trys > 3:
                    print("Failed too many times, exiting")
                    break
                print("Failed to get score for:", row["text"], row["CHI_response"])
    else:
        # Process in batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
            
            current_time = time.time()
            rate_limit = [t for t in rate_limit if current_time - t < 60]
            
            if len(rate_limit) >= RATE_LIMIT:
                wait_time = 2 * 60/RATE_LIMIT
                print(f"Rate limit reached, waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            
            adult_utterances = batch_df["text"].tolist()
            child_utterances = batch_df["CHI_response"].tolist()
            
            scores = get_coherence_score_batch(adult_utterances, child_utterances, model=model_name)
            if type(scores) == dict:
                #convert keys to string
                scores = {str(k): v for k, v in scores.items()}


            if scores is not None:
                trys = 0
                rate_limit.append(current_time)
                
                for j, (_, row) in enumerate(batch_df.iterrows()):
                    score_key = str(j)
                    if score_key in scores:
                        if score_key in scores:
                          adult_coherence_score = None
                          if "adult_utterance_coherence" in scores[score_key]:
                              adult_coherence_score = scores[score_key]["adult_utterance_coherence"]
                          #get the first key with the word adult in it
                          else:
                              for key in scores[score_key]:
                                  if "adult" in key:
                                      adult_coherence_score = scores[score_key][key]
                                      break
                          child_coherence_score = None
                          if "child_responce_coherence" in scores[score_key]:
                              child_coherence_score = scores[score_key]["child_responce_coherence"]
                          #get the first key with the word child in it
                          else:
                              for key in scores[score_key]:
                                  if "child" in key:
                                      child_coherence_score = scores[score_key][key]
                                      break
                                  

                          df_dataset = pd.concat([df_dataset, pd.DataFrame([{
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
                          }])])
                        else:
                            print(f"Missing coherence_score for item {j} in batch")
                    else:
                        print(f"Missing score for index {j} in batch")
                
                # Save after each batch
                df_dataset.to_csv("child_response_pairs_scored.csv", index=False)
            else:
                df_dataset.to_csv("child_response_pairs_scored.csv", index=False)
                trys += 1
                if trys > 3:
                    print("Failed too many times, exiting")
                    break
                print(f"Failed to get scores for batch {i//batch_size + 1}")
    
    df_dataset.to_csv("child_response_pairs_scored.csv", index=False)
    return df_dataset


# Main execution code
import pandas as pd

# Replace the bottom section with this code
df = pd.read_csv("child_response_pairs.csv")

df = df[(df["text"].str.strip().str.endswith("?")) & (df["text"].str.len() > 20) & (df["text"].str.len() < 1000)]
df = df[(~df["CHI_response"].str.contains("yeah.")) & (df["CHI_response"].str.len() < 1000)]

df = df.sample(600)

# Set your parameters here
batch_size = 16  # Set to higher value (e.g., 5 or 10) for batch processing
model_name = "llama-3.2-90b-vision-preview"  # Change to different model if needed

df_result = process_data(df, batch_size=batch_size, model_name=model_name)
print(f"Processed {len(df_result)} records successfully")