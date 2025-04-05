import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
device = "cuda" if torch.cuda.is_available() else "cpu"



score_column = "adult_coherence_score"

# Load and preprocess the data
df = pd.read_csv('../data/child_response_pairs_scored.csv')
# Attempt to convert score to float, filter valid scores
df[score_column] = pd.to_numeric(df[score_column], errors='coerce')
df[score_column] = pd.to_numeric(df[score_column], errors='coerce')


df = df.dropna(subset=[score_column])
df[score_column] = df[score_column].clip(0, 1)

#if CHI_response responce contains "yeah." reduce score by 80% of original score
df[score_column] = df.apply(lambda row: row[score_column] * 0.87 if " " not in row['CHI_response'] else row[score_column], axis=1)

#slightly reward texts with longer responces between 5-20% higher score, but top at 1
df[score_column] = df.apply(lambda row: min(1, row[score_column] + (len(row['CHI_response']) - 30) * 0.001), axis=1)


df = df.drop_duplicates(subset=['text', 'CHI_response']).reset_index(drop=True)

#replace  \[\^ [a-z]*\] with ""
df['text'] = df['text'].str.replace(r'\[\^ [a-z]*\]', '', regex=True)
df['CHI_response'] = df['CHI_response'].str.replace(r'\[\^ [a-z]*\]', '', regex=True)



# Split the dataset into train and test sets

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
train_df = df#to maximize accuracy

# Prepare training and evaluation samples with text and CHI_response
train_samples = [
    InputExample(texts=[row['text']], label=float(row[score_column]))
    for _, row in train_df.iterrows()
]
eval_samples = [
    InputExample(texts=[row['text']], label=float(row[score_column]))
    for _, row in test_df.iterrows()
]

binary_eval_samples = [
    InputExample(texts=[row['text']], label=1 if row[score_column] > 0.5 else 0)
    for _, row in test_df.iterrows()
]


# Create data loaders
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=256)
evaluator_dataset = DataLoader(eval_samples, shuffle=False, batch_size=256)
evaluator_dataset_2 = DataLoader(binary_eval_samples, shuffle=False, batch_size=256)

# Initialize evaluator
evaluatr_2 = CEBinaryClassificationEvaluator.from_input_examples(examples=binary_eval_samples, write_csv=False, show_progress_bar=False)

# Initialize model
automodel_args = {}#{"torch_dtype": torch.bfloat16}
tokenizer_args = {"max_length": 96, "padding": "max_length", "truncation": True}

# Set training parameters
num_epochs = 4
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.05)
evaluation_steps = math.ceil(len(train_dataloader)) * 3
print(warmup_steps, evaluation_steps)

# Define callback function
def callback_func(score, epoch, steps):
    print(f"Score: {score} at epoch {epoch}")

model = CrossEncoder('google-bert/bert-base-uncased',
                    num_labels=1,
                    automodel_args=automodel_args,
                    tokenizer_args=tokenizer_args, 
                    max_length=96,
                    classifier_dropout=0.1)   

# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluatr_2,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    callback=callback_func,
    output_path='../models/adult_coherency_model',
    save_best_model=True,
    evaluation_steps=evaluation_steps)


score2 = evaluatr_2(model, evaluator_dataset_2)
print(f"Final evaluation score: {score2}")


del model
del train_dataloader
del evaluator_dataset
torch.cuda.empty_cache()

model = CrossEncoder('../models/adult_coherency_model', max_length=96)
df = pd.read_csv('../data/child_response_pairs.csv')
texts = [[row['text']] for _, row in df.iterrows()]


predictions = model.predict(texts, batch_size=1024, show_progress_bar=True)
df[f"{score_column}_prediction"] = predictions
df.to_csv('../data/child_response_pairs.csv', index=False)

#print top 10 predictions sorted by score
df_display = df.sort_values(by=f"{score_column}_prediction", ascending=False).head(15)


for index, row in df_display.iterrows():
    prediction = row[f"{score_column}_prediction"]
    print(f"{row['text']}-->>\t{prediction}")

print('done')
