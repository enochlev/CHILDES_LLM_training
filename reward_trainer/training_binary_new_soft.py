import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import InputExample, CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator, CEBinaryAccuracyEvaluator

print(sys.executable)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Load and preprocess the data
df = pd.read_csv('../data/child_full.csv')
df = df[df["speaker"] == "CHI"]

df['age'] = df['years'] + df["months"] / 12
df.rename({'CHI_response': 'text'}, axis=1, inplace=True)
df = df[(df.age <= 6.99) & (df.age >= 1.0)]
max_age = df['age'].max()
min_age = df['age'].min()

# Define classification focus
focus = 2.99

# Split the dataset into train and test sets
train_df = df.sample(frac=0.85, random_state=42)
test_df = df.drop(train_df.index)

# Apply soft labels to training data
def normalize_label(age, focus, min_age, max_age):
    if age <= focus:
        # Normalize ages below focus between 1.0 and 0.5
        # min_age -> 1.0, focus -> 0.5
        return 1.0 - ((age - min_age) / (focus - min_age)) * 0.25
    else:
        # Normalize ages above focus between 0.5 and 0.0
        # focus -> 0.5, max_age -> 0.0
        return 0.25 - ((age - focus) / (max_age - focus)) * 0.25

# Apply soft labels to training data
train_df['label'] = train_df['age'].apply(lambda x: normalize_label(x, focus, min_age, max_age))

# Apply binary labels to test data
test_df['label'] = test_df['age'].apply(lambda x: 1.0 if x <= focus else 0.0)

# Filter out very short responses
train_df = train_df[train_df['text'].str.strip(".").str.len() >= 2]
test_df = test_df[test_df['text'].str.strip(".").str.len() >= 2]

# Prepare training and evaluation samples
train_samples = [InputExample(texts=[row['text']], label=row['label']) for _, row in train_df.iterrows()]
eval_samples = [InputExample(texts=[row['text']], label=row['label']) for _, row in test_df.iterrows()]

# Create data loaders
BATCH_SIZE = 2048
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
evaluator_dataset = DataLoader(eval_samples, shuffle=False, batch_size=BATCH_SIZE)

# Initialize evaluator
evaluator = CEBinaryClassificationEvaluator.from_input_examples(examples=eval_samples, write_csv=False, show_progress_bar=False)

# Initialize model
automodel_args = {}#{"torch_dtype": torch.bfloat16}
tokenizer_args = {"max_length": 32, "padding": "max_length", "truncation": True}



# Set training parameters

warmup_steps = math.ceil(len(train_dataloader) * 0.05)
evaluation_steps = math.ceil(len(train_dataloader)) *3
print(warmup_steps, evaluation_steps)

# Define callback function
def callback_func(score, epoch, steps):
    print(f"Score: {score} at epoch {epoch}")

#sentence-transformers/all-mpnet-base-v2 has a bit higher accuracy then google-bert/bert-base-uncased because its head is larger at initialization.
model = CrossEncoder('google-bert/bert-base-uncased', num_labels=1, automodel_args=automodel_args,tokenizer_args=tokenizer_args, max_length=32,classifier_dropout=0.15)
#model = CrossEncoder('sentence-transformers/all-mpnet-base-v2', num_labels=1, automodel_args=automodel_args,tokenizer_args=tokenizer_args, max_length=32,classifier_dropout=0.15)   

#2000 steps is needed
num_epochs = 3000 // len(train_dataloader)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.05)
# Train the model
model.fit(
    train_dataloader=train_dataloader,
    evaluator=evaluator,
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    output_path="../models/childish_reward_model",
    callback=callback_func,
    show_progress_bar=True,
    save_best_model=True,
    evaluation_steps=evaluation_steps)

# Evaluate the model
score = evaluator(model, evaluator_dataset)
print(f"Final evaluation score: {score}")

