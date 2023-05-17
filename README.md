# chatbotgpt2
I have created a chatbot using GPT-2 model . The data for training was taken from a whatsapp conversation.

import numpy as np

import pandas as pd

import re

from nltk.tokenize import word_tokenize

from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextGenerationPipeline, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

import torch

import os

def preprocess_chat_data(chat_data):

    preprocessed_data = []
    
    for message in chat_data:
    
        cleaned_message = clean_text(message)    
        
        tokens = tokenize_message(cleaned_message)
        
        preprocessed_data.append(tokens)
    
    return preprocessed_data

def tokenize_message(message):
    tokens = word_tokenize(message)
    return tokens

def clean_text(message):

    cleaned_message = re.sub(r'http\S+|www\S+|https\S+', '', message)  # Remove URLs
    
    cleaned_message = re.sub(r'[^\w\s]|_', '', cleaned_message)  # Remove special characters except spaces and underscores
    
    cleaned_message = re.sub(r'\s+', ' ', cleaned_message)  # Remove extra whitespaces
    
    cleaned_message = re.sub(r'\d{6} \d{4}', '', cleaned_message)  # Remove timestamp (e.g., 040221 1003)
    
    cleaned_message = re.sub(r'\d{6}', '', cleaned_message)  # Remove timestamp without user identifier (e.g., 040221)
    
    cleaned_message = re.sub(r'<USER_IDENTIFIER>', '', cleaned_message)  # Remove user identifier
    

    return cleaned_message

file_path = "C:\\Users\\HP\\Desktop\\chatting.txt" 

with open(file_path, 'r', encoding='utf-8') as file:

    chat_data = file.readlines()

preprocessed_data = preprocess_chat_data(chat_data)

for message_tokens in preprocessed_data:

    print(message_tokens)

user1 = 'Manveen'

user2 = 'Raju'

user1_messages = []

user2_messages = []

current_user = None

current_messages = []

for chat in preprocessed_data:

    if len(chat) >= 2:
    
        user = chat[0]
        
        message = chat[1:]
        
        if user == user1:
        
            if current_user != user1:
            
                if current_messages:
                
                    user2_messages.append(current_messages)
                    
                    current_messages = []
                    
                current_user = user1
                
        elif user == user2:
        
            if current_user != user2:
            
                if current_messages:
                
                    user1_messages.append(current_messages)
                    
                
                    current_messages = []
                    
                current_user = user2

        current_messages.extend(message)

# Append the last set of messages

if current_user == user1 and current_messages:

    user1_messages.append(current_messages)
    
elif current_user == user2 and current_messages:

    user2_messages.append(current_messages)

# Print the messages

print("User 1 messages:")

for messages in user1_messages:

    print(messages)

print("User 2 messages:")

for messages in user2_messages:

    print(messages)

combined_messages = []

for i in range(len(user1_messages)):

    user1_message = user1_messages[i]
    
    user2_message = user2_messages[i]
    
    combined_messages.append((user1_message, user2_message))

for pair in combined_messages:

    user1_message = ' '.join(pair[0])
    
    user2_message = ' '.join(pair[1])
    
    print(f"User 1: {user1_message}")
    
    print(f"User 2: {user2_message}")
    
    print()

model_name = "gpt2"

model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer = GPT2Tokenizer.from_pretrained(model_name)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

train_data = []

for user1_messages, user2_messages in zip(user1_messages, user2_messages):

    conversation = " ".join(user1_messages) + " " + " ".join(user2_messages)
    
    train_data.append(conversation)

input_ids = []

attention_masks = []

for conversation in train_data:

    encoded_inputs = tokenizer.encode_plus(conversation, add_special_tokens=True, truncation=True)
    
    input_ids.append(encoded_inputs["input_ids"])
    
    attention_masks.append(encoded_inputs["attention_mask"])

max_length = max(len(ids) for ids in input_ids)

input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]

attention_masks = [masks + [0] * (max_length - len(masks)) for masks in attention_masks]

input_ids = torch.tensor(input_ids)

attention_masks = torch.tensor(attention_masks)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)

model.train()

batch_size = 4
num_epochs = 3
learning_rate = 1e-4

# Set up the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

from transformers import TextDataset, DataCollatorForLanguageModeling

train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=file_path,
    block_size=128  # Adjust the block size as needed
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    learning_rate=learning_rate,
    logging_steps=500,
    save_steps=1000
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset
)

trainer.train()




output_dir = os.path.join(os.path.expanduser("~"), "Desktop", "trained_model")
os.makedirs(output_dir, exist_ok=True)

# Save the trained model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Load the trained model
model = GPT2LMHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

# Set the model to evaluation mode
model.eval()

def generate_response(user_input):
    input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids).to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=30,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,  # Adjust the temperature value
        num_beams=30,  # Adjust the number of beams for beam search
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

# Generate a response given a user input
user_input = "Hello, how are you?"

input_ids = tokenizer.encode(user_input, add_special_tokens=True, return_tensors="pt").to(device)

attention_mask = torch.ones_like(input_ids).to(device)

# Generate the output with attention mask
output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=23, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0], skip_special_tokens=True)

print("Model response:", response)

print("Model initialized. Enter 'exit' to quit.")

while True:

    user_input = input("You: ")
    
    if user_input.lower() == "exit":
    
        break

    response = generate_response(user_input)
    
    print("Model: " + response)
