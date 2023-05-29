import torch
import numpy as np
import pickle
from transformers import AutoTokenizer

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
model_path = "model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Function to preprocess input text
def preprocess_text(text):
    text = text.lower().strip()
    return text

# Function to predict label
def predict_label(text):
    # Preprocess text
    text = preprocess_text(text)
    # Tokenize input
    inputs = tokenizer.encode_plus(text, padding=True, truncation=True, max_length=100, return_tensors="pt")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    # Make prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = np.argmax(logits, axis=1).flatten()
    # Map predicted label to "spam" or "không spam"
    label_map = {0: "không spam", 1: "spam"}
    predicted_label = label_map[predicted_labels.item()]
    # Return predicted label
    return predicted_label

# Interactive loop for user input
while True:
    user_input = input("Nhập bình luận (hoặc 'q' để thoát): ")
    if user_input == "q":
        break
    predicted_label = predict_label(user_input)
    print("Bình luận được dự đoán là:", predicted_label)
