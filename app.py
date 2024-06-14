from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd

app = FastAPI()

# Load tokenizer and model
tokenizer = None
model = None
max_sequence_length = None
df = None

class RequestBody(BaseModel):
    description: str

# Function to load tokenizer and model
def load_model_and_tokenizer():
    global tokenizer, model, max_sequence_length, df
    
    # Load DataFrame and tokenizer
    data = pd.read_excel('/content/training.xlsx')
    df = data
    
    # Prepare the tokenizer
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(df['Sentence'])
    
    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences(df['Sentence'])
    max_sequence_length = max(len(seq) for seq in sequences)
    
    # Load the model
    model_path = "issue_classification_model.h5"
    model = tf.keras.models.load_model(model_path)

# Initialize model and tokenizer when the app starts
load_model_and_tokenizer()

# Prediction endpoint
@app.post("/chatbox")
async def predict(item: RequestBody):
    global tokenizer, model, max_sequence_length, df
    
    description = item.description
    
    # Preprocess the input description
    description_sequence = tokenizer.texts_to_sequences([description])
    description_padded = pad_sequences(description_sequence, maxlen=max_sequence_length, padding='post')

    # Predict using the loaded model
    category_prob, solution_prob = model.predict(description_padded)

    # Get the predicted category and solution
    predicted_category = tf.argmax(category_prob, axis=1).numpy()[0]
    predicted_solution = tf.argmax(solution_prob, axis=1).numpy()[0]

    # Map indices back to category and solution labels
    category_label = df['Category'].unique()[predicted_category]
    solution_label = df['Solution'].unique()[predicted_solution]

    # Prepare the response
    response = f"For the issue '{description}', our recommendation is to check '{category_label}' and consider '{solution_label}'."

    return {
        "code": 200,
        "message": "success",
        "response": response
    }
