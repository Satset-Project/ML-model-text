import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Define the model
from keras.initializers import Orthogonal
from tensorflow.keras.models import load_model
import pandas as pd

categories = {
    "electricity": {
        "keywords": ["light", "bulb", "switch", "electrical", "wire"],
        "actions": ["Installation", "Repair", "Maintenance"]
    },
    "air_conditioner": {
        "keywords": ["air conditioner", "ac", "cooling", "hvac"],
        "actions": ["Installation", "Repair", "Maintenance"]
    },
    "plumbing": {
        "keywords": ["faucet", "leak", "pipe", "drip", "plumbing"],
        "actions": ["Installation", "Repair", "Maintenance"]
    },
    "computer": {
        "keywords": ["computer", "pc", "laptop", "software", "hardware"],
        "actions": ["Installation", "Repair", "Maintenance"]
    },
    "washer": {
        "keywords": ["washer", "washing machine", "laundry", "spin", "drain"],
        "actions": ["Installation", "Repair", "Maintenance"]
    },
    "refrigerator": {
        "keywords": ["refrigerator", "fridge", "cooling", "ice", "defrost"],
        "actions": ["Installation", "Repair", "Maintenance"]
    },
}

# Step 2: Define solutions
solutions = {
    "plumbing": "Plumbing {action}",
    "electricity": "Electrical {action}",
    "air_conditioner": "Air Conditioner {action}",
    "computer": "Computer {action}",
    "washer": "Washer {action}",
    "refrigerator": "Refrigerator {action}",
}

response_template = "It seems like you're experiencing a {problem}, which can lead to {issue}. Our recommended solution for this issue is {solution} to {action}."

# Sample data
data = {
    "description": [
        "Broken faucet, causing continuous dripping and wasting water. Needs fixing to stop the leak and restore proper functionality.",
        "The air conditioner is making strange noises and not cooling the room properly. Needs to be checked and repaired.",
        "The washing machine is not draining water after the cycle. Needs immediate repair to prevent water damage.",
        "The computer won't start up and displays a blue screen error. Requires diagnosis and fixing.",
        "The refrigerator is not keeping food cold and the freezer is defrosting. Needs urgent repair to restore proper cooling.",
        "A light switch is not working and the lights won't turn on. Needs to be inspected and fixed.",
        "The bathroom sink is clogged and water is not draining properly. Needs to be unclogged and restored to normal function.",
        "The laptop's screen is flickering and showing distorted colors. Needs to be repaired or replaced.",
        "The ceiling fan is wobbling and making a rattling sound. Needs to be checked for loose parts and fixed.",
        "The dishwasher is leaking water onto the kitchen floor. Needs to be inspected and repaired to prevent further leaks.",
        "The air conditioner is blowing warm air instead of cool. Needs to be serviced and refilled with coolant."
    ],
    "category": [
        "plumbing",
        "air_conditioner",
        "washer",
        "computer",
        "refrigerator",
        "electricity",
        "plumbing",
        "computer",
        "electricity",
        "washer",
        "air_conditioner"
    ],
    "action": [
        "Repair",
        "Repair",
        "Repair",
        "Repair",
        "Repair",
        "Repair",
        "Maintenance",
        "Repair",
        "Maintenance",
        "Repair",
        "Maintenance"
    ]
}

df = pd.DataFrame(data)

# Prepare the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['description'])

"""Generate Responses"""

# Define the function to generate responses
def generate_response(description):
    # Preprocess the input description
    sequences = tokenizer.texts_to_sequences([description])
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

    custom_objects = {
        'Orthogonal': Orthogonal
    }

    model = load_model('issue_classification_model.h5', custom_objects=custom_objects)
    
    # Predict category and action
    category_pred, action_pred = model.predict(padded_sequence)

    # Get the predicted category and action
    category = df['category'].unique()[category_pred.argmax()]
    action = df['action'].unique()[action_pred.argmax()]

    problem = "an issue with your " + category.replace('_', ' ')
    issue = "an issue related to your " + category.replace('_', ' ')
    solution = solutions[category].format(action=action)

    response = response_template.format(
        problem=problem,
        issue=issue,
        solution=solution,
        action=action
    )
    return response

# Test the function
description = input("Please describe your issue: ")
response = generate_response(description)
print(response)