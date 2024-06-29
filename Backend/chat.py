import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the intents and the trained model
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "model.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Buddy"

def chat_with_bot(sentence: str):
    """
    Function to get a response from the chatbot given an input sentence.
    
    Args:
        sentence (str): The input sentence from the user.
    
    Returns:
        str: The chatbot's response.
    """
    if sentence == "/bye":
        return "Goodbye!"

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand..."

if __name__ == "__main__":
    print("Let's chat! (type '/bye' to exit)")
    while True:
        sentence = input("You: ")
        response = chat_with_bot(sentence)
        print(f"{bot_name}: {response}")
        if sentence == "/bye":
            break
