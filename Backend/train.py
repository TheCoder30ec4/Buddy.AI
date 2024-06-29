import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet  # Ensure the class name matches exactly

# Guard the main execution
if __name__ == "__main__":
    with open('intents.json', 'r') as file:
        data = json.load(file)

    num_epochs = 400
    batch_size = 16
    learning_rate = 0.001
    all_words = []
    tags = []
    xy = []

    # Tokenizing and stemming the words, preparing the training data
    for intent in data['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ['?', '!', '.', ',']
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    X_train = []
    y_train = []

    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        label = tags.index(tag)
        y_train.append(label)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(X_train)
            self.x_data = X_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    # Creating the dataset and data loader
    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Setting up the device and the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(input_size=len(X_train[0]), hidden_size=8, num_classes=len(tags)).to(device)

    # Defining loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device, dtype=torch.float32)  # Ensure correct dtype
            labels = labels.to(device, dtype=torch.long)  # Ensure correct dtype

            # Forward pass
            outputs = model(words)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Model training completed.')

    # Save the model and metadata
    model_data = {
        "model_state": model.state_dict(),
        "input_size": len(X_train[0]),
        "hidden_size": 8,
        "output_size": len(tags),
        "all_words": all_words,
        "tags": tags
    }

    FILE = "model.pth"
    torch.save(model_data, FILE)
    print(f'Training complete. File saved to {FILE}')
