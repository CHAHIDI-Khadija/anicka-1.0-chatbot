#import necessary packages
import json
from nltk_utils import tokenize, stem, bag_of_words
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet 

#load the json file
with open('intents.json', 'r', encoding="utf-8") as f:
    intents = json.load(f)

#create training data
"""
    in these part we will create the training data:
    - all_words is the object that will reserve all existing words in the pattern part ('questions') of our dataset 
    - tag object will reseve tags of our data input
    - xy object will contains tuples of toknized patterns and thier tag
"""
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

#lower and stem each word
"""
    in this part of code, we will delete ignore words from the all_words variable,
    use the stem method to normalizing and lowercasing the words.
    the sorte step is optionnel
"""
ignore_words = ['!', '?', ',', '.', '-']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

#bag of words, Xtrain and y_train
"""
    preparing the bag of words and the train data
"""
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

# class that makes it easier to work with the train data
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train
    
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

#load train data
dataset = ChatDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#the device on which a torch tensor will be allocated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#call the model
model = NeuralNet(input_size, hidden_size, output_size)

#loss and optimizer
criterion = nn.CrossEntropyLoss() #defien Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #define optimizer

#train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        #forward pass
        outputs = model(words) #pass data to model
        loss = criterion(outputs, labels) #forward

        #backward and optimizer step
        optimizer.zero_grad() #reset Gradient
        loss.backward() #update weights
        optimizer.step() #update parameters

#save the necessary information that we will use in the chat into a .pth file
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)