#import the necessary packages
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

#the device on which a torch tensor will be allocated
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#load data
with open("intents.json", 'r', encoding="utf-8") as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

#initialization of necessary information
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

#model initialization
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

"""
    The name of our bot
"""
bot_name = "ANICKA"

"""
    function that return the possible answer of the sentence
"""
def get_response(sentence):
    t = True
    while t:
        #call methodes pre-deffined
        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        #check probabilities and the possible sentence's tag 
        #to choose a random answer from the responses part
        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    return(f"{random.choice(intent['responses'])}")
                    if tag == "goodbye":
                        t=False
        #if the sentence does not exist in any tag. it will display
        else:
            return "I do not understand, what do you mean?"