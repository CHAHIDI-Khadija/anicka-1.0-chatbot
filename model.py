#import requiered packages/librairies
import torch.nn as nn

class NeuralNet(nn.Module):
    """
        constructor, layers declaration
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden_size) #input layer
        self.layer2 = nn.Linear(hidden_size,hidden_size) #hidden layer
        self.layer3 = nn.Linear(hidden_size,num_classes) #output layer
        self.relu = nn.ReLU() #Rectified Linear Unit function (max(x,0))  (activation function)

    def forward(self,x):
        out = self.layer1(x)
        out = self.relu(out) 
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        # no activation and no softmax at the end
        return out