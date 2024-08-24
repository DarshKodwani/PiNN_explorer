# Import necessary libraries
import torch
import torch.nn as nn
import os
from dotenv import load_dotenv

load_dotenv('.env')

# Define the model architecture (same as the one used for training)
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        
    def forward(self, t):
        inputs = t
        for layer in self.hidden_layers:
            inputs = torch.tanh(layer(inputs))
        outputs = self.output_layer(inputs)
        return outputs

# Instantiate the model with the correct architecture
layers = [3, 50, 50, 50, 3]
model = PINN(layers)

# Load the state dictionary into the model
model.load_state_dict(torch.load(os.path.join(os.getenv('BASE_DIR'),'pinn_model.pth')))

# Set the model to evaluation mode
model.eval()