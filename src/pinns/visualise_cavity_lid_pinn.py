import torch
import torch.nn as nn
import os
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt

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

# Load the data from the CSV file
data = pd.read_csv(os.path.join(os.getenv('BASE_DIR'), 'simulation_outputs/moving_lid_simulation/flow_data.csv'))
x_train = torch.tensor(data['x'].values, dtype=torch.float32).view(-1, 1)
y_train = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)
t_train = torch.tensor(data['t'].values, dtype=torch.float32).view(-1, 1)
u_train = torch.tensor(data['u'].values, dtype=torch.float32).view(-1, 1)
v_train = torch.tensor(data['v'].values, dtype=torch.float32).view(-1, 1)
p_train = torch.tensor(data['p'].values, dtype=torch.float32).view(-1, 1)

# Predict values using the model
with torch.no_grad():
    u_pred, v_pred, p_pred = model(torch.cat([x_train, y_train, t_train], dim=1)).split(1, dim=1)

# Convert tensors to numpy arrays for plotting
t_np = t_train.cpu().numpy()
x_np = x_train.cpu().numpy()
y_np = y_train.cpu().numpy()
u_data_np = u_train.cpu().numpy()
v_data_np = v_train.cpu().numpy()
p_data_np = p_train.cpu().numpy()
u_pred_np = u_pred.cpu().numpy()
v_pred_np = v_pred.cpu().numpy()
p_pred_np = p_pred.cpu().numpy()

# Calculate the percentage differences
u_diff = ((u_train - u_pred) / u_train) * 100
v_diff = ((v_train - v_pred) / v_train) * 100
p_diff = ((p_train - p_pred) / p_train) * 100

u_diff_np = u_diff.cpu().numpy()
v_diff_np = v_diff.cpu().numpy()
p_diff_np = p_diff.cpu().numpy()

# Plot the results
fig, axs = plt.subplots(3, 3, figsize=(18, 18))

# u difference vs x
axs[0, 0].scatter(x_np, u_diff_np, c='blue', label='u difference')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('u difference (%)')
axs[0, 0].set_title('u difference vs x')
axs[0, 0].set_yscale('log')

# u difference vs y
axs[0, 1].scatter(y_np, u_diff_np, c='blue', label='u difference')
axs[0, 1].set_xlabel('y')
axs[0, 1].set_ylabel('u difference (%)')
axs[0, 1].set_title('u difference vs y')
axs[0, 1].set_yscale('log')

# u difference vs t
axs[0, 2].scatter(t_np, u_diff_np, c='blue', label='u difference')
axs[0, 2].set_xlabel('t')
axs[0, 2].set_ylabel('u difference (%)')
axs[0, 2].set_title('u difference vs t')
axs[0, 2].set_yscale('log')

# v difference vs x
axs[1, 0].scatter(x_np, v_diff_np, c='green', label='v difference')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('v difference (%)')
axs[1, 0].set_title('v difference vs x')
axs[1, 0].set_yscale('log')

# v difference vs y
axs[1, 1].scatter(y_np, v_diff_np, c='green', label='v difference')
axs[1, 1].set_xlabel('y')
axs[1, 1].set_ylabel('v difference (%)')
axs[1, 1].set_title('v difference vs y')
axs[1, 1].set_yscale('log')

# v difference vs t
axs[1, 2].scatter(t_np, v_diff_np, c='green', label='v difference')
axs[1, 2].set_xlabel('t')
axs[1, 2].set_ylabel('v difference (%)')
axs[1, 2].set_title('v difference vs t')
axs[1, 2].set_yscale('log')

# p difference vs x
axs[2, 0].scatter(x_np, p_diff_np, c='red', label='p difference')
axs[2, 0].set_xlabel('x')
axs[2, 0].set_ylabel('p difference (%)')
axs[2, 0].set_title('p difference vs x')
axs[2, 0].set_yscale('log')

# p difference vs y
axs[2, 1].scatter(y_np, p_diff_np, c='red', label='p difference')
axs[2, 1].set_xlabel('y')
axs[2, 1].set_ylabel('p difference (%)')
axs[2, 1].set_title('p difference vs y')
axs[2, 1].set_yscale('log')

# p difference vs t
axs[2, 2].scatter(t_np, p_diff_np, c='red', label='p difference')
axs[2, 2].set_xlabel('t')
axs[2, 2].set_ylabel('p difference (%)')
axs[2, 2].set_title('p difference vs t')
axs[2, 2].set_yscale('log')

plt.tight_layout()
plt.show()