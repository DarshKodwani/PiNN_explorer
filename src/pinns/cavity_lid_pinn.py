import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

BASE_DIR = os.getenv('BASE_DIR')
# Load data from CSV
data = pd.read_csv(os.path.join(BASE_DIR, 'simulation_outputs/moving_lid_simulation/flow_data.csv'))
x_train = torch.tensor(data['x'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
y_train = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
t_train = torch.tensor(data['t'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
u_train = torch.tensor(data['u'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)
v_train = torch.tensor(data['v'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)    
p_train = torch.tensor(data['p'].values, dtype=torch.float32).view(-1, 1).requires_grad_(True)

# Define the neural network model
class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        for i in range(len(layers) - 2):
            self.hidden_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        
    def forward(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        for layer in self.hidden_layers:
            inputs = torch.tanh(layer(inputs))
        outputs = self.output_layer(inputs)
        return outputs

# Function to compute the physics-informed loss (Navier-Stokes equations)
def compute_loss(model, x, y, t, u_data, v_data, p_data):
    u_pred, v_pred, p_pred = model(x, y, t).split(1, dim=1)
    
    # Compute gradients for the Navier-Stokes equations
    u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    u_t = torch.autograd.grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
    
    v_x = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    v_y = torch.autograd.grad(v_pred, y, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    v_t = torch.autograd.grad(v_pred, t, grad_outputs=torch.ones_like(v_pred), create_graph=True)[0]
    
    p_x = torch.autograd.grad(p_pred, x, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]
    p_y = torch.autograd.grad(p_pred, y, grad_outputs=torch.ones_like(p_pred), create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]
    
    # Define the Navier-Stokes equation residuals
    nu = 0.01  # Kinematic viscosity
    continuity_residual = u_x + v_y
    momentum_x_residual = u_t + (u_pred * u_x + v_pred * u_y) + p_x - nu * (u_xx + u_yy)
    momentum_y_residual = v_t + (u_pred * v_x + v_pred * v_y) + p_y - nu * (v_xx + v_yy)

    # Data loss
    data_loss = torch.mean((u_pred - u_data) ** 2) + torch.mean((v_pred - v_data) ** 2) + torch.mean((p_pred - p_data) ** 2)
    
    # Physics loss
    physics_loss = torch.mean(continuity_residual ** 2) + torch.mean(momentum_x_residual ** 2) + torch.mean(momentum_y_residual ** 2)
    
    # Total loss
    total_loss = data_loss + physics_loss
    
    return total_loss

# Initialize the model
layers = [3, 50, 50, 50, 3]  # Input layer size is 3 (x, y, t), output layer size is 3 (u, v, p)
model = PINN(layers)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 10000
for epoch in range(epochs):
    optimizer.zero_grad()
    
    loss = compute_loss(model, x_train, y_train, t_train, u_train, v_train, p_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training complete.")