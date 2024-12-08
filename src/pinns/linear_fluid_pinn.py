import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Define the Physics-Informed Neural Network
class PiNN(nn.Module):
    def __init__(self):
        super(PiNN, self).__init__()
        self.fc1 = nn.Linear(1, 20)  # Input layer
        self.fc2 = nn.Linear(20, 20)  # Hidden layer
        self.fc3 = nn.Linear(20, 1)  # Output layer

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

# Physics-Informed Loss Function
def physics_informed_loss(model, x, U=1.0):
    # Forward pass to get u(x)
    u_pred = model(x)
    
    # Compute the first derivative of u(x) with respect to x
    u_x = torch.autograd.grad(u_pred, x, torch.ones_like(u_pred), create_graph=True)[0]
    
    # Compute the second derivative of u(x) with respect to x
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    
    # Physics-informed loss: the residual of the governing equation
    physics_loss = torch.mean(u_xx**2)
    
    # Boundary condition loss
    bc_left = model(torch.tensor([[0.0]]))  # u(0) = 0
    bc_right = model(torch.tensor([[1.0]])) - U  # u(1) = U
    bc_loss = torch.mean(bc_left**2) + torch.mean(bc_right**2)
    
    # Total loss
    return physics_loss + bc_loss

# Training the PiNN
def train_pinn(model, optimizer, epochs=500, U=1.0):
    # Random sampling points in the domain [0, 1]
    x_train = torch.tensor(np.random.rand(100, 1), dtype=torch.float32, requires_grad=True)

    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        optimizer.zero_grad()
        loss = physics_informed_loss(model, x_train, U=U)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Visualization
def evaluate_and_plot(model, U=1.0):
    # Test points in the domain [0, 1]
    x_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    with torch.no_grad():
        u_pred = model(x_test).numpy()
    
    # Analytical solution
    u_exact = U * x_test.numpy()

    # Plotting the predicted vs analytical solution
    plt.figure(figsize=(8, 5))
    plt.plot(x_test.numpy(), u_pred, label="Predicted u(x)", color="blue")
    plt.plot(x_test.numpy(), u_exact, 'r--', label="Analytical u(x)", color="red")
    plt.xlabel("Position x")
    plt.ylabel("Velocity u(x)")
    plt.title("Velocity Profile in a 1D Pipe")
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    print("Training a Physics-Informed Neural Network (PiNN) for 1D Fluid Flow")
    # Initialize the PiNN model and optimizer
    model = PiNN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_pinn(model, optimizer, epochs=500, U=1.0)

    # Evaluate and plot the results
    evaluate_and_plot(model, U=1.0)