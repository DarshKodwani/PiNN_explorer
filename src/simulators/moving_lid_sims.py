import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Domain size and grid resolution
Lx, Ly = 1.0, 1.0
nx, ny = 41, 41
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
nt = 500  # Number of time steps
dt = 0.001  # Time step size

# Fluid properties
rho = 1.0  # Density
nu = 0.1  # Kinematic viscosity

# Initialize velocity and pressure fields
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Lid velocity
lid_velocity = 1.0
u[-1, :] = lid_velocity

# Functions for pressure Poisson equation and velocity update
def build_up_b(b, rho, dt, u, v, dx, dy):
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) + 
                 (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                  2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                       (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx))-
                       ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(nt):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
        
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
        
    return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    # Initialize lists to store data
    cavity_flow_data = {
        't': [],
        'x': [],
        'y': [],
        'u': [],
        'v': [],
        'p': []
    }
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny, nx))

    for n in range(nt):
        un = u.copy()
        vn = v.copy()
        
        b = build_up_b(b, rho, dt, un, vn, dx, dy)
        p = pressure_poisson(p, dx, dy, b)
        
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * 
                        (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                         dt / (2 * rho * dx) * 
                        (p[1:-1, 2:] - p[1:-1, :-2]) +
                         nu * (dt / dx**2 * 
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                         dt / dy**2 * 
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))
        
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                         un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                         vn[1:-1, 1:-1] * dt / dy * 
                        (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                         dt / (2 * rho * dy) * 
                        (p[2:, 1:-1] - p[:-2, 1:-1]) +
                         nu * (dt / dx**2 * 
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                         dt / dy**2 * 
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))
        
        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = lid_velocity
        v[0, :] = 0
        v[-1, :] = 0
        v[:, 0] = 0
        v[:, -1] = 0
        # Store the data at each time step
        for i in range(ny):
            for j in range(nx):
                cavity_flow_data['t'].append(n * dt)
                cavity_flow_data['x'].append(j * dx)
                cavity_flow_data['y'].append(i * dy)
                cavity_flow_data['u'].append(u[i, j])
                cavity_flow_data['v'].append(v[i, j])
                cavity_flow_data['p'].append(p[i, j])
    return u, v, p, cavity_flow_data



# Run the simulation
u, v, p, cavity_flow_data = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

# Convert the data to a Pandas DataFrame
df = pd.DataFrame(cavity_flow_data)

# Save the DataFrame to a CSV file
df.to_csv('../../simulation_data/cavity_flow_data.csv', index=False)

# Plot the results
plt.figure(figsize=(11, 7), dpi=100)

# Contour plot for pressure field
contourf = plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, alpha=0.5, cmap=plt.cm.viridis)
plt.colorbar(contourf, label='Pressure')

# Contour lines for pressure field
contour = plt.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, cmap=plt.cm.viridis)
plt.clabel(contour, inline=1, fontsize=10)

# Quiver plot for velocity field
quiver = plt.quiver(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), u, v, color='r', label='Velocity vectors')

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cavity Flow Velocity Field')

# Legend
plt.legend()

# Save the plot to a file
plt.savefig('../../simulation_data/cavity_flow_plot.png')

# Show plot
plt.show()

# Create a figure and axis for the animation
fig, ax = plt.subplots(figsize=(11, 7), dpi=100)

# Initialize the contour and quiver plots
contourf = ax.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, alpha=0.5, cmap=plt.cm.viridis)
contour = ax.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, cmap=plt.cm.viridis)
quiver = ax.quiver(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), u, v, color='r')

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Cavity Flow Velocity Field')

# Function to update the plot for each frame
def update(frame):
    ax.clear()
    
    # Filter the data for the current time step
    current_data = df[df['t'] == frame * dt]
    
    # Reshape the data for plotting
    u = current_data.pivot(index='y', columns='x', values='u').values
    v = current_data.pivot(index='y', columns='x', values='v').values
    p = current_data.pivot(index='y', columns='x', values='p').values
    
    # Update the contour and quiver plots
    contourf = ax.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, alpha=0.5, cmap=plt.cm.viridis)
    contour = ax.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, cmap=plt.cm.viridis)
    quiver = ax.quiver(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), u, v, color='r')
    
    # Add an arrow to represent the lid's velocity
    lid_arrow = ax.arrow(0.5, 1.0, 0.4, 0, head_width=0.05, head_length=0.05, fc='k', ec='k')
    
    # Labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Cavity Flow Velocity Field at t={frame * dt:.3f}s')

# Create the animation
anim = FuncAnimation(fig, update, frames=nt, repeat=False)

# Save the animation to a file
anim.save('../../simulation_data/cavity_flow_animation.mp4', writer='ffmpeg')

# Show the plot
plt.show()