import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

def pressure_poisson(p, dx, dy, b, nit):
    """
    Solves the pressure Poisson equation for a given pressure field.

    Parameters:
    p (numpy.ndarray): The initial pressure field, a 2D array.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    b (numpy.ndarray): The source term, a 2D array of the same shape as p.
    nit (int): The number of iterations to perform.

    Returns:
    numpy.ndarray: The updated pressure field after solving the Poisson equation.
    """
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                         (2 * (dx**2 + dy**2)) -
                         dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])

        # Boundary conditions for pressure
        p[:, -1] = p[:, -2]  # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]    # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]    # dp/dx = 0 at x = 0
        p[-1, :] = 0         # p = 0 at y = 2
    return p

def velocity_update(u, v, dt, dx, dy, p, rho, nu):
    """
    Updates the velocity fields (u, v) for a fluid simulation based on the 
    Navier-Stokes equations.

    Parameters:
    u (numpy.ndarray): The velocity field in the x-direction, a 2D array.
    v (numpy.ndarray): The velocity field in the y-direction, a 2D array.
    dt (float): The time step size.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    p (numpy.ndarray): The pressure field, a 2D array.
    rho (float): The density of the fluid.
    nu (float): The kinematic viscosity of the fluid.

    Returns:
    tuple: A tuple containing the updated velocity fields (u, v).
    """
    un = u.copy()
    vn = v.copy()
    
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx**2 *
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy**2 *
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx**2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy**2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

    # Boundary conditions for velocity
    u[0, :] = 0
    u[:, 0] = 0
    u[:, -1] = 0
    u[-1, :] = 1  # set velocity on cavity lid equal to 1
    v[0, :] = 0
    v[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0
    
    return u, v

def run_simulation(nt, u, v, dt, dx, dy, p, rho, nu, nit):
    """
    Runs the fluid simulation for a specified number of time steps.

    Parameters:
    nt (int): The number of time steps to run the simulation.
    u (numpy.ndarray): The initial velocity field in the x-direction, a 2D array.
    v (numpy.ndarray): The initial velocity field in the y-direction, a 2D array.
    dt (float): The time step size.
    dx (float): The grid spacing in the x-direction.
    dy (float): The grid spacing in the y-direction.
    p (numpy.ndarray): The initial pressure field, a 2D array.
    rho (float): The density of the fluid.
    nu (float): The kinematic viscosity of the fluid.
    nit (int): The number of iterations for the pressure Poisson equation.

    Returns:
    tuple: A tuple containing the updated velocity fields (u, v) and pressure field (p).
    """
    ny, nx = u.shape
    cavity_flow_data = {'t': [], 'x': [], 'y': [], 'u': [], 'v': [], 'p': []}
    
    for n in range(nt):
        b = np.zeros_like(p)
        b[1:-1, 1:-1] = (rho * (1 / dt *
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                     (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx))**2 -
                    2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                         (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                    ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy))**2))
        
        p = pressure_poisson(p, dx, dy, b, nit)
        u, v = velocity_update(u, v, dt, dx, dy, p, rho, nu)

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

def save_data(cavity_flow_data, file_path):
    """
    Save the cavity flow data to a CSV file.

    Parameters:
    - cavity_flow_data (list): List of dictionaries containing the cavity flow data.
    - file_path (str): The file path where the data will be saved.

    Returns:
    None
    """
    df = pd.DataFrame(cavity_flow_data)
    df.to_csv(file_path, index=False)

def plot_results(p, u, v, Lx, Ly, nx, ny, save_path=None):
    """
    Plot the results of a simulation.
    Parameters:
    - p (numpy.ndarray): Pressure field.
    - u (numpy.ndarray): x-component of velocity field.
    - v (numpy.ndarray): y-component of velocity field.
    - Lx (float): Length of the domain in the x-direction.
    - Ly (float): Length of the domain in the y-direction.
    - nx (int): Number of grid points in the x-direction.
    - ny (int): Number of grid points in the y-direction.
    - save_path (str, optional): Path to save the plot. If not provided, the plot will be displayed.
    Returns:
    None
    """
    plt.figure(figsize=(11, 7), dpi=100)
    
    # Contour plot for pressure field
    contourf = plt.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, alpha=0.5, cmap=plt.cm.viridis)
    plt.colorbar(contourf, label='Pressure')
    
    # Contour lines for pressure field
    contour = plt.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, cmap=plt.cm.viridis)
    plt.clabel(contour, inline=1, fontsize=10)
    
    # Quiver plot for velocity field
    plt.quiver(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), u, v, color='r')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_animation(u, v, p, Lx, Ly, nx, ny, dt, nt, cavity_flow_data, save_path):
    """
    Save an animation of the cavity flow velocity field.
    Parameters:
    - u (numpy.ndarray): Array of x-velocity values.
    - v (numpy.ndarray): Array of y-velocity values.
    - p (numpy.ndarray): Array of pressure values.
    - Lx (float): Length of the x-axis.
    - Ly (float): Length of the y-axis.
    - nx (int): Number of grid points along the x-axis.
    - ny (int): Number of grid points along the y-axis.
    - dt (float): Time step size.
    - nt (int): Number of time steps.
    - cavity_flow_data (pandas.DataFrame): Dataframe containing cavity flow data.
    - save_path (str): Path to save the animation.
    Raises:
    - ValueError: If cavity_flow_data cannot be converted to a DataFrame.
    Returns:
    - None
    """
    # Convert cavity_flow_data to a DataFrame if it is not already one
    if not isinstance(cavity_flow_data, pd.DataFrame):
        try:
            cavity_flow_data = pd.DataFrame(cavity_flow_data)
        except Exception as e:
            raise ValueError(f"Failed to convert cavity_flow_data to DataFrame: {e}")
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=100)
    
    def update(frame):
        ax.clear()
        current_data = cavity_flow_data[cavity_flow_data['t'] == frame * dt]
        
        u = current_data.pivot(index='y', columns='x', values='u').values
        v = current_data.pivot(index='y', columns='x', values='v').values
        p = current_data.pivot(index='y', columns='x', values='p').values
        
        # Update the contour and quiver plots
        contourf = ax.contourf(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, alpha=0.5, cmap=plt.cm.viridis)
        contour = ax.contour(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), p, cmap=plt.cm.viridis)
        ax.quiver(np.linspace(0, Lx, nx), np.linspace(0, Ly, ny), u, v, color='r')
        
        # Labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Cavity Flow Velocity Field at t={frame * dt:.3f}s')

    ani = FuncAnimation(fig, update, frames=nt, repeat=False)
    ani.save(save_path, writer='ffmpeg')
    plt.close(fig)

if __name__ == "__main__":
    # Suggested parameters for running the simulation
    nx = 41
    ny = 41
    nt = 500
    nit = 50
    Lx = 2.0
    Ly = 2.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    rho = 1.0
    nu = 0.1
    dt = 0.001

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    u, v, p, cavity_flow_data = run_simulation(nt, u, v, dt, dx, dy, p, rho, nu, nit)
    save_data(cavity_flow_data, '../../simulation_data/cavity_flow_data.csv')
    plot_results(p, u, v, Lx, Ly, nx, ny, save_path='../../simulation_data/cavity_flow_plot.png')
    save_animation(u, v, p, Lx, Ly, nx, ny, dt, nt, cavity_flow_data, save_path='../../simulation_data/cavity_flow_animation.mp4')