import numpy as np
import time as tm
import matplotlib.pyplot as plt
import pyvista as pv


path="./simulations/ns_integrator/"
t_start=tm.time()
# --------------------------------------------------------------------
# Differential operators
# --------------------------------------------------------------------
def gradient_scalar(f, dx):
    """
    Compute the gradient of a scalar field f.
    Returns a 3-component vector field [df/dx, df/dy, df/dz].
    """
    return np.array(np.gradient(f, dx, dx, dx))

def divergence(u, dx):
    """
    Compute the divergence of a 3D vector field u = [u_x, u_y, u_z].
    """
    return (np.gradient(u[0], dx, axis=0) +  np.gradient(u[1], dx, axis=1) +   np.gradient(u[2], dx, axis=2))

def laplacian(f, dx):
    """
    Compute the Laplacian of a scalar or vector field.
    - For scalar fields: returns scalar Laplacian.
    - For vector fields (shape 3xNxNxNx): returns Laplacian component-wise.
    """
    if f.ndim == 3:  # scalar field
        return (np.gradient(np.gradient(f, dx, axis=0), dx, axis=0) +
                np.gradient(np.gradient(f, dx, axis=1), dx, axis=1) +
                np.gradient(np.gradient(f, dx, axis=2), dx, axis=2))
    elif f.ndim == 4:  # vector field
        lap = np.zeros_like(f)
        for i in range(3):
            lap[i] = laplacian(f[i], dx)
        return lap

# --------------------------------------------------------------------
# Fluid dynamics updates
# --------------------------------------------------------------------
def ns(X,Y,Z,u, rho, p, mu, zeta, dt, dx):
    """
    Vectorized explicit Euler step for compressible Navier-Stokes equations.
    
    Parameters:
        u   : velocity field (3, Nx, Ny, Nz)
        rho : density field (Nx, Ny, Nz)
        p   : pressure field (Nx, Ny, Nz)
        mu  : dynamic viscosity
        zeta: bulk viscosity
        dt  : time step
        dx  : grid spacing
    Returns:
        Updated velocity field (3, Nx, Ny, Nz)
    """
    # Compute gradient of all velocity components: shape (3,3,Nx,Ny,Nz)
    grad_u = np.array([np.gradient(u[i], dx, dx, dx) for i in range(3)])

    # Convective term: (u · ∇) u

    conv = np.zeros_like(u)
    for i in range(3):
        for j in range(3):
            conv[i] += u[j] * grad_u[i, j]

    # Pressure gradient
    grad_p = gradient_scalar(p, dx)

    # Viscous terms
    lap_u = laplacian(u, dx)                   # Laplacian of velocity
    div_u = divergence(u, dx)                  # Divergence of velocity
    grad_div_u = gradient_scalar(div_u, dx)    # Gradient of divergence

    # Time derivative of velocity
    dudt = -conv - grad_p / rho + mu * lap_u / rho + (zeta + mu/3) * grad_div_u / rho+F(X,Y,Z)/rho

    # Euler explicit update
    return u + dt * dudt

def continuity_step(rho, u, dt=1.0, dx=1.0):
    """
    Update density field using the continuity equation:
    ∂ρ/∂t + ∇·(ρ u) ≈ 0 
    """
    return rho - dt * divergence(rho*u, dx)

def polytropic(rho, kappa=1.0, gamma=1.4):
    """
    Polytropic equation of state: p = kappa * rho^gamma
    """
    return kappa *(rho**gamma)

def temporal_evolution(X,Y,Z,u, rho, p, dt=1.0, dx=1.0, mu=1.0, zeta=1.0, kappa=1.0, gamma=1.4):
    """
    Advance the compressible fluid dynamics system by one time step using
    explicit Euler integration.

    Parameters
    ----------
    u : ndarray, shape (3, Nx, Ny, Nz)
        Velocity field components [u_x, u_y, u_z].
    rho : ndarray, shape (Nx, Ny, Nz)
        Density field.
    p : ndarray, shape (Nx, Ny, Nz)
        Pressure field.
    dt : float
        Time step for integration.
    dx : float
        Spatial grid spacing.
    mu : float
        Dynamic viscosity coefficient.
    zeta : float
        Bulk viscosity coefficient.
    kappa : float
        Polytropic constant in the equation of state p = kappa * rho^gamma.
    gamma : float
        Polytropic index.

    Returns
    -------
    tuple of ndarrays
        Updated (u, rho, p) after one time step.
    """
    u = ns(X,Y,Z,u, rho, p, mu, zeta, dt, dx)
    rho = continuity_step(rho, u, dt, dx)
    p = polytropic(rho, kappa, gamma)

    # --- Boundary conditions: zero normal velocity at boundaries ---
    # Set velocity normal to each boundary to zero
    u[:, 0, :, :]   = 0  # x=0 face
    u[:, -1, :, :]  = 0  # x=Nx-1 face
    u[:, :, 0, :]   = 0  # y=0 face
    u[:, :, -1, :]  = 0  # y=Nx-1 face
    u[:, :, :, 0]   = 0  # z=0 face
    u[:, :, :, -1]  = 0  # z=Nx-1 face

    # Optional: clamp density at boundaries to initial value
    rho[0, :, :]    = 1
    rho[-1, :, :]   = 1
    rho[:, 0, :]    = 1
    rho[:, -1, :]   = 1
    rho[:, :, 0]    = 1
    rho[:, :, -1]   = 1

    return u, rho, p

def F(X,Y,Z):
    ff=8
    sigma=1
    return np.array([ff*np.exp(-(X**2+Y**2+Z**2)/sigma),np.zeros_like(X),np.zeros_like(X)])


  # --- Save frame VTI ---


# --------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------
Nx = 50              # Grid size in each direction
dx = 1            # Spatial step [?]
t0,t1=0,500 # Time step [?]
CFL=0.1         # Simulation convergence parameter
skip=4000


# --------------------------------------------------------------------
# Simulation preparation
# --------------------------------------------------------------------
grid = pv.ImageData()
grid.dimensions = (Nx, Nx, Nx)
grid.spacing = (dx, dx, dx)
grid.origin  = (0.0, 0.0, 0.0)

# --------------------------------------------------------------------
# Physical parameters
# --------------------------------------------------------------------
mu=1
kappa=100
gamma=2
zeta=1

# --------------------------------------------------------------------
# Initialize fields
# --------------------------------------------------------------------

# Create coordinates
x = np.arange(Nx) - Nx//2
y = np.arange(Nx) - Nx//2
z = np.arange(Nx) - Nx//2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Initialize fields
u = np.zeros((3, Nx, Nx, Nx))  # Velocity field (u_x, u_y, u_z)
rho = np.ones((Nx, Nx, Nx))    # Density field

# Jet region in y-z plane
jy = slice(Nx//2 - Nx//8, Nx//2 + Nx//8)
jz = slice(Nx//2 - Nx//8, Nx//2 + Nx//8)

jx = slice(Nx//2 - Nx//16, Nx//2 + Nx//16)

# Gaussian smoothing parameters
sigma_y = Nx//10
sigma_z = Nx//10


p = polytropic(rho)        # Pressure field

t=t0
# --------------------------------------------------------------------
# Time step ensuring convergence
# --------------------------------------------------------------------

cs = np.sqrt(gamma * p / rho)  # field of sound speed (Nx,Nx,Nx)
u_mag = np.sqrt(np.sum(u**2, axis=0))
umax = np.max(u_mag)
csmax = np.max(cs)

dt = CFL * dx / (umax + csmax)
print(f"At the beginning: umax={umax:.5e},csmax={csmax:.5e},CFL-based dt = {dt:.5e} s")



# --------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------
print("The simulation has started.")
t=t0
n=0

pvd_entries = []
frame = 0

# --- ciclo temporale ---
while t < t1:

    u, rho, p = temporal_evolution(X,Y,Z,u, rho, p, dt, dx, mu, zeta, kappa, gamma)  # evolve fields
    cs = np.sqrt(gamma * p / rho)  # sound speed field
    u_mag = np.sqrt(np.sum(u**2, axis=0))
    umax = np.max(u_mag)
    csmax = np.max(cs)
    dt = CFL * dx / (umax + csmax)
    t = t + dt

    n += 1  # increment step counter

    if n > skip:
        print(f"{t/(t1-t0)*100:.3f}% of the way...")
        n = 0

        # Add fields
        grid["rho"] = rho.flatten(order="F")                 # scalar field
        grid["u"]   = u.reshape(3, Nx*Nx*Nx, order="F").T    # vector field

        # Save file VTI
        filename = f"sim01_{frame:04d}.vti"
        grid.save(path + filename)

        # Update PVD entries
        pvd_entries.append((t, filename))
        frame += 1

        # Write PVD in real time
        with open(path + "sim01.pvd", "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write('  <Collection>\n')
            for time, fname in pvd_entries:
                f.write(f'    <DataSet timestep="{time}" file="{fname}"/>\n')
            f.write('  </Collection>\n')
            f.write('</VTKFile>\n')




T=tm.time() - t_start
print(f"Done! It took {T} s")
