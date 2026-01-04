import numpy as np
import time as tm
import matplotlib.pyplot as plt
import pyvista as pv


path="./simulations/EMNS/"
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
def ns(u, rho, p,Amus, mu, zeta, dt, dx, charge_mass_ratio,polar):
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
            conv[i] += u[j] * grad_u[j, i]

    # Pressure gradient
    grad_p = gradient_scalar(p, dx)

    # Viscous terms
    lap_u = laplacian(u, dx)                   # Laplacian of velocity
    div_u = divergence(u, dx)                  # Divergence of velocity
    grad_div_u = gradient_scalar(div_u, dx)    # Gradient of divergence

    E=compute_E(Amus,dx,dt)
    B=compute_B(Amus,dx)

    uXB=np.array([(u[1]*B[2]-u[2]*B[1]),
                  (u[2]*B[0]-u[0]*B[2]),
                  (u[0]*B[1]-u[1]*B[0])])
    # Time derivative of velocity
    dudt = -conv - grad_p / rho + mu * lap_u / rho + (zeta + mu/3) * grad_div_u / rho + charge_mass_ratio*(E +   uXB )

    # Euler explicit update
    return u + dt * dudt

def continuity_step(rho, u, dt, dx):
    """
    Update density field using the continuity equation:
    ∂ρ/∂t + ∇·(ρ u) ≈ 0 
    """
    return rho - dt * divergence(rho*u, dx)

def polytropic(rho, kappa, gamma):
    """
    Polytropic equation of state: p = kappa * rho^gamma
    """
    return kappa * rho**gamma

def fluid_evolution(u, rho, p,Amus, dt, dx, mu, zeta, kappa, gamma,charge_mass_ratio,polar):
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
    u = ns(u, rho, p,Amus, mu, zeta, dt, dx,charge_mass_ratio,polar)
    rho = continuity_step(rho, u, dt, dx)
    p = polytropic(rho, kappa, gamma)
    return u, rho, p


  # --- Save frame VTI ---



# --------------------------------------------------------------------
# --- Electromagnetic field update using np.gradient -----------------
# --------------------------------------------------------------------
def A_update(Amu, Jmu, dx, dt, c, mu0):
    """
    Update the 4-vector potential Aμ using a finite difference wave equation.

    Amu : ndarray, shape (4, 3, Nx, Ny, Nz)
        Rolling buffer [t-2, t-1, t] for each component.
    Jmu : ndarray, shape (4, Nx, Ny, Nz)
        Source currents for each component.
    """
    A_prev  = Amu[:, 1, :, :, :]
    A_prev2 = Amu[:, 0, :, :, :]
    J_prev  = Jmu[:, 1, :, :, :]

    # 3D Laplacian using np.gradient
    lap = np.zeros_like(A_prev)
    for i in range(4):
        lap_x = np.gradient(np.gradient(A_prev[i], dx, axis=0), dx, axis=0)
        lap_y = np.gradient(np.gradient(A_prev[i], dx, axis=1), dx, axis=1)
        lap_z = np.gradient(np.gradient(A_prev[i], dx, axis=2), dx, axis=2)
        lap[i] = lap_x + lap_y + lap_z

    # Wave equation update
    Amu_next = 2 * A_prev - A_prev2 + (c*dt)**2 * lap / (dx**2) + (mu0 * (c*dt)**2) * J_prev

    return Amu_next

def compute_E(Amu, dx, dt):
    """
    Compute electric field from potentials: E = -∇φ - ∂A/∂t
    """
    phi = Amu[0, 1, :, :, :]                       # scalar potential at t^n
    grad_phi = np.array(np.gradient(phi, dx, dx, dx))  # gradient of scalar potential
    dA_dt = (Amu[1:4, 1, :, :, :] - Amu[1:4, 0, :, :, :]) / dt
    E = -grad_phi - dA_dt
    return np.array(E)

def compute_B(Amu, dx):
    """
    Compute magnetic field from vector potential: B = ∇ × A
    """
    # Extract vector potential at current time slice (t^n)
    Ax, Ay, Az = Amu[1, 1], Amu[2, 1], Amu[3, 1]

    # Compute gradients correctly: returns [df/dx, df/dy, df/dz]
    dAx_dx, dAx_dy, dAx_dz = np.gradient(Ax, dx, dx, dx)
    dAy_dx, dAy_dy, dAy_dz = np.gradient(Ay, dx, dx, dx)
    dAz_dx, dAz_dy, dAz_dz = np.gradient(Az, dx, dx, dx)

    # Curl: B = ∇ × A
    Bx = dAz_dy - dAy_dz
    By = dAx_dz - dAz_dx
    Bz = dAy_dx - dAx_dy

    return np.array([Bx, By, Bz])

def EM_update(Amu, rho, u, charge_mass_ratio, polar, dx, dt, c, mu0):
    """
    Update EM potentials and compute source currents.

    Parameters
    ----------
    Amu : ndarray (4, 3, Nx, Nx, Nx)
        4-potential rolling buffer: [φ, Ax, Ay, Az] x time buffer
    rho : ndarray (Nx, Nx, Nx)
        Charge density
    u : ndarray (3, Nx, Nx, Nx)
        Velocity field
    charge_mass_ratio : float
        Charge/mass scaling
    polar : float
        Polarization factor
    dx, dt, c, mu0 : float
        Grid spacing, timestep, speed of light, vacuum permeability
    """
    # --- Compute electric field ---
    E_field = compute_E(Amu, dx, dt)  # shape (3, Nx, Nx, Nx)

    # --- Compute 4-current ---
    J_vector = rho * charge_mass_ratio * u           # (3, Nx, Nx, Nx)
    J_scalar = rho * charge_mass_ratio               # (Nx, Nx, Nx)

    Jmu = np.zeros_like(Amu)
    Jmu[0, 1, :, :, :] = J_scalar                   # scalar charge density
    Jmu[1:4, 1, :, :, :] = J_vector + rho * polar * E_field  # vector current + polarization

    # --- Update potentials ---
    Amu[:, 2, :, :, :] = A_update(Amu, Jmu, dx, dt, c, mu0)

    # --- Advance rolling buffer ---
    Amu[:, 0] = Amu[:, 1]
    Amu[:, 1] = Amu[:, 2]

    return Amu, Jmu


# --------------------------------------------------------------------
# Simulation parameters
# --------------------------------------------------------------------
Nx = 40              # Grid size in each direction
dx = 1            # Spatial step [?]
t0,t1=0,10   # Time step [?]
CFL=0.01            # Simulation convergence parameter
skip=20


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
mu0=1
mu=1
kappa=1
gamma=1.4
zeta=1
charge_mass_ratio=1e-5
polar=1e-5
c=100
# --------------------------------------------------------------------
# Fluid dynamics
# --------------------------------------------------------------------

x = np.arange(Nx) - Nx//2
y = np.arange(Nx) - Nx//2
z = np.arange(Nx) - Nx//2
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
rho = 1 + 1 * np.exp(-(X**2 + Y**2 + Z**2)/(2*(Nx/10)**2))

Amus = np.zeros((4, 3, Nx, Nx, Nx))
Jmus = np.zeros((4, 3, Nx, Nx, Nx))


u = np.zeros((3, Nx, Nx, Nx))      # Velocity field (u_x, u_y, u_z)
p = polytropic(rho,kappa,gamma)        # Pressure field

t=t0
# --------------------------------------------------------------------
# Time step ensuring convergence
# --------------------------------------------------------------------

cs = np.sqrt(gamma * p / rho)  # field of sound speed (Nx,Nx,Nx)
u_mag = np.sqrt(np.sum(u**2, axis=0))
umax = np.max(u_mag)
csmax = np.max(cs)

dt = CFL * min(dx / (umax + csmax),dx/c)
print(f"At the beginning: umax={umax:.5e},csmax={csmax:.5e}, c={c} ,CFL-based dt = {dt:.5e} s")



# --------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------
print("The simulation has started.")
t=t0
n=0

pvd_entries = []
frame = 0

while t < t1:

    u, rho, p = fluid_evolution(u, rho, p,Amus, dt, dx, mu, zeta, kappa, gamma, charge_mass_ratio, polar)  # evolve fields
    Jmus,Amus=EM_update(Amus,rho,u,charge_mass_ratio,polar,dx,dt,c,mu0)

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
