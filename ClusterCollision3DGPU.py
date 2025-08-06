import cupy as cp  # For GPU acceleration
from cupyx.scipy.fft import fftn, ifftn  # GPU FFT
from tqdm import tqdm  # Progress bar
from mayavi import mlab  # For 3D visualization
import matplotlib.pyplot as plt  # For optional 2D slice

# QSTF 3D Soliton Collision Simulation for Bullet Cluster with GPU and Mayavi
# Author: Tõnis Leissoo
# Date: August 6, 2025
# Description: This script simulates the Bullet Cluster collision in the Quantum Spacetime Fluid (QSTF) model using the Gross-Pitaevskii equation (GPE) on GPU.
# It models two solitons colliding, computing density, vorticity, effective density, offset, mass ratio, and generates a 3D Mayavi visualization.
# Grid: 512^3 (adjust N for smaller tests if needed).

# Constants (astrophysical units)
hbar = 1.0545718e-27  # erg s
G = 4.302e-3  # pc/Msun (km/s)^2
Msun = 1.989e33  # g
kpc = 3.0857e18  # cm
s_to_yr = 3.156e7  # seconds per year
kpc_per_Myr = 3.0857e16 / (s_to_yr * 1e6)  # kpc/Myr per km/s
hbar_kpc = hbar * s_to_yr * 1e6 / (Msun * kpc)  # scaled ħ for Myr units

# Simulation parameters (from user script, tuned for Bullet Cluster)
N = 512  # Grid size (512^3)
L = 500.0  # Box size in kpc
dx = L / N
dt = 0.0001  # Timestep
Nt = 1000  # Number of timesteps
m = 1e-22  # Mass of constituents in eV
g = 1e-45 * (kpc**3 / Msun)  # Nonlinear coupling (scaled)
alpha = 0.05  # Vorticity coupling
rho_0 = 0.01  # Base density in Msun/pc³
v_collision = 4740  # Collision velocity in km/s
t_collision = 150  # Time since collision in Myr
v_x_normalized = v_collision * kpc_per_Myr / 2  # Normalized velocity in kpc/Myr

# Grid (use cp for GPU)
x = cp.linspace(-L/2, L/2, N)
y = cp.linspace(-L/2, L/2, N)
z = cp.linspace(-L/2, L/2, N)
X, Y, Z = cp.meshgrid(x, y, z, indexing='ij')

# Initial soliton wavefunctions (sech-shaped for stability, in kpc/Myr units)
w1 = 500.0  # Main soliton width in kpc
w2 = 200.0  # Sub soliton width in kpc
A1 = cp.sqrt(rho_0)  # Amplitude for main cluster
A2 = cp.sqrt(rho_0 * 0.23)  # Amplitude for subcluster
# Tanh approximation to avoid overflow
xi1 = (X - v_x_normalized * t_collision) / w1
xi2 = (X + v_x_normalized * t_collision) / w2
Psi1 = A1 * (1 - cp.tanh(xi1)**2) / (1 + cp.tanh(xi1)**2)
Psi2 = A2 * (1 - cp.tanh(xi2)**2) / (1 + cp.tanh(xi2)**2)
Psi = Psi1 + Psi2 * cp.exp(1j * cp.arctan(A2/A1))  # Phase shift for collision

# External potential (Newtonian for baryonic gas)
r = cp.sqrt(X**2 + Y**2 + Z**2 + 1e-6)  # Avoid singularity
M_gas = 1e14  # Baryonic gas mass in Msun
V_ext = -G * M_gas / r

# K space for Fourier transform
kx = 2 * cp.pi * cp.fft.fftfreq(N, d=dx)
KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
K2 = KX**2 + KY**2 + KZ**2

# GPE evolution loop
for t in tqdm(range(Nt)):
    Psi = Psi * cp.exp(-1j * dt / hbar_kpc * (V_ext + g * cp.abs(Psi)**2))
    Psi_k = fftn(Psi)
    Psi_k = Psi_k * cp.exp(-1j * dt * hbar_kpc / (2 * m) * K2)
    Psi = ifftn(Psi_k)
    Psi = Psi / cp.sqrt(cp.mean(cp.abs(Psi)**2) + 1e-10) * cp.sqrt(rho_0)

# Compute outputs
rho = cp.abs(Psi)**2  # Density
phase = cp.angle(Psi)  # Phase
v_x = (hbar_kpc / m) * cp.gradient(phase, axis=0) / dx  # Velocity x
v_y = (hbar_kpc / m) * cp.gradient(phase, axis=1) / dx  # Velocity y
v_z = (hbar_kpc / m) * cp.gradient(phase, axis=2) / dx  # Velocity z

# Vorticity components
omega_x = cp.gradient(v_z, axis=1) / dx - cp.gradient(v_y, axis=2) / dx
omega_y = cp.gradient(v_x, axis=2) / dx - cp.gradient(v_z, axis=0) / dx
omega_z = cp.gradient(v_y, axis=0) / dx - cp.gradient(v_x, axis=1) / dx
abs_omega = cp.sqrt(omega_x**2 + omega_y**2 + omega_z**2)  # Magnitude of vorticity

# Effective density for lensing
rho_eff = rho + alpha * abs_omega

# Gas peak (external potential max)
gas_peak = cp.unravel_index(cp.argmax(cp.abs(V_ext)), V_ext.shape)

# Lensing peak (rho_eff max)
lensing_peak = cp.unravel_index(cp.argmax(rho_eff), rho_eff.shape)

# Offset calculation
offset_x = (lensing_peak[0] - gas_peak[0]) * dx
offset_y = (lensing_peak[1] - gas_peak[1]) * dx
offset_z = (lensing_peak[2] - gas_peak[2]) * dx
offset = cp.sqrt(offset_x**2 + offset_y**2 + offset_z**2).get()

# Mass ratio
mass_ratio = cp.sum(rho_eff).get() / cp.sum(rho).get()

# Print results
print(f"Offset: {offset:.2f} kpc")
print(f"Mass Ratio: {mass_ratio:.2f}")
print(f"Parameters: m={m}, g={g}, alpha={alpha}, max rho={cp.max(rho).get():.3e} Msun/pc^3")

# Transfer to CPU for Mayavi visualization
rho_eff_cpu = cp.asnumpy(rho_eff)

# Mayavi 3D visualization
mlab.figure(size=(800, 600), bgcolor=(0.0, 0.0, 0.1))
mlab.contour3d(rho_eff_cpu, contours=8, opacity=0.5, colormap='viridis')
mlab.points3d([gas_peak[0] * dx], [gas_peak[1] * dx], [gas_peak[2] * dx], color=(1,0,0), mode='sphere', scale_factor=10, label='Gas Peak')
mlab.points3d([lensing_peak[0] * dx], [lensing_peak[1] * dx], [lensing_peak[2] * dx], color=(0,0,1), mode='sphere', scale_factor=10, label='Lensing Peak')
mlab.axes()
mlab.colorbar(title='Effective Density (Msun/pc³)')
mlab.title('QSTF Soliton Collision: Bullet Cluster 3D Density')
mlab.show()  # Opens interactive 3D window

# Optional 2D slice for comparison
rho_eff_slice = rho_eff[:, :, N//2].get()
plt.figure(figsize=(10, 8))
plt.imshow(rho_eff_slice, extent=(-L/2, L/2, -L/2, L/2), origin='lower', cmap='viridis')
plt.colorbar(label='Effective Density (Msun/pc³)')
plt.scatter([gas_peak[0] * dx], [gas_peak[1] * dx], c='red', label='Gas Peak')
plt.scatter([lensing_peak[0] * dx], [lensing_peak[1] * dx], c='blue', label='Lensing Peak')
plt.xlabel('x (kpc)')
plt.ylabel('y (kpc)')
plt.title('QSTF Soliton Collision: Bullet Cluster Density Slice (z=0, 512x512)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('bullet_cluster_qstf_512slice.png', dpi=300)
plt.show()