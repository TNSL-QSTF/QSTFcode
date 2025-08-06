import cupy as cp
import numpy as np
from cupyx.scipy.fft import fftn, ifftn
from mayavi import mlab
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='mayavi')

# ===================================================================================
# QSTF 3D BULLET CLUSTER SIMULATION - FINAL VERSION
#
# Author: Tõnis Leissoo
# Date: August 7, 2025
#
# Description:
# This is the final, definitive script incorporating all user-specified parameters
# and feedback. The box size has been corrected to be much larger than the
# solitons to ensure a physically realistic simulation.
# ===================================================================================


# --- 1. PHYSICAL CONSTANTS & UNIT SYSTEM ---
# Units: kiloparsecs (kpc), Megayears (Myr), solar masses (Msun)
# -----------------------------------------------------------------
G = 4.49e-6  # Gravitational constant in kpc^3 / (Msun * Myr^2)


# --- 2. FINAL OPTIMIZED SIMULATION PARAMETERS ---
# -----------------------------------------------------------------
# Grid and Time Discretization
# ⚠️ WARNING: N=1024 requires a high-end GPU with >24 GB of VRAM.
N = 1024
# N = 256       # Use this for testing on consumer GPUs (<16 GB VRAM).
L = 2500.0    # CORRECTED: Larger box to properly contain the solitons.
dx = L / N
dt = 0.00005  # Fine-grained timestep in Myr
Nt = 3000     # MODIFIED: Increased timesteps for longer travel distance.

# Tuned Physics Parameters from Report
alpha = 0.02  # Vorticity Coupling
m_eff = 1.0   # Effective mass, calibrated for this model
g_eff = 0.0   # Self-interaction strength (collisionless model)

# Initial Conditions from Report
rho_0_pc3 = 0.015 # Base density in Msun/pc^3
rho_0 = rho_0_pc3 / (1e-3)**3 # Convert Msun/pc^3 to Msun/kpc^3
w_main = 500.0 # Main soliton width in kpc
w_sub = 200.0  # Sub-soliton width in kpc
v_collision_kms = 4740 # Total relative velocity
v_kpc_myr = v_collision_kms * 1.022e-3 # Convert to kpc/Myr
phase_shift = 0.44 # Relative phase shift between solitons in radians
turbulence_strength = 0.1 # Amplitude for random phase noise (turbulence proxy)


# --- 3. SETUP ADVANCED INITIAL STATE ON GPU ---
# -----------------------------------------------------------------
print(f"Setting up {N}^3 grid in a {L} kpc box...")
x = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')

def create_sech_soliton(peak_density, w, center_pos, X, Y, Z):
    """Creates a 3D hyperbolic secant (sech) soliton profile."""
    R = cp.sqrt((X - center_pos[0])**2 + (Y - center_pos[1])**2 + (Z - center_pos[2])**2)
    return cp.sqrt(peak_density) * (1.0 / cp.cosh(R / w))

# Start halos far apart in the new, larger box
initial_separation = L / 3.0
pos_main = cp.array([-initial_separation, 0, 0])
pos_sub = cp.array([initial_separation, 0, 0])

# Create individual solitons with specified peak densities
peak_density_main = rho_0
peak_density_sub = rho_0 * 0.23
Psi1 = create_sech_soliton(peak_density_main, w_main, pos_main, X, Y, Z)
Psi2 = create_sech_soliton(peak_density_sub, w_sub, pos_sub, X, Y, Z)

# Apply velocity kick, phase shift, and turbulence proxy
kick_main = cp.exp(1j * m_eff * (v_kpc_myr / 2.0) * X)
kick_sub = cp.exp(-1j * m_eff * (v_kpc_myr / 2.0) * X)
phase_kick = cp.exp(1j * phase_shift)

print("Generating turbulence proxy...")
rng = cp.random.default_rng()
random_phase = rng.uniform(low=-np.pi, high=np.pi, size=(N, N, N), dtype=cp.float32)
turbulence_kick = cp.exp(1j * turbulence_strength * random_phase)

# Combine all components into the final initial wavefunction
Psi = (Psi1 * kick_main + Psi2 * kick_sub * phase_kick) * turbulence_kick
Psi = Psi.astype(cp.complex128)

initial_mass = cp.sum(cp.abs(Psi)**2) * dx**3
print(f"Initial total mass: {initial_mass.get():.2e} Msun")


# --- 4. SETUP POTENTIAL & KINETIC OPERATORS ---
# -----------------------------------------------------------------
M_baryon_total = initial_mass * 0.15 # Baryonic mass as fraction of DM mass
V_ext = -G * M_baryon_total / cp.sqrt(X**2 + Y**2 + Z**2 + (dx*5)**2) # Softened potential

kx = 2 * np.pi * cp.fft.fftfreq(N, d=dx)
KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
K2 = cp.array(KX**2 + KY**2 + KZ**2)
T_k = K2 / (2 * m_eff)


# --- 5. GPE TIME EVOLUTION LOOP ---
# -----------------------------------------------------------------
print("Starting definitive high-fidelity simulation...")
for t in tqdm(range(Nt), desc=f"Evolving {N}^3 Grid"):
    V_total = V_ext + g_eff * cp.abs(Psi)**2
    Psi *= cp.exp(-1j * V_total * dt / 2.0)

    Psi_k = fftn(Psi)
    Psi_k *= cp.exp(-1j * T_k * dt)
    Psi = ifftn(Psi_k)

    V_total = V_ext + g_eff * cp.abs(Psi)**2
    Psi *= cp.exp(-1j * V_total * dt / 2.0)


# --- 6. POST-SIMULATION ANALYSIS ---
# -----------------------------------------------------------------
print("Simulation finished. Analyzing results...")
rho = cp.abs(Psi)**2
rho_eff = rho + alpha * cp.abs(cp.gradient(Psi, axis=0)[0]) # Simplified vorticity term for speed

gas_peak_idx_gpu = cp.unravel_index(cp.argmax(-V_ext), V_ext.shape)
lensing_peak_idx_gpu = cp.unravel_index(cp.argmax(rho_eff), rho_eff.shape)

def get_phys_coords_on_cpu(idx_gpu, L, N):
    coords_np = np.array([c.get() for c in idx_gpu])
    return -L/2 + coords_np * (L/N)

gas_peak_coords_np = get_phys_coords_on_cpu(gas_peak_idx_gpu, L, N)
lensing_peak_coords_np = get_phys_coords_on_cpu(lensing_peak_idx_gpu, L, N)

offset = np.linalg.norm(lensing_peak_coords_np - gas_peak_coords_np)
final_mass = cp.sum(rho) * dx**3
max_density_kpc3 = cp.max(rho).get()
max_density_pc3 = max_density_kpc3 * (1e-3)**3


# --- 7. DISPLAY RESULTS ---
# -----------------------------------------------------------------
print("\n--- FINAL SIMULATION RESULTS ---")
print(f"Final DM-Gas Offset: {offset:.2f} kpc")
print(f"Max Density: {max_density_pc3:.4f} Msun/pc^3")
print(f"Mass Conservation: {(final_mass / initial_mass * 100).get():.3f}%")


# --- 8. VISUALIZATION ---
# -----------------------------------------------------------------
print("Generating visualizations...")
rho_eff_cpu = cp.asnumpy(rho_eff)

# 3D Visualization
mlab.figure(f"QSTF Simulation ({N}^3)", size=(800, 700), bgcolor=(0.1, 0.1, 0.2))
src = mlab.pipeline.scalar_field(rho_eff_cpu)
vol = mlab.pipeline.volume(src, vmin=rho_eff_cpu.max()*0.01, vmax=rho_eff_cpu.max()*0.5)
mlab.axes(xlabel='x (kpc)', ylabel='y (kpc)', zlabel='z (kpc)')
mlab.title(f'Effective Density (Offset: {offset:.1f} kpc)', size=0.5)
mlab.colorbar(title='Effective Density (Msun/kpc³)')
mlab.show()

# 2D Slice
rho_slice = rho_eff_cpu[N//2, :, :]
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(rho_slice, extent=(-L/2, L/2, -L/2, L/2), origin='lower', cmap='viridis')
fig.colorbar(im, label='Effective Density Slice (Msun/kpc³)')
ax.scatter(gas_peak_coords_np[0], gas_peak_coords_np[1], s=200, facecolors='none', edgecolors='red', lw=2, label='Gas Center (Potential Min)')
ax.scatter(lensing_peak_coords_np[0], lensing_peak_coords_np[1], s=200, c='cyan', marker='+', label='Lensing Peak (Density Max)')
ax.set_xlabel('x (kpc)')
ax.set_ylabel('y (kpc)')
ax.set_title(f'Central Slice (z=0) - {N}^3 Definitive Run')
ax.legend()
plt.savefig(f'bullet_cluster_qstf_{N}slice.png', dpi=300)
plt.show()

print("Script finished.")