import cupy as cp
import numpy as np
from cupyx.scipy.fft import fftn, ifftn
from mayavi import mlab
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore', category=FutureWarning, module='mayavi')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ===================================================================================
# QSTF 3D SIMULATION - OPTIMIZED FOR 4GB VRAM
#
# Description:
# This version is optimized to run at a high 384^3 resolution, providing the
# best balance of detail and performance for GPUs with ~4GB of VRAM. It uses
# single-precision (complex64) numbers to ensure memory efficiency.
# ===================================================================================


# --- 1. PHYSICAL CONSTANTS & UNIT SYSTEM ---
# -----------------------------------------------------------------
G = 4.49e-6  # Gravitational constant in kpc^3 / (Msun * Myr^2)


# --- 2. SIMULATION PARAMETERS ---
# -----------------------------------------------------------------
# Grid and Time Discretization for 4GB VRAM
N = 384       # OPTIMIZED: High resolution that safely fits in 4GB VRAM
L = 250.0     # Scaled-down box size for a testable scenario
dx = L / N
dt = 0.2
Nt = 750

# Tuned Physics Parameters
alpha = 0.02
m_eff = 1.0
g_eff = 0.0

# Initial Conditions
rho_0_pc3 = 0.015
rho_0 = rho_0_pc3 / (1e-3)**3
w_main = 50.0 # Scaled-down soliton core width
w_sub = 20.0
v_collision_kms = 4740
v_kpc_myr = v_collision_kms * 1.022e-3
phase_shift = 0.44
turbulence_strength = 0.1


# --- 3. SETUP ADVANCED INITIAL STATE ON GPU ---
# -----------------------------------------------------------------
print(f"Setting up optimized {N}^3 simulation for 4GB VRAM...")
# Use float32 for all base arrays to conserve memory
x = cp.linspace(-L/2, L/2, N, dtype=cp.float32)
X, Y, Z = cp.meshgrid(x, x, x, indexing='ij')

def create_sech_soliton(peak_density, w, center_pos, X, Y, Z):
    R = cp.sqrt((X - center_pos[0])**2 + (Y - center_pos[1])**2 + (Z - center_pos[2])**2)
    return cp.sqrt(peak_density) * (1.0 / cp.cosh(R / w))

initial_separation = L / 3.0
pos_main = cp.array([-initial_separation, 0, 0], dtype=cp.float32)
pos_sub = cp.array([initial_separation, 0, 0], dtype=cp.float32)

peak_density_main = rho_0
peak_density_sub = rho_0 * 0.23
Psi1 = create_sech_soliton(peak_density_main, w_main, pos_main, X, Y, Z)
Psi2 = create_sech_soliton(peak_density_sub, w_sub, pos_sub, X, Y, Z)

kick_main = cp.exp(1j * m_eff * (v_kpc_myr / 2.0) * X)
kick_sub = cp.exp(-1j * m_eff * (v_kpc_myr / 2.0) * X)
phase_kick = cp.exp(1j * phase_shift)

rng = cp.random.default_rng()
random_phase = rng.uniform(low=-np.pi, high=np.pi, size=(N, N, N), dtype=cp.float32)
turbulence_kick = cp.exp(1j * turbulence_strength * random_phase)

# Use complex64 to halve memory usage for the main wavefunction
Psi = (Psi1 * kick_main + Psi2 * kick_sub * phase_kick) * turbulence_kick
Psi = Psi.astype(cp.complex64)

initial_mass = cp.sum(cp.abs(Psi)**2) * dx**3
print(f"Initial total mass: {initial_mass.get():.2e} Msun")


# --- 4. SETUP POTENTIAL & KINETIC OPERATORS ---
# -----------------------------------------------------------------
M_baryon_total = initial_mass * 0.15
V_ext = -G * M_baryon_total / cp.sqrt(X**2 + Y**2 + Z**2 + (dx*5)**2)

# Use float32 for k-space arrays
kx = (2 * np.pi * cp.fft.fftfreq(N, d=dx)).astype(cp.float32)
KX, KY, KZ = cp.meshgrid(kx, kx, kx, indexing='ij')
K2 = (KX**2 + KY**2 + KZ**2)
T_k = K2 / (2 * m_eff)


# --- 5. GPE TIME EVOLUTION LOOP ---
# -----------------------------------------------------------------
print("Starting optimized simulation...")
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
rho_eff = rho

gas_peak_idx_gpu = cp.unravel_index(cp.argmax(-V_ext), V_ext.shape)
lensing_peak_idx_gpu = cp.unravel_index(cp.argmax(rho_eff), rho_eff.shape)

def get_phys_coords_on_cpu(idx_gpu, L, N):
    coords_np = np.array([c.get() for c in idx_gpu])
    return -L/2 + coords_np * (L/N)

gas_peak_coords_np = get_phys_coords_on_cpu(gas_peak_idx_gpu, L, N)
lensing_peak_coords_np = get_phys_coords_on_cpu(lensing_peak_idx_gpu, L, N)

offset = np.linalg.norm(lensing_peak_coords_np - gas_peak_coords_np)
final_mass = cp.sum(rho) * dx**3

# --- 7. DISPLAY RESULTS ---
# -----------------------------------------------------------------
print("\n--- OPTIMIZED SIMULATION RESULTS (384^3) ---")
print(f"Final DM-Gas Offset: {offset:.2f} kpc")
print(f"Mass Conservation: {(final_mass / initial_mass * 100).get():.3f}%")


# --- 8. VISUALIZATION ---
# -----------------------------------------------------------------
print("Generating visualizations...")
rho_cpu = cp.asnumpy(rho)

# 3D Visualization
mlab.figure(f"QSTF Simulation ({N}^3)", size=(800, 700), bgcolor=(0.1, 0.1, 0.2))
src = mlab.pipeline.scalar_field(rho_cpu)
vol = mlab.pipeline.volume(src, vmin=rho_cpu.max()*0.01, vmax=rho_cpu.max()*0.5)
mlab.axes(xlabel='x (kpc)', ylabel='y (kpc)', zlabel='z (kpc)')
mlab.title(f'DM Density (Offset: {offset:.1f} kpc)', size=0.5)
mlab.colorbar(title='Density (Msun/kpc³)')
mlab.show()

# 2D Slice
rho_slice = rho_cpu[N//2, :, :]
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(rho_slice, extent=(-L/2, L/2, -L/2, L/2), origin='lower', cmap='viridis')
fig.colorbar(im, label='Density Slice (Msun/kpc³)')
ax.scatter(gas_peak_coords_np[0], gas_peak_coords_np[1], s=200, facecolors='none', edgecolors='red', lw=2, label='Gas Center')
ax.scatter(lensing_peak_coords_np[0], lensing_peak_coords_np[1], s=200, c='cyan', marker='+', label='Lensing Peak')
ax.set_xlabel('x (kpc)')
ax.set_ylabel('y (kpc)')
ax.set_title(f'Central Slice (z=0) - {N}^3 Optimized Run')
ax.legend()
plt.savefig(f'bullet_cluster_optimized_{N}slice.png', dpi=300)
plt.show()

print("Script finished.")