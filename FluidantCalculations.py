import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# === QSTF Fluidant Parameters (Best-fit) ===
psi_0 = 0.1001
g_self = 0.0007
m_eff = 0.9999
alpha_env = 0.0102
z_decoh = 0.4999
sigma_decoh = 0.3000
phi_0 = 0.0000
omega_q = 0.1002
gamma_damp = 0.0101

def analytical_quantum_field(z, psi_0, g_self, m_eff, alpha_env, z_decoh, sigma_decoh, phi_0, omega_q, gamma_damp):
    z = np.atleast_1d(z)
    mass_evolution = np.exp(-gamma_damp * z) * (1 + z)**(-0.2)
    quantum_evolution = 1 + g_self * m_eff * z / (1 + z)
    psi_amp = psi_0 * mass_evolution * quantum_evolution
    psi_phase = omega_q * z + phi_0 + alpha_env * np.exp(-((z - z_decoh)**2) / (2 * sigma_decoh**2))
    return psi_amp, psi_phase

def fluidants_at_z(z):
    psi_amp, psi_phase = analytical_quantum_field(
        z, psi_0, g_self, m_eff, alpha_env, z_decoh, sigma_decoh, phi_0, omega_q, gamma_damp
    )
    density = psi_amp**2
    dz = 1e-3
    psi_amp_plus, psi_phase_plus = analytical_quantum_field(
        z + dz, psi_0, g_self, m_eff, alpha_env, z_decoh, sigma_decoh, phi_0, omega_q, gamma_damp
    )
    psi_amp_minus, psi_phase_minus = analytical_quantum_field(
        max(0, z - dz), psi_0, g_self, m_eff, alpha_env, z_decoh, sigma_decoh, phi_0, omega_q, gamma_damp
    )
    psi_amp_gradient = (psi_amp_plus - psi_amp_minus) / (2 * dz)
    psi_phase_gradient = (psi_phase_plus - psi_phase_minus) / (2 * dz)
    kinetic_density = (1.0 / (2 * m_eff)) * (psi_amp_gradient**2 + psi_amp**2 * psi_phase_gradient**2)
    V_env = alpha_env * np.exp(-((z - z_decoh)**2) / (2 * sigma_decoh**2))
    potential_density = g_self * psi_amp**4 + V_env * psi_amp**2 + (m_eff**2 / 2) * psi_amp**2
    pressure = kinetic_density - potential_density
    vorticity = psi_phase_gradient
    entropy_density = V_env
    return float(pressure), float(density), float(vorticity), float(entropy_density)

# -- Compute fluidants over z = 0 to 1 --
z_grid = np.linspace(0, 1.0, 500)
pressure = np.array([fluidants_at_z(z)[0] for z in z_grid])
density = np.array([fluidants_at_z(z)[1] for z in z_grid])
vorticity = np.array([fluidants_at_z(z)[2] for z in z_grid])
entropy = np.array([fluidants_at_z(z)[3] for z in z_grid])

# -- Find and mark peaks/plateaus in vorticity and entropy --
peaks_vort, _ = find_peaks(vorticity)
peaks_entropy, _ = find_peaks(entropy)

# -- Plot all four fluidants with annotation --
plt.figure(figsize=(12, 8))
plt.plot(z_grid, pressure, label='Pressure', lw=2)
plt.plot(z_grid, density, label='Density', lw=2)
plt.plot(z_grid, vorticity, label='Vorticity', lw=2)
plt.plot(z_grid, entropy, label='Entropy Density', lw=2, color='orange')
plt.scatter(z_grid[peaks_vort], vorticity[peaks_vort], color='blue', marker='o', s=60, label='Vorticity Peaks')
plt.scatter(z_grid[peaks_entropy], entropy[peaks_entropy], color='red', marker='^', s=80, label='Entropy Peaks')
plt.xlabel('Redshift $z$', fontsize=14)
plt.ylabel('Fluidant Value (arbitrary units)', fontsize=14)
plt.title('QSTF Fluidants vs. Redshift (Best-Fit Parameters)', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# -- Print peak epochs numerically for reporting --
print("Vorticity peaks at z =", z_grid[peaks_vort])
print("Entropy density peaks at z =", z_grid[peaks_entropy])
