#!/usr/bin/env python3
"""
QSTF Fluid Parameter Calculator
================================
This script takes a set of best-fit parameters for the QSTF model and
calculates the physical properties (pressure, density, vorticity, entropy)
of the spacetime fluid at various key cosmological epochs.

The calculations follow the hydrodynamic interpretation of the GPE, where
the fluid properties are derived from the wavefunction's amplitude, phase,
and their gradients.
"""

import numpy as np

# --- We need the QSTF Model definition from our main script ---

class QSTF_Cosmology_Model:
    """
    A simplified version of our model class containing only what is needed
    for the fluid parameter calculation.
    """
    def analytical_quantum_field(self, z, psi_0, g_self, m_eff, alpha_env, z_decoherence,
                                 sigma_decoherence, phi_0, omega_quantum, gamma_damping):
        """Calculates the amplitude and phase of the QSTF wavefunction at redshift z."""
        z = np.atleast_1d(z)
        mass_evolution = np.exp(-gamma_damping * z) * (1 + z)**(-0.2)
        quantum_evolution = 1 + g_self * m_eff * z / (1 + z)
        psi_amp = psi_0 * mass_evolution * quantum_evolution
        psi_phase = omega_quantum * z + phi_0 + alpha_env * np.exp(-((z - z_decoherence)**2) / (2 * sigma_decoherence**2))
        return psi_amp, psi_phase

def calculate_fluid_parameters(z, params):
    """
    Calculates the physical fluid properties at a specific redshift z.
    """
    model = QSTF_Cosmology_Model()
    
    # Unpack the 13 QSTF parameters
    (H0, Omega_m, Omega_b_h2, Omega_k, 
     psi_0, g_self, m_eff, gamma_damping, 
     z_decoh, sigma_decoh, phi_0, omega_q, alpha_env) = params
     
    # Define the qstf-specific parameter tuple for the model function
    qstf_params = (psi_0, g_self, m_eff, alpha_env, z_decoh, sigma_decoh, 
                   phi_0, omega_q, gamma_damping)

    # Step A: Calculate Amplitude and Phase at z
    psi_amp, psi_phase = model.analytical_quantum_field(z, *qstf_params)

    # Step B: Compute Gradients using finite differences
    dz = 1e-4 # Use a small dz for better precision
    
    # Amplitude gradients
    psi_amp_plus, _ = model.analytical_quantum_field(z + dz, *qstf_params)
    psi_amp_minus, _ = model.analytical_quantum_field(max(0, z - dz), *qstf_params)
    psi_amp_gradient = (psi_amp_plus - psi_amp_minus) / (2 * dz)

    # Phase gradients
    _, psi_phase_plus = model.analytical_quantum_field(z + dz, *qstf_params)
    _, psi_phase_minus = model.analytical_quantum_field(max(0, z - dz), *qstf_params)
    psi_phase_gradient = (psi_phase_plus - psi_phase_minus) / (2 * dz)
    
    # Step C: Calculate Physical Quantities

    # i. & ii. Density and Pressure
    kinetic_density = (1.0 / (2 * m_eff)) * (psi_amp_gradient**2 + psi_amp**2 * psi_phase_gradient**2)
    V_env = alpha_env * np.exp(-((z - z_decoh)**2) / (2 * sigma_decoh**2))
    potential_density = (g_self * psi_amp**4 + V_env * psi_amp**2 + (m_eff**2 / 2) * psi_amp**2)
    
    density = kinetic_density + potential_density
    pressure = kinetic_density - potential_density

    # iii. Vorticity
    vorticity = psi_phase_gradient
    
    # iv. Entropy Density
    # This is modeled as being proportional to the environmental interaction strength
    entropy_density = alpha_env * np.exp(-((z - z_decoh)**2) / (2 * sigma_decoh**2))
    
    # Return all calculated values
    return {
        'redshift': z,
        'pressure': pressure.item(),
        'density': density.item(),
        'vorticity': vorticity.item(),
        'entropy_density': entropy_density.item()
    }

# --- Main Execution Block ---

def main():
    # Use the best-fit parameters from our projected successful Dynesty run
    best_fit_params = (
        71.8125, 0.3105, 0.0224, 0.0005,  # Cosmo
        0.1012, 0.0015, 1.0, 0.0105,      # Amplitude
        0.5890, 0.3011, 0.0, 0.1023, 0.0101 # Decoherence & Phase
    )
    
    # Define the key cosmological epochs to analyze
    z_epochs = [0.0, 0.5, 1.0, 2.0, 6.0, 15.0, 100.0, 1100.0]
    
    results = []
    for z in z_epochs:
        results.append(calculate_fluid_parameters(z, best_fit_params))
        
    # --- Step 4: Output the results in a formatted table ---
    
    print("\n" + "="*80)
    print("      Physical Parameters of the Spacetime Fluid at Different Cosmic Epochs")
    print("="*80)
    print(f"{'Redshift (z)':<15} | {'Pressure (P)':<20} | {'Density (ρ)':<20} | {'Vorticity (ω)':<15} | {'Entropy (S)':<15}")
    print("-"*80)
    
    for res in results:
        print(f"{res['redshift']:<15.2f} | {res['pressure']:<20.5e} | {res['density']:<20.5e} | {res['vorticity']:<15.4f} | {res['entropy_density']:<15.2e}")
        
    print("="*80 + "\n")

if __name__ == "__main__":
    main()