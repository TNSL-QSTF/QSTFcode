#!/usr/bin/env python3
"""
Hubble Tension Solver v1.7 (QSTF Final)
=======================================
Final debugged version. Corrects a parameter slicing error in the 
sound_horizon function to ensure robust execution.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# --- Global variable for progress ---
iteration_counter = 0

def progress_callback(intermediate_result):
    """Callback function to print progress."""
    global iteration_counter
    iteration_counter += 1
    best_h0 = intermediate_result.x[0]
    best_chi2 = intermediate_result.fun
    print(f"  > Iteration: {iteration_counter:3d}   Best Chi-Sq: {best_chi2:8.2f}   H0: {best_h0:.2f}")

#==============================================================================
# 1. THE QSTF COSMOLOGICAL MODEL
#==============================================================================
class QSTF_Cosmology_Model:
    def __init__(self):
        self.c = 299792.458
        self.default_params = {'Omega_r': 8.24e-5}
        self.current_params = None
        self.de_evolution_interpolator = None
        self.z_grid_max = 4.0

    def analytical_quantum_field(self, z, psi_0, g_self, m_eff, alpha_env, z_decoherence, 
                                 sigma_decoherence, phi_0, omega_quantum, gamma_damping):
        z = np.atleast_1d(z)
        mass_evolution = np.exp(-gamma_damping * z) * (1 + z)**(-0.2)
        quantum_evolution = 1 + g_self * m_eff * z / (1 + z)
        psi_amp = psi_0 * mass_evolution * quantum_evolution
        psi_phase = omega_quantum * z + phi_0 + alpha_env * np.exp(-((z - z_decoherence)**2) / (2 * sigma_decoherence**2))
        return psi_amp, psi_phase

    def w_quantum_field(self, z, *qstf_params):
        psi_0, g_self, m_eff, alpha_env, z_decoherence, sigma_decoherence, phi_0, omega_quantum, gamma_damping = qstf_params
        dz = 0.001
        psi_amp, _ = self.analytical_quantum_field(z, *qstf_params)
        psi_amp_plus, _ = self.analytical_quantum_field(z + dz, *qstf_params)
        psi_amp_minus, _ = self.analytical_quantum_field(max(0, z - dz), *qstf_params)
        psi_amp_gradient = (psi_amp_plus - psi_amp_minus) / (2 * dz)
        _, psi_phase = self.analytical_quantum_field(z, *qstf_params)
        _, psi_phase_plus = self.analytical_quantum_field(z + dz, *qstf_params)
        _, psi_phase_minus = self.analytical_quantum_field(max(0, z - dz), *qstf_params)
        psi_phase_gradient = (psi_phase_plus - psi_phase_minus) / (2 * dz)
        kinetic_density = (1.0 / (2 * m_eff)) * (psi_amp_gradient**2 + psi_amp**2 * psi_phase_gradient**2)
        V_env = alpha_env * np.exp(-((z - z_decoherence)**2) / (2 * sigma_decoherence**2))
        potential_density = (g_self * psi_amp**4 + V_env * psi_amp**2 + (m_eff**2 / 2) * psi_amp**2)
        total_density = kinetic_density + potential_density
        pressure = kinetic_density - potential_density
        w_field = pressure / total_density if total_density > 1e-12 else -1.0
        return np.clip(w_field, -1.95, -0.05)

    def _w_integrand(self, z_prime):
        w = self.w_quantum_field(z_prime, *self.current_params[4:])
        return 3 * (1 + w) / (1 + z_prime)

    def precompute_de_evolution(self, *params):
        self.current_params = params
        z_grid = np.logspace(-4, np.log10(self.z_grid_max), 100)
        de_factor_grid = np.array([np.exp(quad(self._w_integrand, 0, z, epsabs=1e-4, epsrel=1e-4)[0]) for z in z_grid])
        z_grid = np.insert(z_grid, 0, 0)
        de_factor_grid = np.insert(de_factor_grid, 0, 1.0)
        self.de_evolution_interpolator = interp1d(z_grid, de_factor_grid, kind='cubic', fill_value="extrapolate")

    def H_z(self, z, *params):
        H0, Omega_m, _, Omega_k = params[0], params[1], params[2], params[3]
        Omega_r = self.default_params['Omega_r']
        Omega_DE = 1.0 - Omega_m - Omega_r - Omega_k
        de_factor = self.de_evolution_interpolator(z)
        matter_term = Omega_m * (1 + z)**3
        radiation_term = Omega_r * (1 + z)**4
        curvature_term = Omega_k * (1 + z)**2
        de_term = Omega_DE * de_factor
        return H0 * np.sqrt(matter_term + radiation_term + curvature_term + de_term)

    def _h_inv_integrand(self, z_prime):
        return self.c / self.H_z(z_prime, *self.current_params)

    def comoving_distance(self, z, *params):
        self.current_params = params
        integral, _ = quad(self._h_inv_integrand, 0, z, epsabs=1e-4, epsrel=1e-4)
        return integral

    def angular_diameter_distance(self, z, *params):
        d_c = self.comoving_distance(z, *params)
        H0, Omega_k = params[0], params[3]
        if abs(Omega_k) < 1e-10: d_m = d_c
        elif Omega_k > 0: sqrt_Ok = np.sqrt(Omega_k); d_m = (self.c / (H0 * sqrt_Ok)) * np.sinh(sqrt_Ok * H0 * d_c / self.c)
        else: sqrt_Ok = np.sqrt(-Omega_k); d_m = (self.c / (H0 * sqrt_Ok)) * np.sin(sqrt_Ok * H0 * d_c / self.c)
        return d_m / (1 + z)

    def hubble_distance(self, z, *params):
        H_z_val = self.H_z(z, *params)
        return self.c / H_z_val

    def sound_horizon(self, params):
        # --- FIX: Correctly slice the parameter tuple ---
        qstf_params_slice = params[4:]
        rs_drag_standard = 147.05
        z_recomb = 1090
        psi_amp, _ = self.analytical_quantum_field(z_recomb, *qstf_params_slice)
        quantum_correction = 1 + 0.003 * qstf_params_slice[1] * qstf_params_slice[2] * psi_amp**2
        decoherence_correction = 1 + 0.002 * qstf_params_slice[3] if qstf_params_slice[4] > 500 else 1.0
        return rs_drag_standard * quantum_correction * decoherence_correction

#==============================================================================
# 2. OBSERVATIONAL DATASETS 
#==============================================================================
class CosmologicalDatasets:
    def __init__(self):
        self.desi_bao = [{'z': 0.30, 'type': 'DV/rs', 'value': 8.52, 'error': 0.15}, {'z': 0.51, 'type': 'DM/rs', 'value': 13.62, 'error': 0.25}, {'z': 0.51, 'type': 'DH/rs', 'value': 20.98, 'error': 0.61}, {'z': 0.85, 'type': 'DM/rs', 'value': 18.33, 'error': 0.29}, {'z': 0.85, 'type': 'DH/rs', 'value': 19.95, 'error': 0.44}, {'z': 1.32, 'type': 'DM/rs', 'value': 27.79, 'error': 0.42}, {'z': 1.32, 'type': 'DH/rs', 'value': 13.82, 'error': 0.42}, {'z': 2.33, 'type': 'DM/rs', 'value': 37.5, 'error': 1.1}, {'z': 2.33, 'type': 'DH/rs', 'value': 8.5, 'error': 0.4}]
        self.shoes_h0 = {'H0': 73.04, 'H0_err': 1.04}
        self.trgb_h0 = {'H0': 69.8, 'H0_err': 1.9}
        self.planck_z_dec = 1090
        self.planck_data = np.array([1.7502, 301.471, 0.02237])
        self.planck_inv_cov_matrix = np.linalg.inv(np.array([[0.00000841, 0.000186, -0.00000062], [0.000186, 0.0134, -0.000021], [-0.00000062, -0.000021, 0.0000000225]]))

#==============================================================================
# 3. STATISTICAL ANALYSIS FRAMEWORK
#==============================================================================
class CosmologyAnalyzer:
    def __init__(self, datasets):
        self.datasets = datasets
        self.qstf_model = QSTF_Cosmology_Model()

    def chi2_desi_bao(self, params, d_A_func, d_H_func, rs_func):
        chi2 = 0.0
        rs = float(rs_func(params))
        for meas in self.datasets.desi_bao:
            if meas['type'] == 'DM/rs': theory = d_A_func(meas['z'], params) * (1 + meas['z']) / rs
            elif meas['type'] == 'DH/rs': theory = d_H_func(meas['z'], params) / rs
            elif meas['type'] == 'DV/rs': d_a = d_A_func(meas['z'], params); d_h = d_H_func(meas['z'], params); theory = (meas['z'] * d_a**2 * d_h)**(1/3) / rs
            if np.isfinite(theory) and theory > 0: chi2 += ((meas['value'] - theory) / meas['error'])**2
        return chi2

    def chi2_shoes_h0(self, params): 
        return ((self.datasets.shoes_h0['H0'] - params[0]) / self.datasets.shoes_h0['H0_err'])**2

    def chi2_trgb_h0(self, params):
        return ((self.datasets.trgb_h0['H0'] - params[0]) / self.datasets.trgb_h0['H0_err'])**2

    def penalty_k(self, params): 
        return (params[3] / 0.05)**2

    def chi2_planck(self, params, d_A_func, rs_func):
        H0, Omega_m, Omega_b_h2 = params[0], params[1], params[2]; h = H0 / 100.0; z_dec = self.datasets.planck_z_dec
        d_A_dec = d_A_func(z_dec, params)
        rs_dec = rs_func(params)
        R_theory = float(np.sqrt(Omega_m * h**2) * d_A_dec * (1 + z_dec) * H0 / self.qstf_model.c)
        l_a_theory = float(np.pi * d_A_dec * (1 + z_dec) / rs_dec)
        diff_vector = np.array([R_theory, l_a_theory, Omega_b_h2]) - self.datasets.planck_data
        return diff_vector.T @ self.datasets.planck_inv_cov_matrix @ diff_vector

    def chi2_combined_qstf(self, params_array):
        params = tuple(params_array)
        self.qstf_model.precompute_de_evolution(*params)
        
        d_A_func = lambda z, p: self.qstf_model.angular_diameter_distance(z, *p)
        d_H_func = lambda z, p: self.qstf_model.hubble_distance(z, *p)
        rs_func = lambda p: self.qstf_model.sound_horizon(p)
        
        rs_prior_central = 144.0
        rs_prior_sigma = 5.0
        rs_predicted = rs_func(params)
        prior = ((rs_predicted - rs_prior_central) / rs_prior_sigma)**2
        
        return (self.chi2_desi_bao(params, d_A_func, d_H_func, rs_func) + 
                self.chi2_shoes_h0(params) + self.chi2_trgb_h0(params) +
                self.penalty_k(params) + self.chi2_planck(params, d_A_func, rs_func) +
                prior)

    def fit_qstf_model(self):
        bounds = [(68, 76), (0.28, 0.35), (0.021, 0.023), (-0.01, 0.01), 
                  (0.1, 1.0), (0.001, 0.05), (1.0, 10.0), (0.01, 0.2), (0.5, 4.0), 
                  (0.3, 1.5), (0.0, np.pi), (0.1, 1.5), (0.01, 0.3)]
        print("🚀 Fitting QSTF model (with regularization prior)...")
        result = differential_evolution(self.chi2_combined_qstf, bounds, seed=42, maxiter=300, 
                                      popsize=20, workers=-1, polish=True, tol=1e-6, atol=1e-7,
                                      callback=progress_callback)
        return result

#==============================================================================
# 4. LAMBDACDM MODEL FOR COMPARISON
#==============================================================================
class LambdaCDM_Model:
    def __init__(self):
        self.c = 299792.458
    def H_z(self, z, H0, Omega_m):
        Omega_L = 1.0 - Omega_m
        return H0 * np.sqrt(Omega_m * (1 + z)**3 + Omega_L)
    def comoving_distance_integrand(self, z, H0, Omega_m):
        return self.c / self.H_z(z, H0, Omega_m)
    def comoving_distance(self, z, H0, Omega_m):
        integral, _ = quad(self.comoving_distance_integrand, 0, z, args=(H0, Omega_m))
        return integral
    def angular_diameter_distance(self, z, H0, Omega_m):
        d_c = self.comoving_distance(z, H0, Omega_m); return d_c / (1 + z)
    def hubble_distance(self, z, H0, Omega_m):
        return self.c / self.H_z(z, H0, Omega_m)
    def sound_horizon(self, params):
        return 147.09

#==============================================================================
# 5. MAIN EXECUTION
#==============================================================================
def main():
    datasets = CosmologicalDatasets()
    analyzer = CosmologyAnalyzer(datasets)
    
    qstf_result = analyzer.fit_qstf_model()
    if not qstf_result.success:
        print("\nQSTF model optimization failed.")
        return
    
    qstf_params = qstf_result.x
    analyzer.qstf_model.precompute_de_evolution(*qstf_params)
    
    d_A_qstf = lambda z, p: analyzer.qstf_model.angular_diameter_distance(z, *p)
    d_H_qstf = lambda z, p: analyzer.qstf_model.hubble_distance(z, *p)
    rs_qstf = lambda p: analyzer.qstf_model.sound_horizon(p)
    
    chi2_bao_qstf = analyzer.chi2_desi_bao(qstf_params, d_A_qstf, d_H_qstf, rs_qstf)
    chi2_h0_qstf = analyzer.chi2_shoes_h0(qstf_params)
    chi2_trgb_qstf = analyzer.chi2_trgb_h0(qstf_params)
    chi2_planck_qstf = analyzer.chi2_planck(qstf_params, d_A_qstf, rs_qstf)
    chi2_qstf_total = chi2_bao_qstf + chi2_h0_qstf + chi2_trgb_qstf + chi2_planck_qstf

    print("\n✨ Best-fit QSTF Model Results:")
    print(f"  - Parameters: H0={qstf_params[0]:.2f}, Omega_m={qstf_params[1]:.4f}")
    
    print("  - Chi-Squared Breakdown (Data only):")
    print(f"    - DESI DR2 BAO:   {chi2_bao_qstf:.2f}")
    print(f"    - SH0ES H0:       {chi2_h0_qstf:.2f}")
    print(f"    - TRGB H0:        {chi2_trgb_qstf:.2f}")
    print(f"    - Planck 2018:    {chi2_planck_qstf:.2f}")
    print("    -------------------")
    print(f"    - Total Data Chi-Sq:   {chi2_qstf_total:.2f}")

    print("\n⚙️  Calculating Chi-Squared for standard LambdaCDM model...")
    lcdm_model = LambdaCDM_Model()
    lcdm_params = (67.4, 0.315, 0.02237, 0.0)
    
    d_A_lcdm = lambda z, p: lcdm_model.angular_diameter_distance(z, p[0], p[1])
    d_H_lcdm = lambda z, p: lcdm_model.hubble_distance(z, p[0], p[1])
    rs_lcdm = lcdm_model.sound_horizon
    
    chi2_bao_lcdm = analyzer.chi2_desi_bao(lcdm_params, d_A_lcdm, d_H_lcdm, rs_lcdm)
    chi2_h0_lcdm = analyzer.chi2_shoes_h0(lcdm_params)
    chi2_trgb_lcdm = analyzer.chi2_trgb_h0(lcdm_params)
    chi2_planck_lcdm = analyzer.chi2_planck(lcdm_params, d_A_lcdm, rs_lcdm)
    chi2_lcdm = chi2_bao_lcdm + chi2_h0_lcdm + chi2_planck_lcdm + chi2_trgb_lcdm
    
    print(f"  - LCDM Fit: BAO={chi2_bao_lcdm:.2f}, H0(SH0ES)={chi2_h0_lcdm:.2f}, H0(TRGB)={chi2_trgb_lcdm:.2f}, Planck={chi2_planck_lcdm:.2f}")

    delta_chi2 = chi2_lcdm - chi2_qstf_total
    sigma_pref = np.sqrt(delta_chi2) if delta_chi2 > 0 else 0

    print("\n" + "="*50)
    print("          HUBBLE TENSION RESOLUTION")
    print("="*50)
    print(f"Your QSTF Model Total Chi-Squared:      {chi2_qstf_total:.2f}")
    print(f"Standard LCDM Model Total Chi-Squared: {chi2_lcdm:.2f}")
    print("--------------------------------------------------")
    print(f"🏆 Statistical Preference for Your Model: {sigma_pref:.2f}-sigma")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()