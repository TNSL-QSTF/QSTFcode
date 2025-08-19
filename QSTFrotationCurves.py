#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rotation_curve_fitter.py
===============================
Advanced rotation curve fitting for SPARC galaxy data using quantum soliton + phonon model.

This script implements a comprehensive framework for analyzing galaxy rotation curves with:
- Quantum soliton dark matter cores (pseudo-isothermal profile)
- MOND-like phonon contributions for modified gravity effects
- Proper statistical analysis with error floors and reduced chi-squared
- Batch processing of SPARC database
- Detailed output and visualization options

Physics Model:
--------------
V_total² = V_baryons² + V_soliton² + V_phonon²

Where:
- V_baryons² = V_gas² + (f_disk × V_disk)² + (f_disk × V_bulge)²
- V_soliton from ρ(r) = ρ_c / (1 + (r/r_c)²) profile
- V_phonon from MOND-like μ-function: V_ph² = a₀R μ(V_bar²/(a₀R))

Author: Assistant
Date: 2025-08-19
"""

import argparse
import warnings
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from scipy.optimize import curve_fit, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# =====================================
# PHYSICAL CONSTANTS & CONFIGURATION
# =====================================

# Gravitational constant in galactic units
G_GAL = 4.30091e-6  # (kpc/M_sun) × (km/s)²

# MOND acceleration scale in kpc units
A0_MOND = 2.0e-3  # (km/s)² / kpc ≈ 1.2 × 10⁻¹⁰ m/s²

# Critical density for cosmological calculations
RHO_CRIT = 135.9  # M_sun / kpc³ (for H₀ ≈ 70 km/s/Mpc)

# Required columns in input CSV (case-insensitive matching)
REQUIRED_COLUMNS = [
    "Galaxy", "R_kpc", "Vobs_kms", "eVobs_kms", 
    "Vgas_kms", "Vdisk_kms", "Vbul_kms"
]

@dataclass
class FitConfiguration:
    """Configuration parameters for rotation curve fitting."""
    
    # Error handling
    error_floor_fraction: float = 0.05  # 5% minimum fractional error
    min_points_per_galaxy: int = 5       # Minimum data points required
    
    # Parameter bounds for optimization
    f_disk_bounds: Tuple[float, float] = (0.1, 3.0)     # Disk mass scaling
    rho_c_bounds: Tuple[float, float] = (1e4, 1e12)     # Core density [M_sun/kpc³]
    r_c_bounds: Tuple[float, float] = (0.05, 50.0)      # Core radius [kpc]
    
    # Physics options
    use_phonon_term: bool = True         # Include MOND-like phonon correction
    use_differential_evolution: bool = False  # Global optimization method
    
    # Output options
    verbose: bool = True
    create_plots: bool = False
    plot_components: bool = True
    save_individual_fits: bool = False

# =====================================
# UTILITY FUNCTIONS
# =====================================

def find_column(df: pd.DataFrame, target_name: str) -> str:
    """Find column name with case-insensitive matching."""
    column_map = {col.lower(): col for col in df.columns}
    key = target_name.lower()
    if key not in column_map:
        raise KeyError(f"Required column '{target_name}' not found in data. "
                      f"Available columns: {list(df.columns)}")
    return column_map[key]

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean input data."""
    print(f"Initial data shape: {df.shape}")
    
    # Check for required columns
    column_mapping = {}
    for required_col in REQUIRED_COLUMNS:
        column_mapping[required_col] = find_column(df, required_col)
    
    # Create standardized column names
    df_clean = df.copy()
    for standard_name, actual_name in column_mapping.items():
        if standard_name != actual_name:
            df_clean[standard_name] = df_clean[actual_name]
    
    # Remove rows with invalid data
    numeric_cols = ["R_kpc", "Vobs_kms", "eVobs_kms", "Vgas_kms", "Vdisk_kms", "Vbul_kms"]
    
    print("Converting columns to numeric...")
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    print("Creating filtering masks...")
    
    # Create individual masks to avoid pandas boolean operation issues
    try:
        mask_list = []
        
        # Check R > 0
        mask_r = df_clean["R_kpc"].gt(0)
        mask_list.append(mask_r)
        print(f"R > 0: {mask_r.sum()} valid points")
        
        # Check Vobs > 0  
        mask_v = df_clean["Vobs_kms"].gt(0)
        mask_list.append(mask_v)
        print(f"Vobs > 0: {mask_v.sum()} valid points")
        
        # Check eVobs >= 0
        mask_e = df_clean["eVobs_kms"].ge(0)
        mask_list.append(mask_e)
        print(f"eVobs >= 0: {mask_e.sum()} valid points")
        
        # Check for no NaN values
        mask_na = df_clean[numeric_cols].notna().all(axis=1)
        mask_list.append(mask_na)
        print(f"No NaN: {mask_na.sum()} valid points")
        
        # Combine all masks
        final_mask = mask_list[0]
        for m in mask_list[1:]:
            final_mask = final_mask & m
            
    except Exception as e:
        print(f"Error creating masks: {e}")
        # Fallback to more basic filtering
        print("Using fallback filtering method...")
        
        # Drop rows with any NaN in numeric columns
        df_clean = df_clean.dropna(subset=numeric_cols)
        
        # Use query method for filtering
        df_clean = df_clean.query('R_kpc > 0 and Vobs_kms > 0 and eVobs_kms >= 0')
        
        initial_count = len(df)
        final_count = len(df_clean)
        
        if final_count < initial_count:
            print(f"Data cleaning: removed {initial_count - final_count} invalid rows "
                  f"({100*(initial_count - final_count)/initial_count:.1f}%)")
        
        return df_clean
    
    # Apply the combined mask
    initial_count = len(df)
    df_clean = df_clean[final_mask].copy()
    final_count = len(df_clean)
    
    print(f"Final data shape: {df_clean.shape}")
    
    if final_count < initial_count:
        print(f"Data cleaning: removed {initial_count - final_count} invalid rows "
              f"({100*(initial_count - final_count)/initial_count:.1f}%)")
    
    return df_clean

# =====================================
# PHYSICS MODELS
# =====================================

def baryonic_velocity(R: np.ndarray, V_gas: np.ndarray, V_disk: np.ndarray, 
                     V_bulge: np.ndarray, f_disk: float) -> np.ndarray:
    """
    Calculate total baryonic velocity contribution.
    
    Parameters:
    -----------
    R : array_like
        Radial distances [kpc]
    V_gas, V_disk, V_bulge : array_like
        Component velocities [km/s]
    f_disk : float
        Disk mass-to-light ratio scaling factor
        
    Returns:
    --------
    V_baryons : array
        Total baryonic velocity [km/s]
    """
    V_gas = np.asarray(V_gas, dtype=float)
    V_disk = np.asarray(V_disk, dtype=float)
    V_bulge = np.asarray(V_bulge, dtype=float)
    
    V_baryons_squared = V_gas**2 + (f_disk * V_disk)**2 + (f_disk * V_bulge)**2
    return np.sqrt(np.maximum(0.0, V_baryons_squared))

def soliton_velocity_profile(R: np.ndarray, rho_c: float, r_c: float) -> np.ndarray:
    """
    Calculate velocity from quantum soliton dark matter profile.
    
    Uses pseudo-isothermal density profile:
    ρ(r) = ρ_c / (1 + (r/r_c)²)
    
    Enclosed mass: M(<R) = 4π ρ_c r_c³ [x - arctan(x)] where x = R/r_c
    Circular velocity: V(R) = √(G M(<R) / R)
    
    Parameters:
    -----------
    R : array_like
        Radial distances [kpc]
    rho_c : float
        Central density [M_sun/kpc³]
    r_c : float
        Core radius [kpc]
        
    Returns:
    --------
    V_soliton : array
        Soliton contribution to circular velocity [km/s]
    """
    R = np.asarray(R, dtype=float)
    R_safe = np.where(R <= 0.0, 1e-9, R)  # Avoid division by zero
    r_c = max(float(r_c), 1e-9)
    
    x = R_safe / r_c
    enclosed_mass_factor = x - np.arctan(x)
    M_enclosed = 4.0 * np.pi * rho_c * r_c**3 * enclosed_mass_factor
    
    V_squared = G_GAL * M_enclosed / R_safe
    return np.sqrt(np.maximum(0.0, V_squared))

def phonon_velocity_mond(R: np.ndarray, V_baryons: np.ndarray, 
                        a0: float = A0_MOND) -> np.ndarray:
    """
    Calculate MOND-like phonon velocity contribution.
    
    Uses the standard MOND μ-function:
    V_phonon² = a₀ R μ(V_baryons²/(a₀R))
    
    where μ(x) = x / √(1 + x²) is the simple interpolating function.
    
    Parameters:
    -----------
    R : array_like
        Radial distances [kpc]
    V_baryons : array_like
        Baryonic velocity [km/s]
    a0 : float
        MOND acceleration scale [(km/s)²/kpc]
        
    Returns:
    --------
    V_phonon : array
        Phonon contribution to circular velocity [km/s]
    """
    R = np.asarray(R, dtype=float)
    R_safe = np.where(R <= 0.0, 1e-9, R)
    V_baryons = np.maximum(1e-9, np.asarray(V_baryons, dtype=float))
    
    # Calculate MOND parameter
    a_baryons = V_baryons**2 / R_safe
    x = a_baryons / a0
    
    # Simple μ-function
    mu = x / np.sqrt(1.0 + x**2)
    
    V_phonon_squared = a0 * R_safe * mu
    return np.sqrt(np.maximum(0.0, V_phonon_squared))

def total_rotation_curve(R: np.ndarray, V_gas: np.ndarray, V_disk: np.ndarray,
                        V_bulge: np.ndarray, f_disk: float, rho_c: float, 
                        r_c: float, use_phonon: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Calculate total rotation curve from all components.
    
    Returns:
    --------
    V_total : array
        Total circular velocity [km/s]
    components : dict
        Individual velocity components for analysis
    """
    V_baryons = baryonic_velocity(R, V_gas, V_disk, V_bulge, f_disk)
    V_soliton = soliton_velocity_profile(R, rho_c, r_c)
    
    components = {
        'baryons': V_baryons,
        'soliton': V_soliton,
        'phonon': np.zeros_like(R)
    }
    
    if use_phonon:
        V_phonon = phonon_velocity_mond(R, V_baryons)
        components['phonon'] = V_phonon
        V_total_squared = V_baryons**2 + V_soliton**2 + V_phonon**2
    else:
        V_total_squared = V_baryons**2 + V_soliton**2
    
    V_total = np.sqrt(np.maximum(0.0, V_total_squared))
    
    return V_total, components

# =====================================
# FITTING FRAMEWORK
# =====================================

class RotationCurveFitter:
    """Main class for fitting rotation curves to SPARC data."""
    
    def __init__(self, config: FitConfiguration):
        self.config = config
        self.fit_results = {}
        
    def calculate_effective_errors(self, V_obs: np.ndarray, V_err: np.ndarray) -> np.ndarray:
        """Calculate effective errors with floor."""
        error_floor = self.config.error_floor_fraction * V_obs
        return np.sqrt(V_err**2 + error_floor**2)
    
    def create_fit_function(self, R_data: np.ndarray, V_gas: np.ndarray,
                           V_disk: np.ndarray, V_bulge: np.ndarray):
        """Create fitting function for scipy.optimize."""
        
        def fit_func(R_fit, f_disk, rho_c, r_c):
            # Interpolate baryonic components to fit radii
            V_gas_interp = np.interp(R_fit, R_data, V_gas)
            V_disk_interp = np.interp(R_fit, R_data, V_disk)
            V_bulge_interp = np.interp(R_fit, R_data, V_bulge)
            
            V_total, _ = total_rotation_curve(
                R_fit, V_gas_interp, V_disk_interp, V_bulge_interp,
                f_disk, rho_c, r_c, self.config.use_phonon_term
            )
            return V_total
        
        return fit_func
    
    def estimate_initial_parameters(self, R: np.ndarray, V_obs: np.ndarray) -> List[float]:
        """Estimate reasonable initial parameters."""
        f_disk_init = 1.0  # Start with canonical disk mass
        
        # Estimate core density from peak velocity
        V_max = np.max(V_obs)
        rho_c_init = 1e7  # Typical soliton core density
        
        # Estimate core radius from velocity profile shape
        R_half_max = R[np.argmin(np.abs(V_obs - V_max/2))] if len(R) > 1 else 1.0
        r_c_init = max(0.1, min(10.0, R_half_max))
        
        return [f_disk_init, rho_c_init, r_c_init]
    
    def fit_galaxy(self, galaxy_data: pd.DataFrame, galaxy_name: str) -> Optional[Dict]:
        """Fit rotation curve for a single galaxy."""
        
        if len(galaxy_data) < self.config.min_points_per_galaxy:
            if self.config.verbose:
                print(f"Skipping {galaxy_name}: insufficient data points ({len(galaxy_data)})")
            return None
        
        # Extract data arrays
        R = galaxy_data["R_kpc"].values
        V_obs = galaxy_data["Vobs_kms"].values
        V_err = galaxy_data["eVobs_kms"].values
        V_gas = galaxy_data["Vgas_kms"].values
        V_disk = galaxy_data["Vdisk_kms"].values
        V_bulge = galaxy_data["Vbul_kms"].values
        
        # Calculate effective errors
        sigma_eff = self.calculate_effective_errors(V_obs, V_err)
        
        # Set up fitting
        fit_func = self.create_fit_function(R, V_gas, V_disk, V_bulge)
        initial_params = self.estimate_initial_parameters(R, V_obs)
        
        parameter_bounds = (
            [self.config.f_disk_bounds[0], self.config.rho_c_bounds[0], self.config.r_c_bounds[0]],
            [self.config.f_disk_bounds[1], self.config.rho_c_bounds[1], self.config.r_c_bounds[1]]
        )
        
        try:
            if self.config.use_differential_evolution:
                # Global optimization approach
                def objective(params):
                    f_disk, rho_c, r_c = params
                    V_model = fit_func(R, f_disk, rho_c, r_c)
                    residuals = (V_obs - V_model) / sigma_eff
                    return np.sum(residuals**2)
                
                result = differential_evolution(
                    objective, parameter_bounds, seed=42, maxiter=1000
                )
                optimal_params = result.x
                param_errors = np.full(3, np.nan)  # No covariance from DE
                
            else:
                # Local optimization with Levenberg-Marquardt
                optimal_params, param_covariance = curve_fit(
                    fit_func, R, V_obs,
                    sigma=sigma_eff,
                    p0=initial_params,
                    bounds=parameter_bounds,
                    absolute_sigma=True,
                    maxfev=5000
                )
                
                if param_covariance is not None:
                    param_errors = np.sqrt(np.diag(param_covariance))
                else:
                    param_errors = np.full(3, np.nan)
        
        except Exception as e:
            if self.config.verbose:
                print(f"Fit failed for {galaxy_name}: {str(e)}")
            return None
        
        # Extract fitted parameters
        f_disk_fit, rho_c_fit, r_c_fit = optimal_params
        f_disk_err, rho_c_err, r_c_err = param_errors
        
        # Calculate model predictions and residuals
        V_model, components = total_rotation_curve(
            R, V_gas, V_disk, V_bulge, f_disk_fit, rho_c_fit, r_c_fit,
            self.config.use_phonon_term
        )
        
        residuals = (V_obs - V_model) / sigma_eff
        chi_squared = np.sum(residuals**2)
        degrees_of_freedom = len(R) - 3  # 3 fitted parameters
        chi_squared_reduced = chi_squared / max(1, degrees_of_freedom)
        
        # Calculate derived quantities
        soliton_mass_5rc = 4.0 * np.pi * rho_c_fit * r_c_fit**3 * (5.0 - np.arctan(5.0))
        
        # Rough virial mass estimate from outermost point
        R_outer = R[-1]
        V_outer = V_model[-1]
        virial_mass_estimate = (V_outer**2 * R_outer) / G_GAL
        virial_radius_estimate = ((3.0 * virial_mass_estimate) / 
                                (4.0 * np.pi * 200.0 * RHO_CRIT))**(1.0/3.0)
        
        return {
            # Galaxy identification
            'galaxy_name': galaxy_name,
            'n_data_points': len(R),
            
            # Fitted parameters
            'f_disk': f_disk_fit,
            'f_disk_error': f_disk_err,
            'rho_c_Msun_kpc3': rho_c_fit,
            'rho_c_error': rho_c_err,
            'r_c_kpc': r_c_fit,
            'r_c_error': r_c_err,
            
            # Goodness of fit
            'chi_squared': chi_squared,
            'chi_squared_reduced': chi_squared_reduced,
            'degrees_of_freedom': degrees_of_freedom,
            
            # Derived masses
            'soliton_mass_5rc_Msun': soliton_mass_5rc,
            'virial_mass_estimate_Msun': virial_mass_estimate,
            'virial_radius_estimate_kpc': virial_radius_estimate,
            
            # Configuration used
            'used_phonon_term': self.config.use_phonon_term,
            'error_floor_fraction': self.config.error_floor_fraction,
            
            # Data for plotting/analysis
            'data': {
                'R': R,
                'V_obs': V_obs,
                'V_err': V_err,
                'sigma_eff': sigma_eff,
                'V_model': V_model,
                'components': components,
                'residuals': residuals
            }
        }
    
    def fit_all_galaxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit rotation curves for all galaxies in dataset."""
        
        galaxy_names = df['Galaxy'].unique()
        results = []
        
        print(f"Fitting rotation curves for {len(galaxy_names)} galaxies...")
        
        for i, galaxy_name in enumerate(galaxy_names):
            if self.config.verbose and (i + 1) % 25 == 0:
                print(f"Progress: {i + 1}/{len(galaxy_names)} galaxies processed")
            
            galaxy_data = df[df['Galaxy'] == galaxy_name].copy()
            galaxy_data = galaxy_data.sort_values('R_kpc').reset_index(drop=True)
            
            result = self.fit_galaxy(galaxy_data, galaxy_name)
            if result is not None:
                results.append(result)
                self.fit_results[galaxy_name] = result
        
        if not results:
            raise ValueError("No successful fits obtained!")
        
        # Convert to DataFrame for easy analysis
        summary_data = []
        for result in results:
            summary_row = {k: v for k, v in result.items() if k != 'data'}
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        
        if self.config.verbose:
            self.print_fit_statistics(summary_df)
        
        return summary_df
    
    def print_fit_statistics(self, summary_df: pd.DataFrame):
        """Print summary statistics of fitting results."""
        print(f"\n{'='*60}")
        print("ROTATION CURVE FITTING SUMMARY")
        print(f"{'='*60}")
        
        print(f"Successfully fitted: {len(summary_df)} galaxies")
        
        # Chi-squared statistics
        chi2_red = summary_df['chi_squared_reduced'].replace([np.inf, -np.inf], np.nan).dropna()
        if len(chi2_red) > 0:
            print(f"\nReduced χ² statistics:")
            print(f"  Mean:   {chi2_red.mean():.3f}")
            print(f"  Median: {chi2_red.median():.3f}")
            print(f"  Std:    {chi2_red.std():.3f}")
            print(f"  Range:  [{chi2_red.min():.3f}, {chi2_red.max():.3f}]")
        
        # Parameter statistics
        for param in ['f_disk', 'rho_c_Msun_kpc3', 'r_c_kpc']:
            values = summary_df[param].replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) > 0:
                print(f"\n{param}:")
                print(f"  Mean:   {values.mean():.3e}")
                print(f"  Median: {values.median():.3e}")
                print(f"  Range:  [{values.min():.3e}, {values.max():.3e}]")
    
    def create_diagnostic_plots(self, galaxy_name: str, save_path: Optional[str] = None):
        """Create comprehensive diagnostic plots for a galaxy."""
        
        if galaxy_name not in self.fit_results:
            raise ValueError(f"No fit results found for galaxy {galaxy_name}")
        
        result = self.fit_results[galaxy_name]
        data = result['data']
        
        # Set up the plot
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main rotation curve plot
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Plot data with error bars
        ax1.errorbar(data['R'], data['V_obs'], yerr=data['sigma_eff'], 
                    fmt='o', capsize=3, alpha=0.7, label='Observed', color='black')
        
        # Plot model and components
        R_smooth = np.linspace(data['R'].min(), data['R'].max(), 200)
        
        # Interpolate baryonic components for smooth plotting
        V_gas_smooth = np.interp(R_smooth, data['R'], data['components']['baryons'])
        V_disk_smooth = np.interp(R_smooth, data['R'], np.zeros_like(data['R']))  # Simplified
        V_bulge_smooth = np.interp(R_smooth, data['R'], np.zeros_like(data['R']))  # Simplified
        
        V_total_smooth, components_smooth = total_rotation_curve(
            R_smooth, V_gas_smooth, V_disk_smooth, V_bulge_smooth,
            result['f_disk'], result['rho_c_Msun_kpc3'], result['r_c_kpc'],
            self.config.use_phonon_term
        )
        
        ax1.plot(R_smooth, V_total_smooth, '-', linewidth=2, label='Total Model', color='red')
        
        if self.config.plot_components:
            ax1.plot(R_smooth, components_smooth['baryons'], '--', 
                    label='Baryons', alpha=0.8, color='blue')
            ax1.plot(R_smooth, components_smooth['soliton'], '--', 
                    label='Soliton', alpha=0.8, color='green')
            if self.config.use_phonon_term:
                ax1.plot(R_smooth, components_smooth['phonon'], '--', 
                        label='Phonon', alpha=0.8, color='orange')
        
        ax1.set_xlabel('Radius (kpc)')
        ax1.set_ylabel('Velocity (km/s)')
        ax1.set_title(f'{galaxy_name} - Rotation Curve Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.plot(data['R'], data['residuals'], 'o', alpha=0.7)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Radius (kpc)')
        ax2.set_ylabel('Residuals (σ)')
        ax2.set_title('Fit Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Parameter summary
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis('off')
        
        param_text = f"""Fit Parameters:
f_disk = {result['f_disk']:.3f} ± {result['f_disk_error']:.3f}
ρ_c = {result['rho_c_Msun_kpc3']:.2e} M☉/kpc³
r_c = {result['r_c_kpc']:.3f} ± {result['r_c_error']:.3f} kpc

Goodness of Fit:
χ² = {result['chi_squared']:.2f}
χ²_red = {result['chi_squared_reduced']:.3f}
DOF = {result['degrees_of_freedom']}

Derived Masses:
M_soliton(5r_c) = {result['soliton_mass_5rc_Msun']:.2e} M☉
M_virial ≈ {result['virial_mass_estimate_Msun']:.2e} M☉
R_virial ≈ {result['virial_radius_estimate_kpc']:.1f} kpc"""
        
        ax3.text(0.05, 0.95, param_text, transform=ax3.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Chi-squared distribution plot
        ax4 = fig.add_subplot(gs[1, 1])
        residuals_sorted = np.sort(data['residuals'])
        theoretical_quantiles = np.linspace(-3, 3, len(residuals_sorted))
        ax4.plot(theoretical_quantiles, residuals_sorted, 'o', alpha=0.7)
        ax4.plot([-3, 3], [-3, 3], 'r--', alpha=0.7)
        ax4.set_xlabel('Theoretical Quantiles')
        ax4.set_ylabel('Sample Quantiles')
        ax4.set_title('Q-Q Plot (Residuals)')
        ax4.grid(True, alpha=0.3)
        
        # Velocity profile comparison
        ax5 = fig.add_subplot(gs[1, 2])
        
        # Plot observed vs predicted
        ax5.plot(data['V_obs'], data['V_model'], 'o', alpha=0.7)
        v_range = [min(data['V_obs'].min(), data['V_model'].min()),
                   max(data['V_obs'].max(), data['V_model'].max())]
        ax5.plot(v_range, v_range, 'r--', alpha=0.7)
        ax5.set_xlabel('Observed Velocity (km/s)')
        ax5.set_ylabel('Model Velocity (km/s)')
        ax5.set_title('Observed vs Model')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle(f'Diagnostic Plots for {galaxy_name}', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

# =====================================
# ANALYSIS FUNCTIONS
# =====================================

def create_summary_plots(summary_df: pd.DataFrame, save_dir: Optional[str] = None):
    """Create summary plots for the entire dataset analysis."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # 1. Chi-squared distribution
    chi2_red = summary_df['chi_squared_reduced'].replace([np.inf, -np.inf], np.nan).dropna()
    axes[0].hist(chi2_red, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(chi2_red.median(), color='red', linestyle='--', 
                   label=f'Median = {chi2_red.median():.2f}')
    axes[0].set_xlabel('Reduced χ²')
    axes[0].set_ylabel('Number of Galaxies')
    axes[0].set_title('Distribution of Reduced χ²')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Parameter distributions
    params = ['f_disk', 'rho_c_Msun_kpc3', 'r_c_kpc']
    param_labels = ['Disk Mass Factor', 'Core Density [M☉/kpc³]', 'Core Radius [kpc]']
    
    for i, (param, label) in enumerate(zip(params, param_labels)):
        ax = axes[i + 1]
        values = summary_df[param].replace([np.inf, -np.inf], np.nan).dropna()
        
        if param == 'rho_c_Msun_kpc3':
            # Log scale for density
            ax.hist(np.log10(values), bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(f'log₁₀({label})')
        else:
            ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(label)
        
        ax.set_ylabel('Number of Galaxies')
        ax.set_title(f'Distribution of {label}')
        ax.grid(True, alpha=0.3)
    
    # 4. Core mass vs core radius
    axes[4].scatter(summary_df['r_c_kpc'], summary_df['soliton_mass_5rc_Msun'], 
                   alpha=0.6, s=30)
    axes[4].set_xlabel('Core Radius [kpc]')
    axes[4].set_ylabel('Soliton Mass (5r_c) [M☉]')
    axes[4].set_title('Soliton Core Mass vs Radius')
    axes[4].set_yscale('log')
    axes[4].grid(True, alpha=0.3)
    
    # 5. Goodness of fit vs galaxy properties
    n_points = summary_df['n_data_points']
    chi2_color = summary_df['chi_squared_reduced']
    scatter = axes[5].scatter(n_points, summary_df['f_disk'], 
                             c=chi2_color, s=30, alpha=0.7, cmap='viridis')
    axes[5].set_xlabel('Number of Data Points')
    axes[5].set_ylabel('Disk Mass Factor')
    axes[5].set_title('Fit Quality vs Data Density')
    plt.colorbar(scatter, ax=axes[5], label='Reduced χ²')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'summary_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary plots saved to {save_path}")
    
    plt.show()

def export_detailed_results(fitter: RotationCurveFitter, output_dir: str):
    """Export detailed results for each galaxy."""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create individual galaxy files
    for galaxy_name, result in fitter.fit_results.items():
        data = result['data']
        
        # Create detailed data table
        detailed_data = pd.DataFrame({
            'R_kpc': data['R'],
            'V_observed_kms': data['V_obs'],
            'V_error_kms': data['V_err'],
            'sigma_effective_kms': data['sigma_eff'],
            'V_model_kms': data['V_model'],
            'V_baryons_kms': data['components']['baryons'],
            'V_soliton_kms': data['components']['soliton'],
            'V_phonon_kms': data['components']['phonon'],
            'residuals_sigma': data['residuals']
        })
        
        # Add metadata header
        metadata = f"""# Rotation curve fit results for {galaxy_name}
# Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
#
# Fitted parameters:
# f_disk = {result['f_disk']:.6f} ± {result['f_disk_error']:.6f}
# rho_c = {result['rho_c_Msun_kpc3']:.6e} ± {result['rho_c_error']:.6e} [M_sun/kpc^3]
# r_c = {result['r_c_kpc']:.6f} ± {result['r_c_error']:.6f} [kpc]
#
# Goodness of fit:
# chi_squared = {result['chi_squared']:.6f}
# chi_squared_reduced = {result['chi_squared_reduced']:.6f}
# degrees_of_freedom = {result['degrees_of_freedom']}
#
# Derived quantities:
# M_soliton_5rc = {result['soliton_mass_5rc_Msun']:.6e} [M_sun]
# M_virial_estimate = {result['virial_mass_estimate_Msun']:.6e} [M_sun]
# R_virial_estimate = {result['virial_radius_estimate_kpc']:.6f} [kpc]
#
"""
        
        file_path = output_path / f'{galaxy_name}_detailed_fit.csv'
        with open(file_path, 'w') as f:
            f.write(metadata)
        
        detailed_data.to_csv(file_path, mode='a', index=False)

# =====================================
# COMMAND LINE INTERFACE
# =====================================

def main():
    """Main function for command-line execution."""
    
    parser = argparse.ArgumentParser(
        description="Advanced rotation curve fitting for SPARC galaxy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv --output results/
  %(prog)s data.csv --plot-galaxies NGC2403 DDO154 --save-plots
  %(prog)s data.csv --no-phonon --error-floor 0.03
        """
    )
    
    # Input/Output arguments
    parser.add_argument('input_file', type=str,
                       help='Input CSV file with SPARC rotation curve data')
    parser.add_argument('--output', '-o', type=str, default='rotation_curve_results',
                       help='Output directory for results (default: rotation_curve_results)')
    
    # Physics model options
    parser.add_argument('--no-phonon', action='store_true',
                       help='Disable MOND-like phonon term')
    parser.add_argument('--error-floor', type=float, default=0.05,
                       help='Fractional error floor (default: 0.05)')
    parser.add_argument('--min-points', type=int, default=5,
                       help='Minimum data points per galaxy (default: 5)')
    
    # Optimization options
    parser.add_argument('--global-opt', action='store_true',
                       help='Use differential evolution (global optimization)')
    parser.add_argument('--f-disk-bounds', nargs=2, type=float, default=[0.1, 3.0],
                       help='Bounds for disk mass factor (default: 0.1 3.0)')
    parser.add_argument('--rho-bounds', nargs=2, type=float, default=[1e4, 1e12],
                       help='Bounds for core density in M_sun/kpc^3 (default: 1e4 1e12)')
    parser.add_argument('--rc-bounds', nargs=2, type=float, default=[0.05, 50.0],
                       help='Bounds for core radius in kpc (default: 0.05 50.0)')
    
    # Output options
    parser.add_argument('--plot-galaxies', nargs='*', type=str,
                       help='Create diagnostic plots for specified galaxies')
    parser.add_argument('--plot-summary', action='store_true',
                       help='Create summary plots for entire dataset')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to files instead of displaying')
    parser.add_argument('--export-detailed', action='store_true',
                       help='Export detailed per-galaxy results')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found!")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Configure fitting
    config = FitConfiguration(
        error_floor_fraction=args.error_floor,
        min_points_per_galaxy=args.min_points,
        f_disk_bounds=tuple(args.f_disk_bounds),
        rho_c_bounds=tuple(args.rho_bounds),
        r_c_bounds=tuple(args.rc_bounds),
        use_phonon_term=not args.no_phonon,
        use_differential_evolution=args.global_opt,
        verbose=not args.quiet,
        create_plots=bool(args.plot_galaxies),
        save_individual_fits=args.export_detailed
    )
    
    if not args.quiet:
        print("SPARC Rotation Curve Fitter")
        print("=" * 50)
        print(f"Input file: {input_path}")
        print(f"Output directory: {output_dir}")
        print(f"Phonon term: {'enabled' if config.use_phonon_term else 'disabled'}")
        print(f"Error floor: {config.error_floor_fraction:.1%}")
        print(f"Optimization: {'global (DE)' if config.use_differential_evolution else 'local (LM)'}")
    
    try:
        # Load and validate data
        if not args.quiet:
            print("\nLoading and validating data...")
        
        df = pd.read_csv(input_path)
        df_clean = validate_data(df)
        
        if not args.quiet:
            n_galaxies = df_clean['Galaxy'].nunique()
            print(f"Successfully loaded {len(df_clean)} data points for {n_galaxies} galaxies")
        
        # Perform fitting
        if not args.quiet:
            print("\nFitting rotation curves...")
        
        fitter = RotationCurveFitter(config)
        summary_df = fitter.fit_all_galaxies(df_clean)
        
        # Save summary results
        summary_path = output_dir / 'rotation_curve_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        
        if not args.quiet:
            print(f"\nSummary results saved to: {summary_path}")
        
        # Create individual galaxy plots
        if args.plot_galaxies is not None:
            if not args.quiet:
                print("\nCreating diagnostic plots...")
            
            galaxies_to_plot = args.plot_galaxies if args.plot_galaxies else summary_df['galaxy_name'].iloc[:3]
            
            for galaxy_name in galaxies_to_plot:
                if galaxy_name in fitter.fit_results:
                    save_path = None
                    if args.save_plots:
                        save_path = output_dir / f'{galaxy_name}_diagnostic.png'
                    
                    fitter.create_diagnostic_plots(galaxy_name, save_path)
                else:
                    print(f"Warning: No fit results found for galaxy '{galaxy_name}'")
        
        # Create summary plots
        if args.plot_summary:
            if not args.quiet:
                print("\nCreating summary plots...")
            
            save_dir = output_dir if args.save_plots else None
            create_summary_plots(summary_df, save_dir)
        
        # Export detailed results
        if args.export_detailed:
            if not args.quiet:
                print("\nExporting detailed results...")
            
            detailed_dir = output_dir / 'detailed_fits'
            export_detailed_results(fitter, detailed_dir)
            print(f"Detailed results exported to: {detailed_dir}")
        
        if not args.quiet:
            print(f"\nAnalysis complete! Results saved in: {output_dir}")
            
            # Final statistics
            successful_fits = len(summary_df)
            total_galaxies = df_clean['Galaxy'].nunique()
            success_rate = 100 * successful_fits / total_galaxies
            
            print(f"\nFinal Statistics:")
            print(f"  Successful fits: {successful_fits}/{total_galaxies} ({success_rate:.1f}%)")
            
            if successful_fits > 0:
                median_chi2 = summary_df['chi_squared_reduced'].median()
                print(f"  Median reduced χ²: {median_chi2:.3f}")
                
                # Flag potentially problematic fits
                high_chi2 = summary_df[summary_df['chi_squared_reduced'] > 3.0]
                if len(high_chi2) > 0:
                    print(f"  High χ² fits (>3.0): {len(high_chi2)} galaxies")
    
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()