# QSTF Bullet Cluster Simulation Script

This README explains the operation of the Quantum Spacetime Fluid (QSTF) Bullet Cluster simulation script in detail, at a level suitable for university students or researchers. The simulation models the Bullet Cluster merger using a quantum hydrodynamic approach and aims to reproduce observed gravitational lensing features without invoking particle dark matter.

---

1. Theoretical Motivation

---

This script simulates the Bullet Cluster merger using the Quantum Spacetime Fluid (QSTF) model. 
In this framework, spacetime behaves as a quantum superfluid, and massive structures (like galaxy clusters) 
are modeled as solitonic excitations—localized, stable solutions of the Gross-Pitaevskii equation (GPE).

Key goal: To show that, after a collision, the gravitational lensing peak (normally attributed to dark matter) 
can be explained by vorticity-induced “effective mass” in the quantum fluid, without any particle dark matter.

---

2. Simulation Setup

---

a. Physical and Computational Constants

* The simulation uses astrophysically relevant units: kiloparsecs (kpc), solar masses (Msun), and Myr (million years).
* Constants for Planck’s constant (ħ), gravitational constant (G), and conversion factors are defined for scaling the quantum terms to cosmic size.

b. Grid Construction

* The simulation domain is a 3D cube:

  * N x N x N grid points (e.g., 512³ in full scale, 32³ for the demo)
  * Physical size: L x L x L kpc (e.g., 500 kpc in the real run)
* Arrays x, y, z are evenly spaced coordinates; meshgrids X, Y, Z define each point in the 3D simulation volume.

---

3. Initial Conditions: Solitons and Potentials

---

a. Two Solitons

* Two sech-shaped solitons are initialized as the initial wavefunctions (Psi1, Psi2), representing the main and sub-cluster, respectively.
* Each soliton has:

  * Width (w1, w2), amplitude (A1, A2), and initial offset along x (to represent post-collision positions).
  * A relative phase shift (arctan(A2/A1)) to reduce scattering and mimic observed collisionless behavior.

b. Combined Wavefunction

* The total initial wavefunction is a sum: Psi = Psi1 + Psi2 \* exp(i \* phase\_shift)

c. External Potential

* The baryonic gas is modeled as a static Newtonian gravitational potential (V\_ext), centered at the origin, with softening to avoid singularity at r=0.

---

4. Numerical Solution: GPE Evolution

---

a. The Gross-Pitaevskii Equation (GPE)

* The core equation describes a nonlinear, dispersive wave system:
  iħ ∂Ψ/∂t = \[ -ħ²/2m ∇² + V\_ext + g|Ψ|² ] Ψ
* Here, Ψ is the wavefunction of the quantum fluid, g is the self-interaction strength.

b. Split-Step Fourier Method

* The evolution is done in timesteps using operator splitting:

  1. Nonlinear step (position space): Advance Ψ with the potential and nonlinearity.
  2. Kinetic step (momentum/Fourier space): Fourier-transform Ψ, advance with the kinetic operator (Laplacian is diagonal in k-space), then inverse Fourier-transform back.
* This method is efficient and stable for wave equations in periodic boxes.

c. Normalization

* At each step, the wavefunction is renormalized to keep total density consistent, compensating for numerical drift.

---

5. Output and Analysis

---

a. Density and Phase

* Density: rho = |Ψ|² gives the “mass” density at each point.
* Phase: The local phase gives the velocity field via the gradient.

b. Velocity and Vorticity

* Velocity field: Calculated as v = (ħ/m) \* ∇(phase).
* Vorticity: The curl of the velocity field, indicating local rotation or “swirls” in the fluid, is computed component-wise.

c. Effective Density

* A physically motivated “effective density” for lensing is calculated:
  rho\_eff = rho + alpha |ω|, where α tunes how much vorticity enhances the apparent mass.

d. Peak Locations and Offset

* The gas peak is where the baryonic potential is deepest (maximum |V\_ext|).
* The lensing peak is where ρ\_eff is maximal.
* The offset is the 3D distance between these peaks — this mimics the observed mass–gas offset in the Bullet Cluster.

e. Mass Ratio

* The ratio of the sum of effective density to the sum of bare density quantifies the “mass boost” from vorticity effects (can reach 4–7×).

---

6. Visualization

---

* A 2D slice (midplane in z) of the effective density is plotted using matplotlib.
* Gas and lensing peaks are marked for visual comparison.

---

7. Physical Interpretation

---

* Why does this matter? In the real Bullet Cluster, gravitational lensing shows “extra” mass offset from baryonic gas.
* Standard cosmology attributes this to dark matter.
* QSTF Model Result: In this simulation, the offset and mass boost emerge naturally from the quantum fluid’s response (specifically,
* vorticity generated during collision) — potentially explaining lensing without invoking particle dark matter.

---

## Summary Table

| Script Step        | Physical Concept                     | Code Elements           |
| ------------------ | ------------------------------------ | ----------------------- |
| Grid setup         | Discretize spacetime                 | meshgrid, N, L, dx      |
| Initial solitons   | Model colliding clusters             | Psi1, Psi2, Psi         |
| External potential | Model baryonic gas                   | V\_ext                  |
| GPE evolution      | Time evolution of system             | for-loop, fftn/ifftn    |
| Output calculation | Observables: density, vorticity, etc | rho, omega, rho\_eff    |
| Peak finding       | Lensing/gas offsets                  | argmax, offset          |
| Visualization      | 2D slice of results                  | plt.imshow, plt.scatter |

---

## Further Reading & Next Steps

* For higher accuracy and physical realism: increase N, timestep count, adjust physical units, and run on GPU.
* For more physics: couple in more detailed baryonic dynamics, include non-Newtonian gravity, or simulate different initial conditions.
* Compare with actual lensing data and published Bullet Cluster observations.
