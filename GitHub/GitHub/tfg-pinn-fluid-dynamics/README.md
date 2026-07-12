# Physics-Informed Neural Networks for Fluid Dynamics PDEs

Bachelor's Thesis (Trabajo de Fin de Grado en Física) — Universidad de Zaragoza, 2024
Author: Guillermo Gracia Rebullida · Advisors: Sergio Gutiérrez Rodrigo, Adrián Navas Montilla

## Overview

This project studies whether **Physics-Informed Neural Networks (PINNs)** can solve partial differential equations from computational fluid dynamics (CFD), and how their accuracy depends on the mathematical formulation of the equation and the type of solution (smooth vs. discontinuous).

PINNs are implemented from scratch in Python (TensorFlow / Keras): instead of fitting the network to labelled data, the PDE residual, initial condition and boundary condition are baked directly into the loss function via automatic differentiation, and the network is trained by minimizing that composite loss at randomly sampled collocation points.

Three PDEs are solved, all simplifications or extensions of the Navier-Stokes equations, and validated against analytical or finite-volume reference solutions:

- **Linear transport equation** `∂u/∂t + c·∇u = 0` — passive advection of a scalar (e.g. a pollutant) in a fluid.
- **Inviscid Burgers' equation** `∂v/∂t + v·∂v/∂x = 0` — a nonlinear model equation that develops shock waves and rarefaction waves, widely used in CFD to benchmark numerical schemes.
- **Convection-diffusion-reaction equation** (exploratory extension, not in the final written thesis) — `∂T/∂t + v·∂T/∂x = α·∂²T/∂x² - R(T - T∞)`, a heat-transport equation with a linear decay/reaction term.

## Method

A PINN embeds the physics directly into the loss function instead of only fitting data points. For a PDE on domain `D = [a,b] × [0,T]`, the network is trained by minimizing a composite loss:

```
Loss = C_PDE (residual of the PDE at collocation points, via automatic differentiation)
     + C_0   (initial condition mismatch)
     + C_∂D  (boundary condition mismatch)
```

Networks are small feed-forward MLPs (2 inputs `(x,t)`, 1 output), trained with gradient-based optimization (Adam / backpropagation, manual `GradientTape` training loops — no `model.fit`). Hyperparameters (layers, neurons per layer, activation, learning rate, number of collocation points per loss term) were tuned per experiment — see the full write-up for the exact tables.

## Key results

**Transport equation:**
- The PINN matches the analytical solution `u(x,t) = g(x-t)` almost exactly for smooth initial conditions (sine wave) on a small domain.
- Accuracy degrades as the spatial/temporal domain grows, and near discontinuities (step function) or non-differentiable points (triangular peak) — the network systematically predicts these regions worse than smooth ones (Gaussian, sine).

**Burgers' equation:**
- Formulating the PDE loss in **conservative form** (based on flux `f = v²/2`) gives noticeably better results than the naive **convective form**, which produced incorrect solutions as time advanced. This suggests the mathematical formulation of the residual matters as much as network architecture for nonlinear PDEs.
- The network correctly reproduces rarefaction waves and captures the qualitative shock-formation behavior (a sine initial condition steepens into a shock over time), consistent with the physics.
- Cross-validated against a hand-written finite-volume (finite-difference upwind) solver for the inviscid case.

## Repository contents

```
notebooks/
  linear_transport_pinn.ipynb            # core PINN: linear transport equation
  burgers_inviscid_pinn.ipynb            # core PINN: inviscid Burgers, incl. finite-volume cross-check
  burgers_viscous_pinn.ipynb             # core PINN: viscous Burgers (traveling-front solution)
  convection_diffusion_reaction_pinn.py  # exploratory: convection-diffusion with linear reaction term
scripts/
  read_training_data.py        # loads solver output and overlays the analytical solution
  plot_activation_function.py  # plots the step/sigmoid/tanh/ReLU activation functions used in the thesis
  characteristic_lines.py      # plots characteristic lines / rarefaction fan for Burgers' equation
  label_image.py               # utility to annotate exported result figures
results/
  transport/   # sine, step and combined-function GIFs of PINN vs. analytical solution over time
  burgers/     # step and sine-wave GIFs of PINN vs. weak solution over time
  characteristics1.png, characteristics2.png
thesis_full_report_es.md   # full thesis write-up (original Spanish text; section headers in English)
```

The notebooks were originally run on Google Colab (hence the Google Drive mount cell at the top of each — safe to ignore/remove when running locally) and require `tensorflow`, `numpy`, `matplotlib` and `imageio`.

## Tech stack

Python, TensorFlow, Keras, NumPy, Matplotlib, imageio, SymPy.
