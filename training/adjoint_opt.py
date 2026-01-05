import numpy as np
import autograd.numpy as npa
from autograd import grad
import matplotlib.pyplot as plt
import os

from ceviche import fdfd_hz

from sdf_gen import MMIGeometry

# Configuration (Consistent with ceviche_sim.py)
# Use dimensionless units c=1
# wavelength = 1.55 um
OMEGA = 2 * np.pi / 1.55
# DL from MMIGeometry (0.04 um)
# NPML = 20

class MMIAdjointOptimizer:
    def __init__(self, steps=100, learning_rate=0.01):
        self.geo = MMIGeometry()
        self.Nx = self.geo.Nx
        self.Ny = self.geo.Ny
        self.dl = self.geo.dl
        self.npml = 20
        self.steps = steps
        self.lr = learning_rate
        
        # Initialize Source and Probes (Static)
        self._init_ports()
        
        # Design Region Mask (Where we allow changes)
        # We allow optimizing the MMI body region
        self.design_mask = self._create_design_mask()
        
        # Initial Design Variable (normalized 0 to 1)
        # Initialize with 0.5 (Gray) in the design region for best gradient flow
        # Or initialize with the starting geometry
        self.design_init = self.geo.get_density() * self.design_mask + 0.5 * self.design_mask * (1 - self.geo.get_density())
        # Actually, let's start with a seed from the current geometry but slightly smoothed
        self.design_init = self.geo.get_density()
        
    def _init_ports(self):
        # Same logic as ceviche_sim.py
        nx, ny = self.Nx, self.Ny
        
        # Source
        self.source_mask = np.zeros((nx, ny), dtype=np.complex128)
        cy = ny // 2
        sigma_pix = 6.0
        y_idx = np.arange(ny)
        # Note: Use standard numpy for initialization, not autograd
        gaussian_profile = np.exp(-0.5 * ((y_idx - cy) / sigma_pix)**2)
        src_x = self.npml + 10
        self.source_mask[src_x, :] = gaussian_profile * 10.0
        
        # Probes
        out_x = nx - self.npml - 5
        mmi_center_y = self.geo.Ly / 2.0
        out_offset = 1.5 
        y_top_idx = int((mmi_center_y + out_offset) / self.dl)
        y_bot_idx = int((mmi_center_y - out_offset) / self.dl)
        
        self.probe_top = np.zeros((nx, ny))
        self.probe_bot = np.zeros((nx, ny))
        self.probe_top[out_x, y_top_idx] = 1.0
        self.probe_bot[out_x, y_bot_idx] = 1.0

    def _create_design_mask(self):
        # Define the rectangular MMI region as the design domain
        # Get coordinates from geometry utils would be better, but hardcoding for consistency with sdf_gen
        mask = np.zeros((self.Nx, self.Ny))
        
        # Re-derive MMI region indices
        mmi_length_pix = int(self.geo.mmi_length / self.dl)
        mmi_width_pix = int(self.geo.mmi_width / self.dl)
        
        cx, cy = self.Nx // 2, self.Ny // 2
        
        x_start = cx - mmi_length_pix // 2
        x_end = cx + mmi_length_pix // 2
        y_start = cy - mmi_width_pix // 2
        y_end = cy + mmi_width_pix // 2
        
        mask[x_start:x_end, y_start:y_end] = 1.0
        return mask

    def objective(self, design_variable):
        # design_variable is (Nx, Ny) array of density values
        # Sigmoid projection or simple clamping could be used, but adam_minimize handles bounds? 
        # No, we usually map variable to density
        
        # Density projection (simple barrier optional, here just using raw variable interpreted as density)
        # Ideally we want binary designs, so we might add a penalty for non-binary values later
        # specific_density = npa.clip(design_variable, 0, 1) # Clip breaks gradient? Use sigmoid
        
        density = design_variable # Start simple
        
        # epsilon logic
        # eps = mask * (eps_si - eps_bg) + eps_bg
        eps_r = density * (12.0 - 1.0) + 1.0
        
        # Simulation
        # fdfd_hz(omega, dl, eps_r, npml)
        sim = fdfd_hz(OMEGA, self.dl, eps_r, [self.npml, self.npml])
        
        # solve() returns (Hz, Ex, Ey) tuple? Or (Ex, Ey, Hz)?
        # Let's assume ceviche standard which returns a tuple.
        # However, fdfd_hz.solve returns (Ex, Ey, Hz) typically.
        # We need to compute intensity.
        
        out_fields = sim.solve(self.source_mask)
        # Note: autograd wrapping might make tuple unpacking tricky if not careful
        # But ceviche examples usually work with unpacking
        
        # Based on my debug earlier, index 0 was Hz.
        Hz = out_fields[0]
        
        # Calculate Power (Transmission)
        # |Hz|^2 at probes
        # We must use npa (autograd numpy) for operations
        
        t_top = npa.sum(npa.abs(Hz * self.probe_top)**2)
        t_bot = npa.sum(npa.abs(Hz * self.probe_bot)**2)
        
        # Optimization Objective: Maximize Power, Minimize Imbalance
        # Loss = - (Total_T - Penalty * Imbalance^2)
        
        total_t = t_top + t_bot
        imbalance = (t_top - t_bot)**2 # Squared difference to penalize strictly
        
        # Weights (arbitrary)
        # Increase imbalance penalty to force 50:50
        loss = -1.0 * (10.0 * total_t - 50.0 * imbalance)
        
        return loss

    def callback(self, iteration, of, params):
        print(f"Iter {iteration}: Loss = {of}")
        
        if iteration % 10 == 0:
            # Visualize
            density = params
            plt.figure()
            plt.imshow(density.T, origin='lower', cmap='gray')
            plt.title(f"Iter {iteration}, Loss {of:.4f}")
            plt.colorbar()
            plt.savefig(f"opt_iter_{iteration:03d}.png")
            plt.close()

    def run(self):
        print("Starting Adjoint Optimization...")
        
        # Initial parameters
        params = self.design_init
        
        # Adam Parameters
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        m, v = 0, 0
        
        # Define Gradient Function using autograd
        # We need to compute value and gradient
        from autograd import value_and_grad
        
        def obj_wrapper(p):
            # Apply design mask to params
            # We treat params as the full density for simplicity, 
            # but only gradients in the design region will matter if we handle it right.
            # actually, let's keep params as full grid.
            eff_density = p * self.design_mask + self.geo.get_density() * (1 - self.design_mask)
            return self.objective(eff_density)
        
        value_grad_func = value_and_grad(obj_wrapper)
        
        for i in range(self.steps):
            loss, grads = value_grad_func(params)
            
            # Mask gradients (only optimize design region)
            grads = grads * self.design_mask
            
            # Adam Update
            m = beta1 * m + (1 - beta1) * grads
            v = beta2 * v + (1 - beta2) * (grads**2)
            m_hat = m / (1 - beta1**(i + 1))
            v_hat = v / (1 - beta2**(i + 1))
            
            # Update params
            params = params - self.lr * m_hat / (npa.sqrt(v_hat) + epsilon)
            
            # Projection / Clipping (0 to 1)
            # Important for density
            params = npa.clip(params, 0, 1)
            
            # Callback / Logging
            self.callback(i, loss, params)
            
        return params

if __name__ == "__main__":
    optimizer = MMIAdjointOptimizer(steps=100, learning_rate=0.02)
    final_design = optimizer.run()
    
    # Save Final Result
    np.save("final_design_adjoint.npy", final_design)
    plt.figure()
    plt.imshow(final_design.T, origin='lower', cmap='gray')
    plt.savefig("final_design_adjoint.png")
    print("Optimization Complete.")
