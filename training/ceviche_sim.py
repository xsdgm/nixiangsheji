import numpy as np
import ceviche
from ceviche import fdfd_hz, fdfd_ez
import matplotlib.pyplot as plt

class CevicheSim:
    def __init__(self, geometry):
        """
        Initialize Simulation Wrapper.
        Args:
            geometry (MMIGeometry): Instance of MMIGeometry containing grid info.
        """
        self.geo = geometry
        self.geo = geometry
        # Use dimensionless units where c=1
        # Length in microns
        self.omega = 2 * np.pi / 1.55 # 1.55um wavelength, c=1
        self.dl = geometry.dl # already in microns
        self.npml = 20 # Perfectly Matched Layer thickness in pixels
        
        # Source & Probe definitions
        self._init_ports()
        
    def _init_ports(self):
        """Define input source and output monitors."""
        nx, ny = self.geo.Nx, self.geo.Ny
        
        # Input Source (Left)
        # Use Gaussian profile (Complex) - Reverting to Run 185 config
        self.source_mask = np.zeros((nx, ny), dtype=np.complex128) 
        cy = ny // 2
        
        sigma_pix = 6.0
        y_idx = np.arange(ny)
        gaussian_profile = np.exp(-0.5 * ((y_idx - cy) / sigma_pix)**2)
        
        src_x = self.npml + 10
        self.source_mask[src_x, :] = gaussian_profile * 10.0
        
        # Output Monitors (Right)
        out_x = nx - self.npml - 5
        mmi_center_y = self.geo.Ly / 2.0
        out_offset = 1.5 
        y_top_idx = int((mmi_center_y + out_offset) / self.geo.dl)
        y_bot_idx = int((mmi_center_y - out_offset) / self.geo.dl)
        
        self.probe_top = np.zeros((nx, ny))
        self.probe_bot = np.zeros((nx, ny))
        self.probe_top[out_x, y_top_idx] = 1.0
        self.probe_bot[out_x, y_bot_idx] = 1.0
        
    def run(self, eps_r):
        """
        Run FDFD simulation.
        Args:
             eps_r (np.array): Relative permittivity distribution (Nx, Ny).
        Returns:
             transmission (float): Total transmission to output ports.
             fields (np.array): Field distribution (Hz).
        """
        # Run FDFD (Hz polarization for TE)
        simulation = fdfd_hz(self.omega, self.dl, eps_r, [self.npml, self.npml])
        
        try:
             # fdfd_hz returns (Ex, Ey, Hz) tuple
             fields_tuple = simulation.solve(self.source_mask)
             Hz = fields_tuple[0] # The first component contains the Hz field in this version
        except Exception as e:
             # Fallback for stability
             print(f"Warning: Simulation failed: {e}")
             return 0.0, np.zeros_like(self.source_mask)
        
        # Calculate power/intensity at probes
        t_top = np.sum(np.abs(Hz * self.probe_top)**2)
        t_bot = np.sum(np.abs(Hz * self.probe_bot)**2)
        
        # Return individual transmissions for reward calculation
        return (t_top, t_bot), Hz

if __name__ == "__main__":
    from sdf_gen import MMIGeometry
    geo = MMIGeometry()
    sim = CevicheSim(geo)
    eps = geo.get_permittivity()
    
    T, field = sim.run(eps)
    print(f"Baseline Transmission: {T}")
    
    plt.imshow(np.real(field).T, origin='lower', cmap='RdBu')
    plt.savefig("sim_test.png")
