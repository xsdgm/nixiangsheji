import numpy as np
import matplotlib.pyplot as plt

class MMIGeometry:
    def __init__(self, region_size=(10.0, 10.0), resolution=0.04):
        """
        Initialize the MMI Geometry generator.
        
        Args:
            region_size (tuple): (Lx, Ly) physical size in microns.
            resolution (float): grid step size in microns.
        """
        self.Lx, self.Ly = region_size
        self.dl = resolution
        self.Nx = int(self.Lx / self.dl)
        self.Ny = int(self.Ly / self.dl)
        
        # Create grid
        self.x = np.linspace(0, self.Lx, self.Nx)
        self.y = np.linspace(0, self.Ly, self.Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # MMI Design Parameters (Defaults)
        self.wg_width = 0.5
        self.mmi_width = 4.0
        self.mmi_length = 6.0 # Kept small for fast testing, real MMI is larger
        self.taper_width = 1.0
        self.taper_length = 1.0
        
        # Initialize Base SDF
        self.sdf = np.zeros((self.Nx, self.Ny))
        self.reset_geometry()

    def reset_geometry(self):
        """Resets the SDF to the initial MMI shape."""
        # Clean slate
        self.sdf = np.full((self.Nx, self.Ny), -1.0)
        
        # Center coordinates
        cy = self.Ly / 2.0
        cx = self.Lx / 2.0
        
        # 1. Input Waveguide (Left)
        # Defined from x=0 to start of MMI
        input_y_dist = np.abs(self.Y - cy) - (self.wg_width / 2.0)
        
        # 2. MMI Body (Center)
        mmi_start_x = (self.Lx - self.mmi_length) / 2.0
        mmi_end_x = mmi_start_x + self.mmi_length
        mmi_y_dist = np.abs(self.Y - cy) - (self.mmi_width / 2.0)
        
        # 3. Combine shapes (Union)
        # Simple box approximation for now
        # We want SD > 0 for material (Silicon), < 0 for cladding (Air/SiO2)
        # Using simple rectangle logic for initial shape
        
        # Input WG
        mask_input = (self.X < mmi_start_x) & (input_y_dist < 0)
        
        # MMI Body
        mask_mmi = (self.X >= mmi_start_x) & (self.X <= mmi_end_x) & (mmi_y_dist < 0)
        
        # Output WGs (Two arms)
        out_offset = 1.5 # micron separation
        mask_out_top = (self.X > mmi_end_x) & (np.abs(self.Y - (cy + out_offset)) < self.wg_width/2.0)
        mask_out_bot = (self.X > mmi_end_x) & (np.abs(self.Y - (cy - out_offset)) < self.wg_width/2.0)
        
        self.mask = mask_input | mask_mmi | mask_out_top | mask_out_bot
        
        # Convert binary mask to rough SDF (1 inside, -1 outside)
        self.sdf = np.where(self.mask, 1.0, -1.0)

    def update_sdf(self, action_map):
        """
        Update SDF based on actions.
        For simplicity, action_map is an additive change to the SDF in the MMI region.
        """
        # Assume action_map matches MMI region shape or is interpolated
        # For this prototype, we'll assume global small perturbations
        # Real implementation would map action vector to spline points or pixels
        self.sdf += action_map * 0.1 # Scale factor

    def get_density(self):
        """Returns binary density (0 or 1) based on SDF > 0."""
        return (self.sdf > 0).astype(float)
        
    def get_permittivity(self, eps_struct=12.0, eps_bg=1.0):
        """Returns relative permittivity distribution."""
        mask = self.get_density()
        return mask * (eps_struct - eps_bg) + eps_bg

if __name__ == "__main__":
    geo = MMIGeometry()
    plt.imshow(geo.get_density().T, origin='lower')
    plt.savefig("mmi_init.png")
    print("Geometry test complete.")
