import numpy as np
import ceviche
from ceviche import fdfd_hz

def test_minimal():
    print("Initializing Minimal Test...")
    # Parameters
    omega = 2 * np.pi * 200e12 # 200 THz
    dl = 40e-9 # 40 nm
    Nx, Ny = 100, 100
    npml = 10
    
    # Epsilon (Free space)
    eps_r = np.ones((Nx, Ny))
    
    # Source
    source = np.zeros((Nx, Ny))
    source[Nx//2, Ny//2] = 1.0
    
    # Simulation
    print("Creating FDFD...")
    try:
        F = fdfd_hz(omega, dl, eps_r, [npml, npml])
        print("Solving...")
        Ex, Ey, Hz = F.solve(source)
        print("Solved.")
        print(f"Hz max: {np.max(np.abs(Hz))}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Test Failed: {e}")

if __name__ == "__main__":
    test_minimal()
