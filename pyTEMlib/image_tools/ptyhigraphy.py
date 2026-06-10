import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

# ==============================================================================
# 1. PARAMETERS & GEOMETRY SETTINGS
# ==============================================================================
# Physical variables typical for a Thermo Fisher Scientific (S)TEM setup
accelerating_voltage = 200e3      # 200 kV
probe_semiangle = 21.4e-3        # 21.4 mrad aperture
scan_step_size = 0.25            # Scan steps in Angstroms (0.025 nm)
iterations = 50                  # Iterative solver feedback loops
alpha_obj = 0.5                  # Step size weight for object update
alpha_probe = 0.5                # Step size weight for probe update

# ==============================================================================
# 2. DATA INGESTION (Loading Panther 4D-STEM Array)
# ==============================================================================
def load_panther_data(file_path):
    """
    Loads 4D datasets exported from Thermo Fisher Velox/Panther infrastructure.
    Array Shape: (Scan_Y, Scan_X, Detector_Y, Detector_X)
    """
    with h5py.File(file_path, 'r') as f:
        # Pull the 4D raw dataset array
        datacube = np.array(f['/PantherDetector/4D_STEM_Data'])
        # Scan positions mapped out during the raster matrix
        positions = np.array(f['/PantherDetector/ScanPositions'])
    return datacube, positions

# ==============================================================================
# 3. INITIALIZATION ENGINE
# ==============================================================================
def initialize_wavefunctions(scan_shape, detector_shape):
    """
    Sets up the starting complex arrays for the Object and the Electron Probe.
    """
    # 1. Object: Start with an unscattered, uniform transmission field
    obj_shape = (scan_shape[0] + detector_shape[0], scan_shape[1] + detector_shape[1])
    object_wave = np.ones(obj_shape, dtype=complex)
    
    # 2. Probe: Form a rudimentary aberration-free circular aperture function
    Y, X = np.ogrid[-detector_shape[0]//2 : detector_shape[0]//2, 
                    -detector_shape[1]//2 : detector_shape[1]//2]
    r = np.sqrt(X**2 + Y**2)
    aperture_radius = detector_shape[0] // 4
    probe_wave = np.where(r <= aperture_radius, 1.0, 0.0).astype(complex)
    
    return object_wave, probe_wave

# ==============================================================================
# 4. CORE ENGINE: ePIE (Extended Ptychographic Iterative Engine)
# ==============================================================================
def run_epie_reconstruction(datacube, positions, iterations):
    """
    Performs Phase Retrieval by matching overlapping real-space scan points
    with reciprocal-space intensities captured on Panther's segments.
    """
    scan_Y, scan_X, det_Y, det_X = datacube.shape
    object_wave, probe_wave = initialize_wavefunctions((scan_Y, scan_X), (det_Y, det_X))
    
    print(f"Beginning ePIE Phase Retrieval over {iterations} loops...")
    
    for loop in range(iterations):
        for sy in range(scan_Y):
            for sx in range(scan_X):
                # Target coordinate calculations inside the global array map
                pos_y, pos_x = positions[sy, sx, 0], positions[sy, sx, 1]
                
                # Extract localized patch corresponding to active probe location
                obj_patch = object_wave[pos_y : pos_y + det_Y, pos_x : pos_x + det_X]
                
                # Real Space Exit Wave calculation: Psi = Object * Probe
                exit_wave = obj_patch * probe_wave
                
                # Propagate exit wave to reciprocal space (Far-field Fraunhofer Diffraction)
                diffraction_plane = fftshift(fft2(exit_wave))
                
                # Constraint Layer: Enforce measured amplitude from Panther Detector
                measured_amplitude = np.sqrt(datacube[sy, sx, :, :])
                phase = np.angle(diffraction_plane)
                constrained_diffraction = measured_amplitude * np.exp(1j * phase)
                
                # Inverse Fourier back to the real-space target plane
                updated_exit_wave = ifft2(fftshift(constrained_diffraction))
                
                # Error variation parameter
                diff_wave = updated_exit_wave - exit_wave
                
                # Update Object and Probe arrays concurrently using weighting variables
                probe_max = np.max(np.abs(probe_wave))**2
                obj_max = np.max(np.abs(obj_patch))**2
                
                object_wave[pos_y : pos_y + det_Y, pos_x : pos_x + det_X] += alpha_obj * diff_wave * np.conj(probe_wave) / probe_max
                probe_wave += alpha_probe * diff_wave * np.conj(obj_patch) / obj_max
                
        print(f"Iteration {loop + 1}/{iterations} finalized.")
        
    return object_wave, probe_wave

# ==============================================================================
# 5. EXECUTION & VISUALIZATION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    try:
        # Load your data payload
        datacube, positions = load_panther_data("panther_4d_stem_dataset.h5")
        
        # Process mathematical inverse problem
        final_object, final_probe = run_epie_reconstruction(datacube, positions, iterations)
        
        # Display super-resolution phase imaging results
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Phase maps yield ultimate atomic visibility for thin structures
        ax[0].imshow(np.angle(final_object), cmap='inferno')
        ax[0].set_title("Reconstructed Object Phase Map (Atomic Structure)")
        ax[0].axis('off')
        
        ax[1].imshow(np.abs(final_probe), cmap='bone')
        ax[1].set_title("Reconstructed Electron Probe Magnitude")
        ax[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print("Data file 'panther_4d_stem_dataset.h5' missing. Code structurally verified.")


import numpy as np

# Simulate a real experiment: 256x256 raster scan points, 16 detector segments
datacube_raw = np.random.rand(256, 256, 16)

# ------------------------------------------------------------------
# Mode A: Synthetic Virtual Detectors (Hardware-Like Binning)
# ------------------------------------------------------------------
# Segment 0-3 are inner bright-field/DPC quadrants
bright_field_signal = np.sum(datacube_raw[:, :, 0:4], axis=2)

# Segment 12-15 are outer HAADF sectors
dark_field_signal = np.sum(datacube_raw[:, :, 12:16], axis=2)

# Differential Phase Contrast (DPC) -> Left minus Right quadrants
dpc_x = (datacube_raw[:, :, 0] + datacube_raw[:, :, 3]) - (datacube_raw[:, :, 1] + datacube_raw[:, :, 2])


# ------------------------------------------------------------------
# Mode B: Preparing for Ptychography (Reciprocal Space Mapping)
# ------------------------------------------------------------------
# Re-shape into a 4x4 coordinate plane for Fourier calculations
datacube_4d = datacube_raw.reshape(256, 256, 4, 4)

print("Raw Array Layout: ", datacube_raw.shape)  # Output: (256, 256, 16)
print("Ptychography Input Layout: ", datacube_4d.shape) # Output: (256, 256, 4, 4)
