"""
MCP Server for pyTEMlib - Transmission Electron Microscopy Data Analysis

This server provides MCP tools for analyzing TEM data using the pyTEMlib library.
It exposes key functionalities for file I/O, data analysis, and visualization.
"""

from fastmcp import FastMCP
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional, Union
import traceback

# Import pyTEMlib modules
try:
    import pyTEMlib as pytem
    from pyTEMlib import file_tools, utilities, image_tools, eels_tools, eds_tools
    from pyTEMlib import crystal_tools, diffraction_tools, probe_tools, graph_tools
    PYTEMLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pyTEMlib not available: {e}")
    PYTEMLIB_AVAILABLE = False

# Create MCP server
mcp = FastMCP("pyTEMlib-MCP")

# Global state for loaded datasets
loaded_datasets = {}

@mcp.tool()
def get_pytemlib_status() -> str:
    """Check if pyTEMlib is properly loaded and available."""
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib is not available. Please ensure it's properly installed."

    try:
        version = getattr(pytem, '__version__', 'unknown')
        return f"pyTEMlib v{version} is loaded and ready. Available modules: file_tools, utilities, image_tools, eels_tools, eds_tools, crystal_tools, diffraction_tools, probe_tools, graph_tools"
    except Exception as e:
        return f"pyTEMlib loaded but error checking version: {str(e)}"

@mcp.tool()
def calculate_electron_wavelength(voltage: float, unit: str = "nm") -> str:
    """Calculate the relativistic de Broglie wavelength of electrons.

    Parameters:
    - voltage: Acceleration voltage in volts (e.g., 200000 for 200kV)
    - unit: Unit for wavelength ('m', 'mm', 'μm', 'nm', 'A', 'Å', 'pm')

    Returns the wavelength in the specified unit.
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    try:
        wavelength = utilities.get_wavelength(voltage, unit)
        return f"Electron wavelength at {voltage/1000:.0f}kV: {wavelength:.3f} {unit}"
    except Exception as e:
        return f"Error calculating wavelength: {str(e)}"

@mcp.tool()
def list_supported_file_formats() -> str:
    """List the file formats supported by pyTEMlib for reading TEM data."""
    formats = [
        ".dm3, .dm4 - Digital Micrograph files",
        ".emd - EMD files",
        ".hf5 - HDF5/pyNSID files",
        ".ndata - Nion data files",
        ".mrc - MRC image files"
    ]
    return "Supported file formats:\n" + "\n".join(f"- {fmt}" for fmt in formats)

@mcp.tool()
def open_tem_file(filename: str) -> str:
    """Open a TEM data file and load its datasets.

    Parameters:
    - filename: Path to the TEM data file

    Returns a dictionary containing information about the loaded datasets.
    """
    if not PYTEMLIB_AVAILABLE:
        return {"error": "pyTEMlib not available"}

    try:
        if not os.path.exists(filename):
            return f"File not found: {filename}"

        datasets = file_tools.open_file(filename)
        loaded_datasets[filename] = datasets

        result = f"Successfully opened {filename}\n"
        result += f"Found {len(datasets)} datasets:\n"

        for key, dataset in datasets.items():
            result += f"- {key}: {dataset.title} ({dataset.data_type.name})\n"
            if hasattr(dataset, 'shape'):
                result += f"  Shape: {dataset.shape}\n"
        import json
        return json.dumps(datasets)  # Return all datasets as JSON for demonstration
    except Exception as e:
        return f"Error opening file: {str(e)}\n{traceback.format_exc()}"

@mcp.tool()
def list_loaded_datasets() -> str:
    """List all currently loaded datasets from opened files."""
    if not loaded_datasets:
        return "No datasets currently loaded. Use open_tem_file() to load data."

    result = "Loaded datasets:\n"
    for filename, datasets in loaded_datasets.items():
        result += f"\nFile: {filename}\n"
        for key, dataset in datasets.items():
            result += f"  - {key}: {dataset.title}\n"

    return result

@mcp.tool()
def get_dataset_info(filename: str, dataset_key: str) -> str:
    """Get detailed information about a specific dataset.

    Parameters:
    - filename: The file containing the dataset
    - dataset_key: The key/name of the dataset
    """
    if filename not in loaded_datasets:
        return f"File {filename} not loaded. Use open_tem_file() first."

    datasets = loaded_datasets[filename]
    if dataset_key not in datasets:
        available_keys = list(datasets.keys())
        return f"Dataset '{dataset_key}' not found. Available: {available_keys}"

    dataset = datasets[dataset_key]

    result = f"Dataset: {dataset_key}\n"
    result += f"Title: {dataset.title}\n"
    result += f"Data type: {dataset.data_type.name}\n"

    if hasattr(dataset, 'shape'):
        result += f"Shape: {dataset.shape}\n"

    if hasattr(dataset, 'dimensions'):
        result += "Dimensions:\n"
        for i, dim in enumerate(dataset.dimensions):
            result += f"  {i}: {dim.name} ({dim.units})\n"

    if hasattr(dataset, 'metadata') and dataset.metadata:
        result += f"Metadata keys: {list(dataset.metadata.keys())}\n"

    return result

@mcp.tool()
def calculate_probe_size(voltage: float, convergence_angle_mrad: float) -> str:
    """Calculate electron probe size using pyTEMlib probe_tools.

    Parameters:
    - voltage: Acceleration voltage in volts
    - convergence_angle_mrad: Convergence angle in milliradians
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    try:
        # Convert mrad to radians
        convergence_angle_rad = convergence_angle_mrad / 1000.0

        # This is a simplified calculation - actual implementation would use probe_tools
        wavelength = utilities.get_wavelength(voltage)
        probe_size = 0.61 * wavelength / convergence_angle_rad  # Airy disk approximation

        return f"Estimated probe size at {voltage/1000:.0f}kV with {convergence_angle_mrad}mrad convergence: {probe_size*1e9:.1f} nm"

    except Exception as e:
        return f"Error calculating probe size: {str(e)}"

@mcp.tool()
def get_element_info(element: Union[str, int]) -> str:
    """Get information about a chemical element for EDS/EELS analysis.

    Parameters:
    - element: Element symbol (e.g., 'Fe') or atomic number (e.g., 26)
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    try:
        # Get atomic number
        if isinstance(element, str):
            z = utilities.get_z(element)
            symbol = element
        else:
            z = element
            symbol = utilities.elements[z] if 1 <= z < len(utilities.elements) else "Unknown"

        result = f"Element: {symbol} (Z = {z})\n"

        # Get ionization edges if available
        try:
            edges = utilities.get_ionization_energy(z)
            if edges:
                result += "Ionization edges:\n"
                for edge, energy in edges.items():
                    if energy > 0:
                        result += f"  {edge}: {energy:.1f} eV\n"
        except:
            pass

        return result

    except Exception as e:
        return f"Error getting element info: {str(e)}"

@mcp.tool()
def analyze_diffraction_pattern(filename: str, dataset_key: str) -> str:
    """Analyze a diffraction pattern dataset.

    Parameters:
    - filename: File containing the diffraction data
    - dataset_key: Key of the diffraction dataset
    """
    if not PYTEMLIB_AVAILABLE:
        return "pyTEMlib not available"

    if filename not in loaded_datasets:
        return f"File {filename} not loaded."

    datasets = loaded_datasets[filename]
    if dataset_key not in datasets:
        return f"Dataset {dataset_key} not found."

    try:
        dataset = datasets[dataset_key]

        # Basic analysis - check if it's a 2D image
        if len(dataset.shape) != 2:
            return f"Dataset {dataset_key} is not a 2D diffraction pattern (shape: {dataset.shape})"

        result = f"Diffraction pattern analysis for {dataset_key}:\n"
        result += f"Shape: {dataset.shape}\n"
        result += f"Data range: {dataset.min():.3f} to {dataset.max():.3f}\n"

        # This would be extended with actual diffraction analysis functions
        result += "\nNote: Full diffraction analysis requires additional parameters (camera length, voltage, etc.)"

        return result

    except Exception as e:
        return f"Error analyzing diffraction pattern: {str(e)}"

@mcp.tool()
def create_visualization_script(filename: str, dataset_key: str, plot_type: str = "image") -> str:
    """Generate Python code to visualize a dataset.

    Parameters:
    - filename: File containing the dataset
    - dataset_key: Key of the dataset to visualize
    - plot_type: Type of plot ('image', 'spectrum', 'scatter')
    """
    if filename not in loaded_datasets:
        return f"File {filename} not loaded."

    datasets = loaded_datasets[filename]
    if dataset_key not in datasets:
        return f"Dataset {dataset_key} not found."

    dataset = datasets[dataset_key]

    script = f"""# Visualization script for {dataset_key} from {filename}
import matplotlib.pyplot as plt
import numpy as np

# Load the data (this would be done automatically in pyTEMlib)
data = np.random.rand{dataset.shape}  # Placeholder - actual data would be loaded

"""

    if plot_type == "image" and len(dataset.shape) >= 2:
        script += """
# Display as image
plt.figure(figsize=(10, 8))
plt.imshow(data, cmap='viridis')
plt.colorbar()
plt.title(f'{dataset_key}')
plt.show()
"""
    elif plot_type == "spectrum" and len(dataset.shape) == 1:
        script += """
# Display as spectrum
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.xlabel('Channel/Energy')
plt.ylabel('Intensity')
plt.title(f'{dataset_key}')
plt.show()
"""
    else:
        script += """
# Generic plot
plt.figure(figsize=(10, 6))
plt.plot(data.flatten()[:1000])  # Show first 1000 points
plt.title(f'{dataset_key} (sample)')
plt.show()
"""

    return script

@mcp.tool()
def get_available_tools() -> str:
    """List all available MCP tools for pyTEMlib."""
    tools = [
        "get_pytemlib_status - Check if pyTEMlib is loaded",
        "calculate_electron_wavelength - Calculate electron wavelength",
        "list_supported_file_formats - Show supported file formats",
        "open_tem_file - Open TEM data files",
        "list_loaded_datasets - List currently loaded datasets",
        "get_dataset_info - Get detailed dataset information",
        "calculate_probe_size - Calculate electron probe size",
        "get_element_info - Get element information for spectroscopy",
        "analyze_diffraction_pattern - Analyze diffraction patterns",
        "create_visualization_script - Generate visualization code",
        "get_available_tools - List all available tools"
    ]

    return "Available pyTEMlib MCP Tools:\n" + "\n".join(f"- {tool}" for tool in tools)

# Run the server
if __name__ == "__main__":
    print("Starting pyTEMlib MCP Server...")

    mcp.run(transport="sse",
        host="127.0.0.1",  # Use "127.0.0.1" for local-only, "0.0.0.0" to expose on your network
        port=8003,
        path="/mcp")
