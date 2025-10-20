# -*- coding: utf-8 -*-
"""
Napari GUI for XVR Model-Based 2D/3D Registration.

This script provides a user-friendly interface within Napari to run the `xvr`
command-line tool for registering a 3D medical volume (e.g., CT) to a 2D X-ray,
using a pre-trained deep learning model for the initial pose estimate.

It features two main panels:
1. A launcher panel (left) to open other related GUI scripts.
2. A main registration panel (right) with all the parameters for the
   `xvr register model` command.
"""

# --- Standard Library Imports --- 
import os
import subprocess
import sys
from pathlib import Path

# --- Third-Party Imports ---
import napari
from magicgui import magicgui
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QGroupBox, QPushButton, QVBoxLayout, QWidget

# =============================================================================
# --- 1. SCRIPT CONFIGURATION (EDIT YOUR PATHS HERE) ---
# =============================================================================

# Define the base directory as the location of this script file.
# This makes all other paths portable.
BASE_DIR = Path(__file__).resolve().parent

# --- Default Paths for GUI Fields ---
# These are the initial paths that will appear in the GUI.
# They are set to a user's home directory for portability.
# ⬇️ PLEASE EDIT THESE to point to your default data locations. ⬇️
DEFAULT_VOLUME_PATH = Path.home() / "xvr_data" / "volume.nii.gz"
DEFAULT_CHECKPOINT_PATH = Path.home() / "xvr_models" / "model.pth"
DEFAULT_DICOM_PATH = Path.home() / "xvr_data" / "xray.dcm"
DEFAULT_OUTPUT_DIR = BASE_DIR / "registration_results"

# --- Paths to Other GUI Scripts ---
# Assumes other GUI scripts are in the same directory as this one.
GUI_FILE_PATHS = {
    "Train Model": BASE_DIR / "simple_xvr_gui.py",
    "Fine-tune Model": BASE_DIR / "xvr_finetune_gui.py",
    "Register Model": BASE_DIR / "xvr_register_withanimation.py",  # This script
    "Register Model : Dicom": BASE_DIR / "31-7-25registrationdicom.py",
    "Register Model : Fixed": BASE_DIR / "31-7-25registrationfixed.py",
    "View Results": BASE_DIR / "parameters_display_gui.py",
}

# =============================================================================
# --- 2. CORE FUNCTIONS ---
# =============================================================================

def run_xvr_register_cli(config: dict) -> bool:
    """
    Constructs and executes the `xvr register model` command as a subprocess.

    This function is optimized for performance and correct terminal display by
    launching the command inside a pseudo-terminal on macOS/Linux.

    Args:
        config: A dictionary containing all parameters from the GUI.

    Returns:
        True if the command executed successfully, False otherwise.
    """
    show_info("Starting XVR registration (see terminal for live output)...")
    print("\n--- XVR Registration Log ---")

    # Use the 'script' utility (on macOS/Linux) to force a pseudo-terminal.
    # This ensures the tqdm progress bar renders correctly as a single line.
    command = ["script", "-q", "/dev/null", "xvr", "register", "model"]

    # --- Assemble Required Arguments ---
    command.extend(["-v", config["volume_path"]])
    command.extend(["-c", config["checkpoint_path"]])
    command.extend(["-o", config["output_path"]])

    # --- Assemble Optional Arguments ---
    # This block adds flags to the command only if their values differ
    # from the CLI's default, keeping the command clean.
    if config["mask_path"]:
        command.extend(["-m", config["mask_path"]])
    if config["crop"] != 0:
        command.extend(["--crop", str(config["crop"])])
    if config["subtract_background"]:
        command.append("--subtract_background")
    if config["reducefn"] != "max":
        command.extend(["--reducefn", config["reducefn"]])
    if config["warp_path"]:
        command.extend(["--warp", config["warp_path"]])
    if config["invert"]:
        command.append("--invert")
    if config["labels"]:
        command.extend(["--labels", config["labels"]])
    if config["scales"] != "8":
        command.extend(["--scales", config["scales"]])
    if config["reverse_x_axis"]:
        command.append("--reverse_x_axis")
    if config["renderer"] != "trilinear":
        command.extend(["--renderer", config["renderer"]])
    if config["parameterization"] != "euler_angles":
        command.extend(["--parameterization", config["parameterization"]])
    if config["convention"] != "ZXY":
        command.extend(["--convention", config["convention"]])
    if config["lr_rot"] != 0.01:
        command.extend(["--lr_rot", str(config["lr_rot"])])
    if config["lr_xyz"] != 1.0:
        command.extend(["--lr_xyz", str(config["lr_xyz"])])
    if config["patience"] != 10:
        command.extend(["--patience", str(config["patience"])])
    if config["threshold"] != 0.0001:
        command.extend(["--threshold", str(config["threshold"])])
    if config["max_n_itrs"] != 500:
        command.extend(["--max_n_itrs", str(config["max_n_itrs"])])
    if config["max_n_plateaus"] != 3:
        command.extend(["--max_n_plateaus", str(config["max_n_plateaus"])])
    if config["init_only"]:
        command.append("--init_only")
    if config["save_images"]:
        command.append("--saveimg")
    if config["pattern"] != "*.dcm":
        command.extend(["--pattern", config["pattern"]])
    if config["verbose_level"] != 1:
        command.extend(["--verbose", str(config["verbose_level"])])

    # Add the final positional argument (the X-ray path)
    command.append(config["dicom_path"])

    print(f"Executing command: {' '.join(command)}")

    try:
        # Set environment variables for the subprocess
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Launch the subprocess. stdout is not piped, allowing it to print
        # directly to the terminal for maximum performance.
        process = subprocess.Popen(
            command,
            stderr=subprocess.PIPE, # Capture error messages separately
            text=True,
            env=env,
            encoding='utf-8'
        )

        # Wait for the process to complete and capture any stderr output
        _, stderr_output = process.communicate()
        return_code = process.returncode

        # --- Handle Process Completion ---
        if return_code == 0:
            show_info("XVR Registration completed successfully!")
            print("\nXVR Registration CLI completed successfully.")
            if stderr_output:
                show_info("XVR Registration finished with warnings (check console).")
                print("--- STDERR Output (Warnings/Errors) ---")
                print(stderr_output)
            return True
        else:
            error_message = f"XVR Registration failed with exit code {return_code}."
            show_info(error_message)
            print(f"\n{error_message}")
            if stderr_output:
                print("--- STDERR Output (Failure) ---")
                print(stderr_output)
            return False

    except FileNotFoundError:
        show_info("Error: `script` or `xvr` command not found. Ensure XVR is installed.")
        return False
    except Exception as e:
        show_info(f"An unexpected error occurred: {e}")
        return False
    finally:
        print("--- End XVR Registration Log ---")

def launch_selected_gui(gui_name: str) -> None:
    """Launches another GUI script in a new Python process."""
    file_path = GUI_FILE_PATHS.get(gui_name)
    if not file_path or not file_path.is_file():
        show_info(f"Error: GUI file not found for {gui_name}")
        return
    try:
        # Launch in a new process so it doesn't block the current Napari window
        subprocess.Popen([sys.executable, file_path])
    except Exception as e:
        show_info(f"Failed to launch {gui_name}: {e}")

# =============================================================================
# --- 3. NAPARI WIDGET DEFINITIONS ---
# =============================================================================

def create_launcher_widget() -> QWidget:
    """
    Creates a native Qt widget with buttons to launch other GUI scripts.
    
    Returns:
        A Qt GroupBox containing the launch buttons.
    """
    launcher_widget = QGroupBox("Launch Other GUIs")
    layout = QVBoxLayout()
    
    for gui_name in GUI_FILE_PATHS:
        button = QPushButton(f"Launch {gui_name}")
        button.clicked.connect(lambda checked=False, name=gui_name: launch_selected_gui(name))
        layout.addWidget(button)
        
    launcher_widget.setLayout(layout)
    return launcher_widget

@magicgui(
    call_button="Run XVR Registration",
    layout="vertical",

    # -- Widget Definitions --
    # Main Paths
    volume_path={"widget_type": "FileEdit", "mode": "r", "label": "NIfTI Volume (-v)"},
    checkpoint_path={"widget_type": "FileEdit", "mode": "r", "label": "Model Checkpoint (-c)"},
    output_path={"widget_type": "FileEdit", "mode": "d", "label": "Output Directory (-o)"},
    dicom_path={"widget_type": "FileEdit", "mode": "r", "label": "DICOM X-ray (positional)"},
    mask_path={"widget_type": "FileEdit", "mode": "r", "label": "Mask Labelmap (-m, optional)"},
    
    # Preprocessing
    crop={"label": "X-ray Crop (px)", "min": 0, "max": 1000},
    subtract_background={"label": "Subtract Background"},
    linearize={"label": "Linearize X-ray (exp to linear)"},
    reducefn={"label": "Multiframe Reduce Function", "choices": ["max", "mean", "sum"]},
    
    # Pose Correction
    warp_path={"widget_type": "FileEdit", "mode": "r", "label": "Warp Transform File (--warp, optional)"},
    invert={"label": "Invert Warp"},

    # Rendering Options
    labels={"label": "Labels to Render (comma-sep, optional)"},
    reverse_x_axis={"label": "Reverse X-axis (radiologic convention)"},
    renderer={"label": "Renderer", "choices": ["siddon", "trilinear"]},

    # Optimization Parameters
    scales={"label": "Scales (comma-sep)"},
    parameterization={"label": "SO(3) Parameterization", "choices": ["euler_angles", "quaternions", "axis_angle", "rotation_6d"]},
    convention={"label": "Euler Convention", "choices": ["ZXY", "XYZ", "ZYX", "YXZ", "YZX", "XZY", "RAS", "LPS"]},
    lr_rot={"label": "LR Rotation", "min": 1e-6, "max": 1.0, "step": 1e-4},
    lr_xyz={"label": "LR Translation", "min": 1e-3, "max": 100.0, "step": 0.01},
    patience={"label": "Patience (epochs)", "min": 1, "max": 100},
    threshold={"label": "Threshold (for LR reduction)", "min": 1e-6, "max": 1.0, "step": 1e-5},
    max_n_itrs={"label": "Max Iterations/Scale", "min": 1, "max": 2000},
    max_n_plateaus={"label": "Max Plateaus/Scale", "min": 1, "max": 10},
    
    # Output & Logging
    init_only={"label": "Initial Pose Only (no refinement)"},
    save_images={"label": "Save Output Images (--saveimg)"},
    pattern={"label": "X-ray Pattern (if input is dir)"},
    verbose_level={"label": "Verbose Level", "min": 0, "max": 3},
)
def xvr_register_widget(
    # --- Function Signature with Type Hints and Defaults ---
    # Main Paths
    volume_path: Path = DEFAULT_VOLUME_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    output_path: Path = DEFAULT_OUTPUT_DIR,
    dicom_path: Path = DEFAULT_DICOM_PATH,
    mask_path: Path = None,
    
    # Preprocessing
    crop: int = 0,
    subtract_background: bool = False,
    linearize: bool = True,
    reducefn: str = "max",
    
    # Pose Correction
    warp_path: Path = None,
    invert: bool = False,

    # Rendering Options
    labels: str = "",
    reverse_x_axis: bool = False,
    renderer: str = "trilinear",

    # Optimization Parameters
    scales: str = "8",
    parameterization: str = "euler_angles",
    convention: str = "ZXY",
    lr_rot: float = 0.01,
    lr_xyz: float = 1.0,
    patience: int = 10,
    threshold: float = 0.0001,
    max_n_itrs: int = 500,
    max_n_plateaus: int = 3,
    
    # Output & Logging
    init_only: bool = False,
    save_images: bool = True,
    pattern: str = "*.dcm",
    verbose_level: int = 1,
) -> None:
    """The main magicgui widget for XVR registration."""
    
    # --- Input Validation ---
    if not volume_path.is_file():
        show_info(f"Error: NIfTI Volume not found at {volume_path}")
        return
    if not dicom_path.exists():
        show_info(f"Error: DICOM X-ray input not found at {dicom_path}")
        return
    if not checkpoint_path.is_file():
        show_info(f"Error: Model Checkpoint not found at {checkpoint_path}")
        return
    
    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect all function parameters into a dictionary for the CLI function
    config = locals()
    
    # Convert Path objects to strings for the command line
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value) if value else None

    # Run the registration command
    run_xvr_register_cli(config)

# =============================================================================
# --- 4. SCRIPT ENTRYPOINT ---
# =============================================================================

if __name__ == "__main__":
    # This block runs when the script is executed directly via `python <script_name>.py`
    
    # 1. Create a Napari viewer instance
    print("Initializing Napari viewer...")
    viewer = napari.Viewer(title="XVR Registration GUI")
    
    # 2. Create the launcher widget from our helper function
    launcher_panel = create_launcher_widget()
    
    # 3. Add the launcher widget to the LEFT side of the viewer
    viewer.window.add_dock_widget(launcher_panel, area='left', name="Launch GUIs")
    
    # 4. Add the main registration magicgui widget to the RIGHT side of the viewer
    viewer.window.add_dock_widget(xvr_register_widget, area='right', name="XVR Registration")
    
    # 5. Start the Napari event loop to show the GUI
    print("Starting Napari application...")
    napari.run()
    print("Napari application closed.")
