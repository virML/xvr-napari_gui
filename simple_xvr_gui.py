# -*- coding: utf-8 -*-
"""
Napari GUI for XVR Model Training.

This script provides a user-friendly interface within Napari to run the `xvr`
command-line tool for training a 2D/3D registration model from scratch using
a directory of CT scans.
"""

# --- Standard Library Imports ---
import os
import sys
import time
from pathlib import Path

# --- Third-Party Imports ---
import napari
from magicgui import magicgui
from napari.utils.notifications import show_info

# =============================================================================
# --- 1. SCRIPT CONFIGURATION (MUST BE EDITED BY THE USER) ---
# =============================================================================

# --- Default Paths for GUI Fields ---
# These are the initial paths that will appear when the GUI is launched.
# ⬇️ PLEASE EDIT THESE to point to your default data locations. ⬇️
DEFAULT_INPATH = Path.home() / "xvr_data" / "ct_scans_for_training"
DEFAULT_OUTPATH = Path.home() / "xvr_models" / "trained_model_output"

# --- Default Pose Generation Ranges ---
# These define the random poses used to generate synthetic X-rays during training.
# Rotations are in degrees, and translations are in millimeters.
DEFAULT_R1_RANGE = (-10.0, 10.0)  # Primary angle (e.g., around Z-axis)
DEFAULT_R2_RANGE = (-5.0, 5.0)   # Secondary angle (e.g., around X-axis)
DEFAULT_R3_RANGE = (-10.0, 10.0) # Tertiary angle (e.g., around Y-axis)
DEFAULT_TX_RANGE = (-50.0, 50.0) # Translation along X-axis
DEFAULT_TY_RANGE = (-50.0, 50.0) # Translation along Y-axis
DEFAULT_TZ_RANGE = (-50.0, 50.0) # Translation along Z-axis

# --- Default DRR Intrinsic Parameters ---
DEFAULT_SDD = 1000.0  # Source-to-Detector Distance (mm)
DEFAULT_HEIGHT = 128  # DRR height (pixels)
DEFAULT_DELX = 0.2    # DRR pixel size (mm/pixel)

# =============================================================================
# --- 2. XVR LIBRARY SETUP ---
# =============================================================================

# Add the 'src' directory to Python's path to find the xvr library.
# This assumes the script is run from the root of the xvr repository.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from xvr.commands.train import train_model
    print("✅ XVR 'train_model' function imported successfully.")
except ImportError as e:
    show_info(f"Could not import XVR. Using a dummy training function. Error: {e}")
    print(f"⚠️ WARNING: Could not import 'train_model' from 'xvr'. Using a dummy function.")
    print(f"   Error details: {e}")

    # Define a dummy function so the GUI can still launch for demonstration.
    def train_model(config: dict, run_obj) -> None:
        """A placeholder function to simulate training if XVR is not found."""
        print("--- Running DUMMY training function ---")
        n_epochs = config.get("n_epochs", 3)
        for i in range(n_epochs):
            time.sleep(1)  # Simulate work
            print(f"  Simulating Epoch {i + 1}/{n_epochs}...")
        print("--- DUMMY training complete ---")

# =============================================================================
# --- 3. NAPARI GUI WIDGET DEFINITION ---
# =============================================================================

@magicgui(
    call_button="Start Training",
    layout="vertical",

    # -- Widget Definitions --
    # Grouped for readability. magicgui infers widget types from function hints
    # but we can provide explicit options for more control.

    # --- Main Paths ---
    inpath={"mode": "d", "label": "Input CTs Directory (-i)"},
    outpath={"mode": "d", "label": "Output Model Directory (-o)"},

    # --- Pose Generation Ranges (degrees and mm) ---
    r1={"label": "Rotation 1 Range (deg)", "widget_type": "FloatRangeSlider", "min": -180, "max": 180},
    r2={"label": "Rotation 2 Range (deg)", "widget_type": "FloatRangeSlider", "min": -180, "max": 180},
    r3={"label": "Rotation 3 Range (deg)", "widget_type": "FloatRangeSlider", "min": -180, "max": 180},
    tx={"label": "Translation X Range (mm)", "widget_type": "FloatRangeSlider", "min": -200, "max": 200},
    ty={"label": "Translation Y Range (mm)", "widget_type": "FloatRangeSlider", "min": -200, "max": 200},
    tz={"label": "Translation Z Range (mm)", "widget_type": "FloatRangeSlider", "min": -200, "max": 200},

    # --- DRR Intrinsic Parameters ---
    sdd={"label": "SDD (mm)", "min": 500.0, "max": 2000.0, "step": 10.0},
    height={"label": "DRR Height (px)", "min": 64, "max": 512, "step": 16},
    delx={"label": "DRR Pixel Size (mm/px)", "min": 0.05, "max": 1.0, "step": 0.01},

    # --- Model & Rendering Parameters ---
    renderer={"label": "Renderer", "choices": ["siddon", "trilinear"]},
    orientation={"label": "CT Orientation", "choices": ["AP", "PA"]},
    reverse_x_axis={"label": "Reverse X-axis (Radiologic Convention)"},
    parameterization={"label": "SO(3) Parameterization", "choices": ["euler_angles", "quaternions"]},
    convention={"label": "Euler Convention", "choices": ["ZXY", "XYZ", "ZYX", "YXZ", "YZX", "XZY", "RAS", "LPS"]},
    model_name={"label": "Model Architecture", "choices": ["resnet18", "resnet34", "resnet50"]},
    pretrained={"label": "Use ImageNet Pretrained Weights"},
    norm_layer={"label": "Normalization Layer", "choices": ["batchnorm", "groupnorm", "instancenorm"]},

    # --- Training Hyperparameters ---
    lr={"label": "Learning Rate", "min": 1e-5, "max": 1e-2, "step": 1e-4, "tooltip": "Maximum learning rate for the scheduler."},
    weight_geo={"label": "Geodesic Loss Weight", "min": 0.0, "max": 1.0, "step": 0.01},
    batch_size={"label": "Batch Size", "min": 1, "max": 256, "step": 1},
    n_epochs={"label": "Number of Epochs", "min": 1, "max": 2000, "step": 1},
    n_batches_per_epoch={"label": "Batches per Epoch", "min": 1, "max": 200, "step": 1},
)
def xvr_train_widget(
    # --- Function Signature with Type Hints and Defaults ---
    # These values populate the GUI on startup.
    
    # Paths
    inpath: Path = DEFAULT_INPATH,
    outpath: Path = DEFAULT_OUTPATH,
    
    # Pose Ranges
    r1: tuple[float, float] = DEFAULT_R1_RANGE,
    r2: tuple[float, float] = DEFAULT_R2_RANGE,
    r3: tuple[float, float] = DEFAULT_R3_RANGE,
    tx: tuple[float, float] = DEFAULT_TX_RANGE,
    ty: tuple[float, float] = DEFAULT_TY_RANGE,
    tz: tuple[float, float] = DEFAULT_TZ_RANGE,
    
    # DRR Intrinsics
    sdd: float = DEFAULT_SDD,
    height: int = DEFAULT_HEIGHT,
    delx: float = DEFAULT_DELX,
    
    # Model & Rendering Parameters (defaults from xvr/commands/train.py)
    renderer: str = "trilinear",
    orientation: str = "PA",
    reverse_x_axis: bool = False,
    parameterization: str = "euler_angles",
    convention: str = "ZXY",
    model_name: str = "resnet18",
    pretrained: bool = False,
    norm_layer: str = "groupnorm",
    
    # Training Hyperparameters (defaults from xvr/commands/train.py)
    lr: float = 5e-3,
    weight_geo: float = 1e-2,
    batch_size: int = 116,
    n_epochs: int = 1000,
    n_batches_per_epoch: int = 100,
) -> None:
    """
    A magicgui widget that collects training parameters and launches the
    XVR training process.
    """
    show_info("Starting XVR training (this may block the GUI)...")
    print("\n--- XVR Training Log ---")

    # --- Input Validation ---
    if not inpath.is_dir():
        show_info(f"Error: Input directory not found at {inpath}")
        return
        
    # Ensure output directory exists
    outpath.mkdir(parents=True, exist_ok=True)

    # --- Configuration Dictionary ---
    # Assemble the config dictionary required by the `train_model` function.
    config = {
        "inpath": str(inpath),
        "outpath": str(outpath),
        "alphamin": r1[0], "alphamax": r1[1],
        "betamin": r2[0], "betamax": r2[1],
        "gammamin": r3[0], "gammamax": r3[1],
        "txmin": tx[0], "txmax": tx[1],
        "tymin": ty[0], "tymax": ty[1],
        "tzmin": tz[0], "tzmax": tz[1],
        "sdd": sdd,
        "height": height,
        "delx": delx,
        "renderer": renderer,
        "orientation": orientation,
        "reverse_x_axis": reverse_x_axis,
        "parameterization": parameterization,
        "convention": convention,
        "model_name": model_name,
        "pretrained": pretrained,
        "norm_layer": norm_layer,
        "lr": lr,
        "weight_geo": weight_geo,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_batches_per_epoch": n_batches_per_epoch,
        "project": "xvr-napari-gui-train", # Hardcoded project for simplicity
        "name": None,
    }

    # --- Run Training ---
    try:
        # Use wandb for logging if available, otherwise train without it.
        # Running in "offline" mode is best practice for GUIs to prevent
        # the app from hanging while waiting for network requests.
        wandb_run = None
        try:
            import wandb
            wandb_run = wandb.init(
                project=config["project"],
                name=config["name"],
                mode="offline",
                reinit=True,
                dir=str(outpath)
            )
            print("WandB initialized in offline mode. Logs will be in the output directory.")
        except ImportError:
            print("WandB not found. Training will proceed without logging.")

        # Call the main training function
        train_model(config, wandb_run)

        if wandb_run:
            wandb_run.finish()

        show_info("✅ XVR training completed! Check output directory for model weights.")
    except Exception as e:
        show_info(f"❌ Error during training: {e}")
        print(f"\n--- ERROR DURING TRAINING ---\n{e}\n----------------------------")
        
    print("--- End XVR Training Log ---")


# =============================================================================
# --- 4. SCRIPT ENTRYPOINT ---
# =============================================================================

if __name__ == "__main__":
    """
    This block is executed when the script is run directly
    
    """
    # 1. Create a Napari viewer instance
    print("Initializing Napari viewer...")
    viewer = napari.Viewer(title="XVR Training GUI")
    
    # 2. Add the magicgui widget to the viewer's right dock
    viewer.window.add_dock_widget(xvr_train_widget, area='right', name="XVR Model Training")
    
    # 3. Start the Napari application event loop
    print("Starting Napari application...")
    napari.run()
    print("Napari application closed.")