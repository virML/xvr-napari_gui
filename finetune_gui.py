"""
Napari GUI for XVR Model Finetuning.

This script provides a user-friendly interface within Napari to finetune a
pre-trained `xvr` registration model on a new, specific CT scan.
"""

# --- Standard Library Imports ---
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
DEFAULT_INPATH = Path.home() / "xvr_data" / "patient_ct.nii.gz"
DEFAULT_OUTPATH = Path.home() / "xvr_models" / "finetuned_model_output"
DEFAULT_CKPTPATH = Path.home() / "xvr_models" / "pretrained_model.pth"

# =============================================================================
# --- 2. XVR LIBRARY SETUP ---
# =============================================================================

# Add the 'src' directory to Python's path to find the xvr library.
# This assumes the script is run from the root of the xvr repository.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from click.testing import CliRunner
    from xvr.cli import cli
    print("✅ XVR CLI and CliRunner imported successfully.")

    def run_xvr_finetune_cli(config: dict, run_obj) -> None:
        """Runs the 'xvr finetune' command programmatically using CliRunner."""
        runner = CliRunner()
        args = [
            "finetune",
            "-i", config["inpath"],
            "-o", config["outpath"],
            "-c", config["ckptpath"],
            "--lr", str(config["lr"]),
            "--batch_size", str(config["batch_size"]),
            "--n_epochs", str(config["n_epochs"]),
            "--n_batches_per_epoch", str(config["n_batches_per_epoch"]),
            "--rescale", str(config["rescale"]),
        ]
        # Add wandb arguments if a run object is provided
        if run_obj and config.get("project"):
            args.extend(["--project", config["project"]])
            if config.get("name"):
                args.extend(["--name", config["name"]])

        print(f"Calling XVR finetune CLI with args: {args}")
        result = runner.invoke(cli, args)

        if result.exit_code != 0:
            error_message = f"XVR Finetuning failed: {result.exception or result.output}"
            show_info(error_message)
            print(f"--- XVR FINETUNING ERROR ---\n{result.output}\n--------------------------")
            raise result.exception if result.exception else Exception(result.output)
        else:
            print("--- XVR FINETUNING OUTPUT ---\n", result.output)

except ImportError as e:
    show_info(f"Could not import XVR/Click. Using a dummy function. Error: {e}")
    print(f"⚠️ WARNING: Could not import from 'xvr' or 'click'. Using a dummy function.")

    def run_xvr_finetune_cli(config: dict, run_obj) -> None:
        """A placeholder function to simulate finetuning if XVR is not found."""
        print("--- Running DUMMY finetuning function ---")
        n_epochs = config.get("n_epochs", 3)
        for i in range(n_epochs):
            time.sleep(0.5)
            print(f"  Simulating Finetuning Epoch {i + 1}/{n_epochs}...")
        print("--- DUMMY finetuning complete ---")

# =============================================================================
# --- 3. NAPARI GUI WIDGET DEFINITION ---
# =============================================================================

@magicgui(
    call_button="Start Finetuning",
    layout="vertical",
    inpath={"widget_type": "FileEdit", "mode": "r", "label": "Input CT Volume (-i)"},
    outpath={"widget_type": "FileEdit", "mode": "d", "label": "Output Model Directory (-o)"},
    ckptpath={"widget_type": "FileEdit", "mode": "r", "label": "Pretrained Checkpoint (-c)"},
    lr={"label": "Learning Rate", "min": 1e-6, "max": 1e-1, "step": 1e-5},
    batch_size={"label": "Batch Size", "min": 1, "max": 512},
    n_epochs={"label": "Number of Epochs", "min": 1, "max": 500},
    n_batches_per_epoch={"label": "Batches per Epoch", "min": 1, "max": 100},
    rescale={"label": "Rescale Virtual Detector", "min": 0.1, "max": 5.0, "step": 0.1},
)
def xvr_finetune_widget(
    inpath: Path = DEFAULT_INPATH,
    outpath: Path = DEFAULT_OUTPATH,
    ckptpath: Path = DEFAULT_CKPTPATH,
    lr: float = 0.005,
    batch_size: int = 116,
    n_epochs: int = 10,
    n_batches_per_epoch: int = 25,
    rescale: float = 1.0,
) -> None:
    """A magicgui widget to collect parameters and launch the XVR finetuning process."""
    show_info("Starting XVR finetuning (this may block the GUI)...")
    print("\n--- XVR Finetuning Log ---")

    # --- Input Validation ---
    if not inpath.is_file():
        show_info(f"Error: Input CT volume not found at {inpath}")
        return
    if not ckptpath.is_file():
        show_info(f"Error: Pretrained checkpoint not found at {ckptpath}")
        return

    outpath.mkdir(parents=True, exist_ok=True)

    # --- Configuration Dictionary ---
    # Assemble the config dict required by the `run_xvr_finetune_cli` function.
    config = {
        "inpath": str(inpath),
        "outpath": str(outpath),
        "ckptpath": str(ckptpath),
        "lr": lr,
        "batch_size": batch_size,
        "n_epochs": n_epochs,
        "n_batches_per_epoch": n_batches_per_epoch,
        "rescale": rescale,
        "project": "xvr-napari-gui-finetune", # Hardcoded project for simplicity
        "name": f"finetune_{inpath.stem}", # Example run name
    }

    # --- Run Finetuning ---
    try:
        # Use wandb for logging if available, otherwise train without it.
        # Running in "offline" mode is best for GUIs to prevent hanging.
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
            print("WandB not found. Finetuning will proceed without logging.")

        # Call the main finetuning function
        run_xvr_finetune_cli(config, wandb_run)

        if wandb_run:
            wandb_run.finish()

        show_info("✅ XVR finetuning completed! Check output directory for model weights.")
    except Exception as e:
        show_info(f"❌ Error during finetuning: {e}")
        print(f"\n--- ERROR DURING FINETUNING ---\n{e}\n----------------------------")
        
    print("--- End XVR Finetuning Log ---")

# =============================================================================
# --- 4. SCRIPT ENTRYPOINT ---
# =============================================================================

if __name__ == "__main__":
    """
    This block is executed when the script is run directly.
    """
    print("Initializing Napari viewer...")
    viewer = napari.Viewer(title="XVR Finetuning GUI")
    
    # Add the widget to the viewer's right dock
    viewer.window.add_dock_widget(xvr_finetune_widget, area='right', name="XVR Finetuning")
    
    print("Starting Napari application...")
    napari.run()
    print("Napari application closed.")
