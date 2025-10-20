import napari
from magicgui import magicgui
from napari.utils.notifications import show_info
from pathlib import Path
import sys
import subprocess
import os

# --- XVR CLI Execution Function for 'fixed' mode ---
def run_xvr_register_fixed_cli(config):
    """
    Executes the 'xvr register fixed' CLI command as a subprocess.
    """
    show_info("Starting XVR Fixed-Pose Registration (check console for live output)...")
    print("\n--- XVR Fixed-Pose Registration Log ---")

    command = ["xvr", "register", "fixed"]

    # Required arguments 
    command.extend(["-v", config["volume_path"]])
    command.extend(["-o", config["output_path"]])
    command.extend(["--orientation", config["orientation"]])

    command.append("--rot")
    command.append(f'{config["rx"]},{config["ry"]},{config["rz"]}')
    command.append("--xyz")
    command.append(f'{config["tx"]},{config["ty"]},{config["tz"]}')

    if config["mask_path"]:
        command.extend(["-m", config["mask_path"]])
    if config["crop"] != 0:
        command.extend(["--crop", str(config["crop"])])
    if config["subtract_background"]:
        command.append("--subtract_background")
    if config["linearize"]:
        command.append("--linearize")
    if config["reducefn"] != "max":
        command.extend(["--reducefn", config["reducefn"]])
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
    if config["verbose_level"] != 1:
        command.extend(["--verbose", str(config["verbose_level"])])

    command.append(config["dicom_path"])

    print(f"Executing command: {' '.join(command)}")

    try:
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1
        )

        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                print(line.strip())
                sys.stdout.flush()

        stderr_output = process.stderr.read()
        return_code = process.wait()

        if return_code == 0:
            show_info("XVR Registration completed successfully!")
            if stderr_output:
                print("--- STDERR Output ---")
                print(stderr_output)
        else:
            error_message = f"XVR Registration failed with exit code {return_code}."
            show_info(error_message)
            print(error_message)
            if stderr_output:
                print("--- STDERR Output (Failure) ---")
                print(stderr_output)

        return return_code == 0

    except FileNotFoundError:
        show_info("Error: `xvr` command not found. Make sure your environment is activated.")
        return False
    except Exception as e:
        show_info(f"An unexpected error occurred: {e}")
        return False
    finally:
        print("--- End XVR Registration Log ---")


# --- The Napari GUI Widget for Fixed-Pose Registration ---
@magicgui(
    call_button="Run Fixed-Pose Registration",
    layout="vertical",

    # Main Inputs
    volume_path={"widget_type": "FileEdit", "mode": "r", "label": "NIfTI Volume (-v)"},
    output_path={"widget_type": "FileEdit", "mode": "d", "label": "Output Directory (-o)"},
    dicom_path={"widget_type": "FileEdit", "mode": "r", "label": "DICOM X-ray (positional)"},
    mask_path={"widget_type": "FileEdit", "mode": "r", "label": "Mask Labelmap (-m, optional)"},

    # Pose Inputs
    orientation={"label": "X-ray Orientation", "choices": ["AP", "PA"]},
    rx={"label": "Initial Rotation: rx (deg)", "tooltip": "Euler angle for the first axis.", "min": -360.0, "max": 360.0},
    ry={"label": "Initial Rotation: ry (deg)", "tooltip": "Euler angle for the second axis.", "min": -360.0, "max": 360.0},
    rz={"label": "Initial Rotation: rz (deg)", "tooltip": "Euler angle for the third axis.", "min": -360.0, "max": 360.0},

    # Corrected Translation widgets to allow negative values
    tx={"label": "Initial Translation: tx (mm)", "tooltip": "Translation along the x-axis.", "min": -5000.0, "max": 5000.0},
    ty={"label": "Initial Translation: ty (mm)", "tooltip": "Translation along the y-axis.", "min": -5000.0, "max": 5000.0},
    tz={"label": "Initial Translation: tz (mm)", "tooltip": "Translation along the z-axis.", "min": -5000.0, "max": 5000.0},

    # Preprocessing
    crop={"label": "X-ray Crop (px)", "min": 0, "max": 1000},
    subtract_background={"label": "Subtract Background"},
    linearize={"label": "Linearize X-ray (exp to linear)"},
    reducefn={"label": "Multiframe Reduce Function", "choices": ["max", "mean", "sum"]},

    # Rendering Options
    labels={"label": "Labels to Render (comma-sep)", "tooltip": "e.g., '1,2,3' (uses mask)"},
    reverse_x_axis={"label": "Reverse X-axis (radiologic convention)"},
    renderer={"label": "Renderer", "choices": ["siddon", "trilinear"]},

    # Optimization Parameters
    scales={"label": "Scales (comma-sep)", "tooltip": "e.g., '8,4,2'"},
    parameterization={"label": "SO(3) Parameterization", "choices": ["euler_angles", "quaternions", "axis_angle", "rotation_6d"]},
    convention={"label": "Euler Convention", "choices": ["ZXY", "XYZ", "ZYX", "YXZ", "YZX", "XZY"]},
    lr_rot={"label": "LR Rotation", "min": 1e-6, "max": 1.0, "step": 1e-4},
    lr_xyz={"label": "LR Translation", "min": 1e-3, "max": 100.0, "step": 0.01},
    patience={"label": "Patience (epochs)", "min": 1, "max": 100},
    threshold={"label": "Threshold (for LR reduction)", "min": 1e-6, "max": 1.0, "step": 1e-5},
    max_n_itrs={"label": "Max Iterations/Scale", "min": 1, "max": 2000, "step": 10},
    max_n_plateaus={"label": "Max Plateaus/Scale", "min": 1, "max": 10},

    # Output & Logging
    init_only={"label": "Initial Pose Only (no refinement)"},
    save_images={"label": "Save Output Images (--saveimg)"},
    verbose_level={"label": "Verbose Level", "min": 0, "max": 3},
)
def xvr_register_fixed_widget(
    volume_path=Path.home(),
    output_path=Path.home(),
    dicom_path=Path.home(),
    mask_path: Path = None,

    orientation: str = "AP",
    rx: float = 0.0,
    ry: float = 0.0,
    rz: float = 0.0,
    tx: float = 0.0,
    ty: float = 0.0,
    tz: float = 1500.0,

    crop: int = 0,
    subtract_background: bool = False,
    linearize: bool = False,
    reducefn: str = "max",

    labels: str = "",
    reverse_x_axis: bool = False,
    renderer: str = "trilinear",

    scales: str = "8",
    parameterization: str = "euler_angles",
    convention: str = "ZXY",
    lr_rot: float = 0.01,
    lr_xyz: float = 1.0,
    patience: int = 10,
    threshold: float = 0.0001,
    max_n_itrs: int = 500,
    max_n_plateaus: int = 3,

    init_only: bool = False,
    save_images: bool = True,
    verbose_level: int = 1,
):
    if not volume_path.is_file():
        show_info(f"Error: NIfTI Volume not found at {volume_path}")
        return
    if not dicom_path.exists():
        show_info(f"Error: DICOM X-ray input not found at {dicom_path}")
        return
    if mask_path and not mask_path.is_file():
        show_info(f"Error: Mask Labelmap not found at {mask_path}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    config = locals()
    config["volume_path"] = str(config["volume_path"])
    config["output_path"] = str(config["output_path"])
    config["dicom_path"] = str(config["dicom_path"])
    config["mask_path"] = str(config["mask_path"]) if config["mask_path"] else None

    run_xvr_register_fixed_cli(config)


# --- Main script execution ---
if __name__ == "__main__":
    viewer = napari.Viewer(title="XVR Fixed-Pose Registration GUI")
    viewer.window.add_dock_widget(xvr_register_fixed_widget, area='right', name="Fixed-Pose Registration")
    napari.run()
