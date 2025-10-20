"""
XVR Pose Viewer — Napari Plugin

"""

import torch
import napari
import numpy as np
from pathlib import Path
from magicgui import magicgui
from scipy.spatial.transform import Rotation as R_scipy
from napari.utils.notifications import show_info
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QDoubleSpinBox,
)

# ------------------------------------------------------------------------------
# Optional: Handle pathlib._local PosixPath Deserialization
# ------------------------------------------------------------------------------
# Uncomment if your `.pt` file was saved on a different OS (e.g. Mac → Linux)
# to prevent "pathlib._local.PosixPath not found" deserialization errors.
#
from pathlib import PosixPath
import sys, types
sys.modules['pathlib._local'] = types.SimpleNamespace(PosixPath=PosixPath)

# ------------------------------------------------------------------------------
# Default Parameters File Path
# ------------------------------------------------------------------------------
DEFAULT_PARAMS_FILE = Path(
    "/parameters.pt"
)

# ------------------------------------------------------------------------------
# Napari GUI Widget — XVR Pose Viewer
# ------------------------------------------------------------------------------


@magicgui(
    call_button="Load Parameters & Display Pose",
    parameters_file={
        "widget_type": "FileEdit",
        "mode": "r",
        "label": "Select Parameters (.pt) File",
    },
    layout="vertical",
)
def xvr_pose_viewer_widget(
    parameters_file: Path = DEFAULT_PARAMS_FILE,
    viewer: napari.Viewer = None,  # Injected automatically by Napari
):
    """
    Napari widget for displaying 6 Degrees of Freedom (6 DoF)
    pose parameters (translations in mm, rotations in degrees)
    from a `.pt` parameters file.
    """

    # --------------------------------------------------------------------------
    # One-Time UI Setup (on first run)
    # --------------------------------------------------------------------------
    if not hasattr(xvr_pose_viewer_widget, "_initialized_ui"):
        # Group box for displaying 6 DoF values
        six_dof_group = QGroupBox(
            "6 Degrees of Freedom (Translations in mm, Rotations in deg)"
        )
        six_dof_layout = QVBoxLayout()
        six_dof_group.setLayout(six_dof_layout)

        # --- Rotation Spinboxes (Euler ZXY convention) ---
        xvr_pose_viewer_widget._rot_spinboxes = {}
        for axis in ["Z", "X", "Y"]:
            h_layout = QHBoxLayout()
            label = QLabel(f"Rot {axis}:")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-360.0, 360.0)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(3)
            spinbox.setReadOnly(True)
            h_layout.addWidget(label)
            h_layout.addWidget(spinbox)
            six_dof_layout.addLayout(h_layout)
            xvr_pose_viewer_widget._rot_spinboxes[axis] = spinbox

        # --- Translation Spinboxes (XYZ) ---
        xvr_pose_viewer_widget._trans_spinboxes = {}
        for axis in ["X", "Y", "Z"]:
            h_layout = QHBoxLayout()
            label = QLabel(f"Trans {axis}:")
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-5000.0, 5000.0)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(3)
            spinbox.setReadOnly(True)
            h_layout.addWidget(label)
            h_layout.addWidget(spinbox)
            six_dof_layout.addLayout(h_layout)
            xvr_pose_viewer_widget._trans_spinboxes[axis] = spinbox

        # Add the 6 DoF group box to the widget layout
        xvr_pose_viewer_widget.native.layout().addWidget(six_dof_group)
        xvr_pose_viewer_widget._six_dof_group = six_dof_group
        xvr_pose_viewer_widget._six_dof_layout = six_dof_layout
        xvr_pose_viewer_widget._initialized_ui = True
        print("Initialized 6 DoF display widgets.")

    # --------------------------------------------------------------------------
    # Load the Parameters File
    # --------------------------------------------------------------------------
    show_info(f"Attempting to load parameters from: {parameters_file}")
    print(f"\n--- Loading XVR Parameters from: {parameters_file} ---")

    # Reset GUI display
    for spinbox in xvr_pose_viewer_widget._rot_spinboxes.values():
        spinbox.setValue(0.0)
    for spinbox in xvr_pose_viewer_widget._trans_spinboxes.values():
        spinbox.setValue(0.0)

    if not parameters_file.is_file():
        show_info(f"Error: Parameters file not found at {parameters_file}")
        print(f"Error: Parameters file not found at {parameters_file}")
        return

    try:
        # Load file with full deserialization enabled
        data = torch.load(str(parameters_file), map_location="cpu", weights_only=False)

        if not isinstance(data, dict):
            show_info("Error: Loaded .pt file is not a dictionary.")
            print("Error: Loaded .pt file is not a dictionary.")
            return

        print("Keys found in .pt file:", data.keys())

        final_pose_tensor = None

        # Case 1: final_pose key present directly
        if "final_pose" in data and isinstance(data["final_pose"], torch.Tensor):
            final_pose_tensor = data["final_pose"]
            print("Found 'final_pose' in parameters file.")

        # Case 2: Separate 'rotations' and 'translations'
        elif "rotations" in data and "translations" in data:
            rot_tensor = data["rotations"].squeeze().cpu().numpy()
            trans_tensor = data["translations"].squeeze().cpu().numpy()

            # Ensure correct shapes
            rot_tensor = np.atleast_1d(rot_tensor)
            trans_tensor = np.atleast_1d(trans_tensor)

            if rot_tensor.shape == (3,) and trans_tensor.shape == (3,):
                # Convert Euler angles (radians) → rotation matrix
                rotation_mat = R_scipy.from_euler("ZXY", rot_tensor, degrees=False).as_matrix()

                # Build 4x4 homogeneous pose matrix
                final_pose = np.eye(4)
                final_pose[:3, :3] = rotation_mat
                final_pose[:3, 3] = trans_tensor
                final_pose_tensor = torch.from_numpy(final_pose)

                print("Constructed 'final_pose' from separate rotation/translation tensors.")
            else:
                show_info("Warning: Unexpected tensor shapes for rotation/translation.")
                print("Warning: Unexpected tensor shapes — skipping pose extraction.")

        # ----------------------------------------------------------------------
        # Extract and Display 6 DoF Data
        # ----------------------------------------------------------------------
        if final_pose_tensor is not None:
            final_pose_matrix = final_pose_tensor.squeeze().numpy()

            if final_pose_matrix.shape != (4, 4):
                show_info(f"Error: Invalid final_pose shape: {final_pose_matrix.shape}")
                print(f"Error: Invalid final_pose shape: {final_pose_matrix.shape}")
                return

            # Decompose transformation
            rotation_matrix = final_pose_matrix[:3, :3]
            translation_vec = final_pose_matrix[:3, 3]

            # Convert to Euler angles (ZXY convention)
            euler_angles = R_scipy.from_matrix(rotation_matrix).as_euler("ZXY", degrees=True)

            # Update GUI spinboxes
            xvr_pose_viewer_widget._rot_spinboxes["Z"].setValue(euler_angles[0])
            xvr_pose_viewer_widget._rot_spinboxes["X"].setValue(euler_angles[1])
            xvr_pose_viewer_widget._rot_spinboxes["Y"].setValue(euler_angles[2])
            xvr_pose_viewer_widget._trans_spinboxes["X"].setValue(translation_vec[0])
            xvr_pose_viewer_widget._trans_spinboxes["Y"].setValue(translation_vec[1])
            xvr_pose_viewer_widget._trans_spinboxes["Z"].setValue(translation_vec[2])

            show_info("6 DoF loaded and displayed in GUI.")
            print("\n6 DoF (from parameters file):")
            print(
                f"  Rotations (ZXY deg): "
                f"Z={euler_angles[0]:.3f}, X={euler_angles[1]:.3f}, Y={euler_angles[2]:.3f}"
            )
            print(
                f"  Translations (XYZ mm): "
                f"X={translation_vec[0]:.3f}, Y={translation_vec[1]:.3f}, Z={translation_vec[2]:.3f}"
            )

        else:
            show_info("Warning: No pose data found in parameters file.")
            print("Warning: No 'final_pose' or rotation/translation data found.")

    except Exception as e:
        show_info(f"Error processing parameters file: {e}")
        print(f"Error processing parameters file: {e}")

    finally:
        print("--- End XVR Parameters ---")


# ------------------------------------------------------------------------------
# Launch Napari with the Pose Viewer Widget
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    viewer = napari.Viewer(title="XVR Pose Viewer")
    viewer.window.add_dock_widget(xvr_pose_viewer_widget, area="right")
    napari.run()
