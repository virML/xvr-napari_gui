# XVR Napari GUI Suite

A collection of user-friendly [Napari](https://napari.org/) plugins that provide a graphical user interface (GUI) for the powerful `xvr` 2D/3D registration library.

This project makes the advanced features of `xvr` accessible to clinicians and researchers who may not be comfortable with command-line tools.

### Highlights

  * 🚀 **Graphical Interfaces** for all major `xvr` commands: `train`, `finetune`, `register`, and `view results`.
  * 🖥️ **Cross-Platform:** Works on macOS, Linux, and Windows.
  * 🖱️ **One-Click Launch:** A central launcher panel allows you to open any of the different GUIs with a single click.
  * 🔬 **Interactive:** Built on Napari for integrated, multi-dimensional viewing of medical images.
  * 🔗 **Direct Integration:** Provides a seamless bridge to the powerful `xvr` library developed by Vivek Gopalakrishnan.

### Installation and Setup

Follow these steps to set up the environment and run the GUIs.

#### 1\. Prerequisites

Before you begin, ensure you have **Git** and **Python** (version 3.9 or higher) installed on your system.

#### 2\. Clone This Repository

Open your terminal and clone this repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### 3\. Set Up a Python Virtual Environment

It is highly recommended to use a virtual environment. Choose one of the options below.

**Option A: Using `venv` (Python's built-in tool)**

```bash
# Create the virtual environment (named .venv)
python -m venv .venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .\.venv\Scripts\activate
```

**Option B: Using `conda`**

```bash
# Create and activate the environment
conda create -n xvr-gui python=3.11 -y
conda activate xvr-gui
```

Your terminal prompt should now show the name of your activated environment (e.g., `(.venv)`).

#### 4\. Install Libraries

The installation is a two-step process. First, install the main `xvr` engine. Second, install the dependencies for these GUI scripts.

```bash
# 1. Install the main XVR library from GitHub
# This will download and install xvr, diffdrr, and other core dependencies.
pip install git+https://github.com/eigenvivek/xvr.git

# 2. Install the GUI dependencies from this repository
pip install -r requirements.txt
```

-----

### Usage Guide

#### 1\. Download Data and Models

You will need the datasets and pre-trained models to run the registration. Download them from the official sources:

  * **Datasets:** [https://huggingface.co/datasets/eigenvivek/xvr-data](https://huggingface.co/datasets/eigenvivek/xvr-data)
  * **Pre-trained Models:** [https://huggingface.co/eigenvivek/xvr](https://www.google.com/search?q=https://huggingface.co/eigenvivek/xvr)

Organize these files in a convenient location on your computer.

#### 2\. Configure the GUI Scripts

Before launching a GUI for the first time, you must open its Python script in a text editor and **update the default file paths** at the top of the file to point to the data you downloaded.

For example, in `xvr_register_withanimation.py`, edit this section:

```python
# ⬇️ PLEASE EDIT THESE to point to your default data locations. ⬇️
DEFAULT_VOLUME_PATH = Path.home() / "path/to/your/volume.nii.gz"
DEFAULT_CHECKPOINT_PATH = Path.home() / "path/to/your/model.pth"
# ...and so on
```

#### 3\. Launch the GUI

You can now launch any of the GUIs by running its script from your activated terminal. For example:

```bash
python xvr_register_withanimation.py
```

This will open Napari with the custom widgets ready to use.

-----

### Included GUIs

This repository includes the following tools:

  * **Registration GUI:** (`xvr_register_gui.py`) The main tool for running 2D/3D registration using a pre-trained model.
  * **Training GUI:** (`training-gui.py`) A simple interface for training a new registration model from scratch.
  * **Finetuning GUI:** (`finetune-gui.py`) An interface for finetuning a general model on a specific patient's CT scan.
  * **Pose Viewer GUI:** (`parameters_display_gui.py`) A utility to load a `parameters.pt` file and display the 6 DoF pose.

-----

### Acknowledgments

This project is a graphical wrapper for the `xvr` library.

  * **xvr GitHub Repository:** [https://github.com/eigenvivek/xvr](https://github.com/eigenvivek/xvr)
  * **Original Paper:** [Rapid patient-specific neural networks for intraoperative X-ray to volume registration](https://arxiv.org/abs/2503.16309)
