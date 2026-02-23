# Dynamic ScanNet++

A project for preprocessing ScanNet data for inference with ShapeR.

## Setup

### 1. Clone the repository with submodules

```bash
git clone --recursive <your-repo-url>
cd dyna-scannet
```

If you already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### 2. Test the submodule

```bash
python scripts/test_submodule.py
```

### 3. Install dependencies

See `external/shaper/INSTALL.md` for ShapeR installation instructions.

## Running ShapeR Inference

### Method 1: Using the wrapper script (recommended)

From the repository root, use the wrapper script:

```bash
# Basic usage with a pickle file in shaper/data/
python scripts/run_shaper_inference.py --input_pkl sample.pkl

# With absolute path
python scripts/run_shaper_inference.py --input_pkl /path/to/sample.pkl --is_local_path

# With quality config and visualization
python scripts/run_shaper_inference.py --input_pkl sample.pkl --config quality --save_visualization

# Available configs: quality (best, slowest), balance (recommended), speed (fastest)
```

### Method 2: Direct submodule usage

You can also run ShapeR directly from the submodule:

```bash
cd external/shaper
python infer_shape.py --input_pkl <sample.pkl> --config balance --output_dir output
```

**Arguments:**
- `--input_pkl`: Path to preprocessed pickle file (relative to `data/` or absolute with `--is_local_path`)
- `--config`: Preset configuration (`quality`, `balance`, or `speed`)
- `--output_dir`: Output directory for meshes and visualizations
- `--save_visualization`: Save comparison visualization
- `--do_transform_to_world`: Transform output mesh to world coordinates



## ShapeR Submodule

This project uses [ShapeR](https://github.com/facebookresearch/ShapeR) as a submodule for 3D shape generation.

To update the submodule to the latest version:

```bash
cd external/shaper
git pull origin main
cd ../..
git add external/shaper
git commit -m "Update ShapeR submodule"
```