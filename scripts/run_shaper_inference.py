#!/usr/bin/env python3
"""
Wrapper script to run ShapeR inference from the parent repository.

This script allows you to run ShapeR inference without having to cd into the submodule.
It handles path resolution and ensures the submodule is properly set up.
"""

import sys
import subprocess
from pathlib import Path

def get_shaper_path():
    """Get the path to the ShapeR submodule."""
    repo_root = Path(__file__).parent.parent
    shaper_path = repo_root / "external" / "shaper"
    
    if not shaper_path.exists():
        raise FileNotFoundError(
            f"ShapeR submodule not found at {shaper_path}\n"
            "Run: git submodule update --init --recursive"
        )
    
    return shaper_path

def run_inference(input_pkl, config="balance", output_dir="output", **kwargs):
    """
    Run ShapeR inference.
    
    Args:
        input_pkl: Path to input pickle file (can be relative to shaper/data or absolute)
        config: Inference config ('quality', 'balance', or 'speed')
        output_dir: Output directory for results
        **kwargs: Additional arguments to pass to infer_shape.py
    """
    shaper_path = get_shaper_path()
    infer_script = shaper_path / "infer_shape.py"
    
    if not infer_script.exists():
        raise FileNotFoundError(f"ShapeR inference script not found at {infer_script}")
    
    # Build command
    cmd = [
        sys.executable,
        str(infer_script),
        "--input_pkl", str(input_pkl),
        "--config", config,
        "--output_dir", str(output_dir),
    ]
    
    # Add optional arguments
    if kwargs.get("save_visualization", False):
        cmd.append("--save_visualization")
    
    if kwargs.get("do_transform_to_world", False):
        cmd.append("--do_transform_to_world")
    
    if kwargs.get("is_local_path", False):
        cmd.append("--is_local_path")
    
    # Change to shaper directory to run (so relative paths work correctly)
    print(f"Running ShapeR inference...")
    print(f"  Input: {input_pkl}")
    print(f"  Config: {config}")
    print(f"  Output: {output_dir}")
    print()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(shaper_path),
            check=True
        )
        print("\n✅ Inference completed successfully!")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Inference failed with error code {e.returncode}")
        return e.returncode

def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run ShapeR inference from the parent repository",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with a pickle file in shaper/data/
  python scripts/run_shaper_inference.py --input_pkl sample.pkl
  
  # Run with absolute path to pickle file
  python scripts/run_shaper_inference.py --input_pkl /path/to/sample.pkl --is_local_path
  
  # Run with quality config and save visualization
  python scripts/run_shaper_inference.py --input_pkl sample.pkl --config quality --save_visualization
        """
    )
    
    parser.add_argument(
        "--input_pkl",
        type=str,
        required=True,
        help="Path to input pickle file (relative to shaper/data/ or absolute with --is_local_path)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="balance",
        choices=["quality", "balance", "speed"],
        help="Inference configuration preset (default: balance)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for meshes and visualizations (default: output)"
    )
    
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Save comparison visualization"
    )
    
    parser.add_argument(
        "--do_transform_to_world",
        action="store_true",
        help="Transform output mesh to world coordinates"
    )
    
    parser.add_argument(
        "--is_local_path",
        action="store_true",
        help="Treat input_pkl as an absolute path (don't look in shaper/data/)"
    )
    
    args = parser.parse_args()
    
    return run_inference(
        args.input_pkl,
        config=args.config,
        output_dir=args.output_dir,
        save_visualization=args.save_visualization,
        do_transform_to_world=args.do_transform_to_world,
        is_local_path=args.is_local_path,
    )

if __name__ == "__main__":
    sys.exit(main())
