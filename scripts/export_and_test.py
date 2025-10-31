#!/usr/bin/env python3
"""
Piper Model Export and Test Script
Exports latest checkpoint, creates properly named files, and generates test audio
"""

import os
import sys
import argparse
import subprocess
import shutil
import json
from pathlib import Path

# Default paths (expanduser handles ~)
BASE_DIR = Path.home() / "piper1-gpl"
LIGHTNING_LOGS = BASE_DIR / "lightning_logs"
CONFIG_SOURCE = BASE_DIR / "my-training" / "config.json"
OUTPUT_BASE = BASE_DIR / "my-model"
DEFAULT_TEST_SENTENCE = "CQ Contest, this is Whiskey Seven India Yankee."

def find_latest_version():
    """Find the highest version number in lightning_logs"""
    if not LIGHTNING_LOGS.exists():
        print(f"Error: Lightning logs directory not found: {LIGHTNING_LOGS}")
        return None
    
    versions = []
    for item in LIGHTNING_LOGS.iterdir():
        if item.is_dir() and item.name.startswith("version_"):
            try:
                version_num = int(item.name.split("_")[1])
                versions.append(version_num)
            except (IndexError, ValueError):
                continue
    
    if not versions:
        print(f"Error: No version directories found in {LIGHTNING_LOGS}")
        return None
    
    latest = max(versions)
    print(f"Found latest version: {latest}")
    return latest

def find_latest_checkpoint(version_num):
    """Find the most recent checkpoint in the version directory"""
    checkpoint_dir = LIGHTNING_LOGS / f"version_{version_num}" / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        return None
    
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    
    if not checkpoints:
        print(f"Error: No checkpoints found in {checkpoint_dir}")
        return None
    
    # Sort by modification time, get the most recent
    latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
    print(f"Found latest checkpoint: {latest_checkpoint.name}")
    return latest_checkpoint

def export_model(checkpoint_path, output_dir):
    """Export checkpoint to ONNX model"""
    print(f"\nExporting model from checkpoint...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output dir: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run piper export command (GPL version)
    cmd = [
        sys.executable, "-m", "piper.train.export_onnx",
        "--checkpoint", str(checkpoint_path),
        "--output-file", str(output_dir / "model.onnx")
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Export successful!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Export failed with error:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("Error: piper_train.export_onnx not found. Make sure piper is installed.")
        return False

def setup_model_files(output_dir, model_name, config_source):
    """Rename exported files and copy config to match piper's naming convention"""
    print(f"\nSetting up model files with base name: en_US-{model_name}")
    
    # Expected files after export
    model_file = output_dir / "model.onnx"
    config_file = output_dir / "config.json"
    
    # Target names
    target_model = output_dir / f"en_US-{model_name}.onnx"
    target_config = output_dir / f"en_US-{model_name}.onnx.json"
    
    # Check if model was exported
    if not model_file.exists():
        print(f"Error: Exported model not found: {model_file}")
        return False
    
    # Rename model file
    if target_model.exists():
        print(f"Removing existing model: {target_model}")
        target_model.unlink()
    
    shutil.move(str(model_file), str(target_model))
    print(f"✓ Created: {target_model.name}")
    
    # Copy and rename config file
    if config_source.exists():
        if target_config.exists():
            print(f"Removing existing config: {target_config}")
            target_config.unlink()
        
        shutil.copy(str(config_source), str(target_config))
        print(f"✓ Created: {target_config.name}")
    elif config_file.exists():
        # Use exported config if source not available
        shutil.move(str(config_file), str(target_config))
        print(f"✓ Created: {target_config.name} (from export)")
    else:
        print(f"Warning: Config file not found at {config_source}")
        return False
    
    return True

def generate_test_audio(model_path, text, output_wav, **synthesis_params):
    """Generate test audio using piper"""
    print(f"\nGenerating test audio...")
    print(f"  Model: {model_path.name}")
    print(f"  Text: {text}")
    print(f"  Output: {output_wav}")
    
    # Build piper command
    cmd = ["piper", "--model", str(model_path), "--output_file", str(output_wav)]
    
    # Add synthesis parameters if provided
    if synthesis_params:
        print(f"  Synthesis parameters:")
        if 'length_scale' in synthesis_params:
            cmd.extend(["--length_scale", str(synthesis_params['length_scale'])])
            print(f"    length_scale: {synthesis_params['length_scale']}")
        if 'noise_scale' in synthesis_params:
            cmd.extend(["--noise_scale", str(synthesis_params['noise_scale'])])
            print(f"    noise_scale: {synthesis_params['noise_scale']}")
        if 'noise_w' in synthesis_params:
            cmd.extend(["--noise_w", str(synthesis_params['noise_w'])])
            print(f"    noise_w: {synthesis_params['noise_w']}")
    
    try:
        # Send text to piper via stdin
        result = subprocess.run(
            cmd,
            input=text,
            text=True,
            check=True,
            capture_output=True
        )
        print(f"✓ Audio generated: {output_wav}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating audio:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("Error: 'piper' command not found. Make sure piper is installed and in PATH.")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Export Piper checkpoint and generate test audio',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python export_and_test.py --name w7iy
  
  # Custom test sentence
  python export_and_test.py --name w7iy --text "Hello from my custom voice"
  
  # With synthesis parameters (slower speech, more variation)
  python export_and_test.py --name w7iy --length_scale 1.2 --noise_scale 0.8
  
  # Specify custom paths
  python export_and_test.py --name w7iy --version 3 --output /tmp/my-model
  
Synthesis Parameters:
  --length_scale: Speed (default 1.0, <1.0=faster, >1.0=slower)
  --noise_scale:  Variation in speech (default 0.667, higher=more variation)
  --noise_w:      Phoneme duration variation (default 0.8)
        """
    )
    
    parser.add_argument('--name', type=str, required=True,
                       help='Model name (e.g., w7iy)')
    parser.add_argument('--version', type=int, default=None,
                       help='Specific version number (default: auto-detect latest)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint file (default: latest in version)')
    parser.add_argument('--output', type=str, default=None,
                       help=f'Output directory (default: {OUTPUT_BASE})')
    parser.add_argument('--text', type=str, default=DEFAULT_TEST_SENTENCE,
                       help=f'Test sentence (default: "{DEFAULT_TEST_SENTENCE}")')
    parser.add_argument('--no-test', action='store_true',
                       help='Skip audio generation test')
    
    # Synthesis parameters
    parser.add_argument('--length_scale', type=float, default=None,
                       help='Speaking rate (1.0=normal, <1.0=faster, >1.0=slower)')
    parser.add_argument('--noise_scale', type=float, default=None,
                       help='Variation in speech (default 0.667)')
    parser.add_argument('--noise_w', type=float, default=None,
                       help='Phoneme duration variation (default 0.8)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Piper Model Export and Test")
    print("=" * 70)
    
    # Determine version
    if args.version is not None:
        version_num = args.version
        print(f"Using specified version: {version_num}")
    else:
        version_num = find_latest_version()
        if version_num is None:
            return 1
    
    # Determine checkpoint
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return 1
        print(f"Using specified checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = find_latest_checkpoint(version_num)
        if checkpoint_path is None:
            return 1
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_BASE
    
    # Export model
    if not export_model(checkpoint_path, output_dir):
        return 1
    
    # Setup files with proper naming
    if not setup_model_files(output_dir, args.name, CONFIG_SOURCE):
        return 1
    
    # Generate test audio if requested
    if not args.no_test:
        model_path = output_dir / f"en_US-{args.name}.onnx"
        test_wav = output_dir / f"test_{args.name}.wav"
        
        # Collect synthesis parameters
        synthesis_params = {}
        if args.length_scale is not None:
            synthesis_params['length_scale'] = args.length_scale
        if args.noise_scale is not None:
            synthesis_params['noise_scale'] = args.noise_scale
        if args.noise_w is not None:
            synthesis_params['noise_w'] = args.noise_w
        
        if not generate_test_audio(model_path, args.text, test_wav, **synthesis_params):
            return 1
        
        print(f"\n{'=' * 70}")
        print("SUCCESS! Model exported and tested.")
        print(f"{'=' * 70}")
        print(f"\nModel files:")
        print(f"  {output_dir / f'en_US-{args.name}.onnx'}")
        print(f"  {output_dir / f'en_US-{args.name}.onnx.json'}")
        print(f"\nTest audio:")
        print(f"  {test_wav}")
        print(f"\nTo use this model:")
        print(f"  echo 'Your text here' | piper --model {output_dir / f'en_US-{args.name}.onnx'} --output_file output.wav")
    else:
        print(f"\n{'=' * 70}")
        print("SUCCESS! Model exported (audio test skipped).")
        print(f"{'=' * 70}")
        print(f"\nModel files:")
        print(f"  {output_dir / f'en_US-{args.name}.onnx'}")
        print(f"  {output_dir / f'en_US-{args.name}.onnx.json'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

