# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import ast
from pathlib import Path
import sys
from types import SimpleNamespace
from collections import OrderedDict

def str2bool(v):
    """Convert string to boolean for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_command_line_args():
    """Parse command line arguments and override config values."""
    parser = argparse.ArgumentParser(description='Process command line arguments')
    
    # Add --config_file argument to allow specifying an alternative config file
    parser.add_argument('--config_file', type=str, help='Path to custom config file')
    
    # Parse known args first to get the config file if specified
    known_args, remaining_args = parser.parse_known_args()
    
    # Import config after potentially getting an alternative config file
    if known_args.config_file:
        sys.path.insert(0, str(Path(known_args.config_file).parent))
        config_name = Path(known_args.config_file).stem
        config = __import__(config_name)
    else:
        import config
    
    # Add arguments dynamically based on FLAGS in config
    for key, value in config.FLAGS.__dict__.items():
        arg_type = type(value)
        
        # Handle all types as strings initially, then parse them appropriately later
        if key == 'dynamic_prompt_settings' or arg_type in (list, dict):
            parser.add_argument(f'--{key}', type=str, help=f'Override {key} (default: {value})')
        elif arg_type == bool:
            parser.add_argument(f'--{key}', type=str, help=f'Override {key} (default: {value})')
        elif arg_type == type(None):
            parser.add_argument(f'--{key}', type=str, help=f'Override {key} (default: None)')
        elif arg_type == float and str(value) == 'inf':
            parser.add_argument(f'--{key}', type=str, help=f'Override {key} (default: inf)')
        else:
            parser.add_argument(f'--{key}', type=str, help=f'Override {key} (default: {value})')
    
    # Parse all arguments
    args = parser.parse_args()
    
    # Update config.FLAGS with command line arguments
    for key, value in vars(args).items():
        if key != 'config_file' and value is not None:
            orig_value = getattr(config.FLAGS, key, None)
            
            # Special handling for each type
            if isinstance(orig_value, bool):
                try:
                    parsed_value = str2bool(value)
                    setattr(config.FLAGS, key, parsed_value)
                except (ArgumentTypeError, ValueError):
                    print(f"Warning: Could not parse {key}={value} as a boolean. Using original value.")
            elif isinstance(orig_value, (list, dict)) or key == 'dynamic_prompt_settings':
                try:
                    parsed_value = ast.literal_eval(value)
                    setattr(config.FLAGS, key, parsed_value)
                except (SyntaxError, ValueError):
                    print(f"Warning: Could not parse {key}={value} as a Python literal. Using as string.")
                    setattr(config.FLAGS, key, value)
            elif isinstance(orig_value, float) and str(orig_value) == 'inf' and value.lower() == 'inf':
                setattr(config.FLAGS, key, float('inf'))
            elif isinstance(orig_value, int):
                try:
                    parsed_value = int(value)
                    setattr(config.FLAGS, key, parsed_value)
                except ValueError:
                    print(f"Warning: Could not parse {key}={value} as an integer. Using original value.")
            elif isinstance(orig_value, float):
                try:
                    parsed_value = float(value)
                    setattr(config.FLAGS, key, parsed_value)
                except ValueError:
                    print(f"Warning: Could not parse {key}={value} as a float. Using original value.")
            else:
                # For strings and other types
                setattr(config.FLAGS, key, value)
    
    return config.FLAGS
