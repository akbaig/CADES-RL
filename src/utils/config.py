import os
import argparse
import yaml
from types import SimpleNamespace

def load_yaml_config(config_file):
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

def merge_configs(*configs):
    merged_config = {}
    for config in configs:
        merged_config.update(config)
    return merged_config

def dict_to_namespace(config_dict):
    return SimpleNamespace(**config_dict)

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, nargs='+', help='Paths to the config files')

    # Determine the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Define paths to the descriptions and default config files in the 'configs' subdirectory
    base_config_path = os.path.join(script_dir, 'configs')
    descriptions_path = os.path.join(base_config_path, 'description.yaml')
    default_config_path = os.path.join(base_config_path, 'default.yaml')
    
    # Load descriptions from descriptions.yaml
    descriptions = load_yaml_config(descriptions_path)
    
    # Add arguments for each configuration parameter based on descriptions
    for arg, desc in descriptions.items():
        parser.add_argument(
            f"--{arg}",
            type=str,  # We'll parse and convert these to the correct types later
            help=desc,
        )

    args = parser.parse_args()

    # Load the default YAML config file
    default_config = load_yaml_config(default_config_path)

    # If additional config files are provided
    if args.config:
        # Load and merge user-specified YAML config files in the order they are provided
        user_configs = [load_yaml_config(config_file) for config_file in args.config]
        merged_config = merge_configs(default_config, *user_configs)
    else:
        merged_config = default_config
    
    # Convert argparse Namespace to dict and remove None values
    cli_args = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}

    # Update merged config with CLI args, converting types as necessary
    final_config = merged_config.copy()
    for key, value in cli_args.items():
        if key in merged_config:
            final_config[key] = type(merged_config[key])(value)

    # Convert final_config to an object
    return dict_to_namespace(final_config)