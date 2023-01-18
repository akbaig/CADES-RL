import argparse

parser = argparse.ArgumentParser()
arg_lists = []

def str2bool(v):
    return v.lower() in ("true", "1")

parameters_definition = {

    # PROBLEM CONDITIONS #
    "min_item_size": { "value": 100, "type": int, "desc": "Minimum item size"},
    "max_item_size": { "value": 800, "type": int, "desc": "Maximum item size"},
    "min_num_items": { "value": 10, "type": int, "desc": "Minimum number of items"},
    "max_num_items": { "value": 20, "type": int, "desc": "Maximum number of items"},
    "bin_size": { "value": 1000, "type": int, "desc": "Bin size"},
    "agent_heuristic": {
        "value": "NF",
        "type": str, 
        "desc": "HeuriStic used by the agent to allocate the sequence output"
    },
    "number_of_copies": {"value": 2, "type": int, "desc": "Number of critical item copies"},
    "number_of_critical_items": {"value": 3, "type": int, "desc": "Number of critical item"},

    # TRAINING PARAMETERS #
    "seed": { "value": 3, "type": int, "desc": "Random seed"},
    "n_episodes": { "value": 1000, "type": int, "desc": "Number of episodes"},
    "batch_size": { "value": 8, "type": int, "desc": "Batch size"},
    "lr": { "value": 1.0e-3, "type": float, "desc": "Initial learning rate"},
    "alpha": {"value": 0.3, "type": float, "desc": "Alpha Value to compute reward"},

    # NETWORK PARAMETERS #
    "hid_dim": { "value": 128, "type": int, "desc": "Hidden dimension"},

    # RUN OPTIONS #
    "device": { "value": "cpu", "type": str, "desc": "Device to use (if no GPU available, value should be 'cpu')"},
    "inference": {"value": False, "type": str2bool, "desc": "Do not train the model"},
    "model_path": {
        # "value": "./experiments/models/policy_dnn_100_800_10_20_100_FF.pkl",
        "value": "/bin-packing-drl/experiments/models/policy_dnn_10_20_NF_3_Decoder.pkl",
        "type": str, 
        "desc": "Path to the model checkpoint to save if in training mode, or to load if in inference mode"
    },
    "inference_data_path": {
        # "value": "./experiments/inference_data/input_states.txt",
        "value": "",
        "type": str,
        "desc": "Path to the inference data. If None, a random batch of states will be generated according to the config parameters"
    }
}

def get_config():
    parser = argparse.ArgumentParser()
    for arg, arg_def in parameters_definition.items():
        parser.add_argument(f"--{arg}", type=arg_def["type"], default=arg_def["value"], help=arg_def["desc"])
    config, unparsed = parser.parse_known_args()
    return config, unparsed
