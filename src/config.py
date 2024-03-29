import argparse

parser = argparse.ArgumentParser()
arg_lists = []


def str2bool(v):
    return v.lower() in ("true", "1")


parameters_definition = {
    "min_task_size": {"value": 4, "type": int, "desc": "Minimum task size"},
    "max_task_size": {"value": 4, "type": int, "desc": "Maximum task size"},
    "min_num_tasks": {"value": 12, "type": int, "desc": "Minimum number of tasks"},
    "max_num_tasks": {"value": 12, "type": int, "desc": "Maximum number of tasks"},
    "min_node_size": {"value": 12, "type": int, "desc": "Minimum node size"},
    "max_node_size": {"value": 12, "type": int, "desc": "Maximum node size"},
    "total_nodes": {"value": 8, "type": int, "desc": "Total number of nodes"},
    "number_of_copies": {
        "value": 2,
        "type": int,
        "desc": "Number of critical task copies",
    },
    "number_of_critical_tasks": {
        "value": 3,
        "type": int,
        "desc": "Number of critical task",
    },
    "min_num_comms": {
        "value": 10,
        "type": int,
        "desc": "Min number of communications",
    },
    "max_num_comms": {
        "value": 10,
        "type": int,
        "desc": "Max number of communications",
    },
    # TRAINING PARAMETERS #
    "seed": {"value": 3, "type": int, "desc": "Random seed"},
    "epochs": {"value": 150, "type": int, "desc": "Number of episodes"},
    "batch_size": {"value": 64, "type": int, "desc": "Batch size"},
    "lr": {"value": 0.0003, "type": float, "desc": "Initial learning rate"},
    "alpha": {"value": 0.3, "type": float, "desc": "Alpha Value to compute reward"},
    # RUN OPTIONS #
    "device": {
        "value": "cuda",
        "type": str,
        "desc": "Device to use (if no GPU available, value should be 'cpu')",
    },
    "inference": {"value": False, "type": str2bool, "desc": "Do not train the model"},
    "experiment_name": {
        "value": "message_communication_channel_2d_matrix",
        "type": str,
        "desc": "Experiment Name for mlflow",
    },
    "run_name": {
        "value": "Message Communication Channel 2D Matrix - Run 1",
        "type": str,
        "desc": "Run Name for mlflow",
    },
    # REWARD SHAPING
    "SUCCESS_reward": {"value": 10, "type": int, "desc": "Success Reward"},
    "DUPLICATE_PICK_reward": {
        "value": -1,
        "type": int,
        "desc": "DUPLICATE_PICK Reward",
    },
    "NODE_OVERFLOW_reward": {"value": -2, "type": int, "desc": "NODE_OVERFLOW Reward"},
    "STEP_reward": {"value": 1, "type": int, "desc": "Step Reward"},
    "BONUS_reward": {"value": 0.25, "type": int, "desc": "Step Reward"},
    "CRITICAL_reward": {"value": 1, "type": int, "desc": "Critical Task Reward"},
    "DUPLICATE_CRITICAL_PICK_reward": {
        "value": -1,
        "type": int,
        "desc": "Duplicate Critical Task Reward",
    },
    "COMM_reward": {"value": 10, "type": int, "desc": "Total Communication Reward"},
}


def get_config():
    parser = argparse.ArgumentParser()
    for arg, arg_def in parameters_definition.items():
        parser.add_argument(
            f"--{arg}",
            type=arg_def["type"],
            default=arg_def["value"],
            help=arg_def["desc"],
        )
    config, unparsed = parser.parse_known_args()
    return config, unparsed
