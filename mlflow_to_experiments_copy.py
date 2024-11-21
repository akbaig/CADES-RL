import os
import yaml
import shutil

def get_yaml_value(file_path, key):
    with open(file_path, 'r') as file:
        content = yaml.safe_load(file)
        return content.get(key, None)

def main():
    mlruns_path = 'src/mlruns'
    experiments_path = 'experiments copy/models'

    # Traverse each subfolder in mlruns directory
    for subfolder in os.listdir(mlruns_path):
        subfolder_path = os.path.join(mlruns_path, subfolder)

        # Skip subfolders starting with "0" or ".trash"
        if subfolder.startswith("0") or subfolder.startswith(".trash"):
            continue

        # Skip subfolders which have more than 3 folders within them
        if len([name for name in os.listdir(subfolder_path) if os.path.isdir(os.path.join(subfolder_path, name))]) > 3:
            continue

        # Ensure it's a directory
        if not os.path.isdir(subfolder_path):
            continue

        # Read the meta.yaml and extract the "name" value
        meta_file_path = os.path.join(subfolder_path, 'meta.yaml')
        if not os.path.exists(meta_file_path):
            continue

        value = get_yaml_value(meta_file_path, 'name')
        if not value:
            continue

        # Separate parts of <value> by underscore
        val_parts = value.split('_')
        if len(val_parts) < 2:
            continue
        val1 = val_parts[0]
        val2 = '_'.join(val_parts[1:])

        # Check if the destination directory exists
        destination_path = os.path.join(experiments_path, val1, val2)
        if not os.path.exists(destination_path):
            continue

        # Traverse subfolders of the current mlruns subfolder
        for run_subfolder in os.listdir(subfolder_path):
            run_subfolder_path = os.path.join(subfolder_path, run_subfolder)
            if not os.path.isdir(run_subfolder_path):
                continue

            # Read the meta.yaml in the run subfolder and look for "run_name" param
            run_meta_file_path = os.path.join(run_subfolder_path, 'meta.yaml')
            if not os.path.exists(run_meta_file_path):
                continue

            run_name = get_yaml_value(run_meta_file_path, 'run_name')
            if not run_name:
                continue

            # Determine file names and paths based on run_name
            artifacts_path = os.path.join(run_subfolder_path, 'artifacts/models')
            if not os.path.exists(artifacts_path):
                continue

            if run_name.startswith('ppo_early_term'):
                dest_best = os.path.join(destination_path, 'early_term_best.zip')
                dest_1000 = os.path.join(destination_path, 'early_term_1000.zip')
            elif run_name.startswith('ppo_invalid_replace'):
                dest_best = os.path.join(destination_path, 'act_replace_best.zip')
                dest_1000 = os.path.join(destination_path, 'act_replace_1000.zip')
            elif run_name.startswith('ppo_logits_mask'):
                dest_best = os.path.join(destination_path, 'act_mask_best.zip')
                dest_1000 = os.path.join(destination_path, 'act_mask_1000.zip')
            else:
                continue

            # Copy the relevant files
            best_model_src = os.path.join(artifacts_path, 'best_model.zip')
            epoch_1000_src = os.path.join(artifacts_path, 'epoch_1000.zip')

            if os.path.exists(best_model_src):
                shutil.copy(best_model_src, dest_best)

            if os.path.exists(epoch_1000_src):
                shutil.copy(epoch_1000_src, dest_1000)

if __name__ == "__main__":
    main()