#!/usr/bin/env python3

import os
import argparse
import yaml
import subprocess
import shutil
from datetime import datetime
import glob

# Define available tasks in sbibm
TASKS = [
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "two_moons",
    "slcp",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "slcp_distractors"
]

# Define model types
MODEL_TYPES = ["normalizing_flow", "flow_matching"]

def create_experiment_dir(base_dir, task, model_type):
    """Create a unique directory for this experiment"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{task}_{model_type}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir

def create_settings(task, model_type, exp_dir):
    """Create a settings.yaml file for this experiment"""
    # Load base settings template
    with open("settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    
    # Update settings for the specific experiment
    settings["task"]["name"] = task
    settings["model"]["type"] = model_type
    
    # For normalizing_flow, set the required posterior_kwargs
    if model_type == "normalizing_flow":
        # Hard-coded dimensions for different tasks - you may need to adjust these
        input_dims = {"two_moons": 2, "gaussian_linear": 10, "gaussian_mixture": 2, 
                     "slcp": 5, "bernoulli_glm": 10, "bernoulli_glm_raw": 10,
                     "gaussian_linear_uniform": 10, "slcp_distractors": 5}
        
        context_dims = {"two_moons": 2, "gaussian_linear": 10, "gaussian_mixture": 2, 
                       "slcp": 5, "bernoulli_glm": 10, "bernoulli_glm_raw": 10,
                       "gaussian_linear_uniform": 10, "slcp_distractors": 8}
        
        settings["model"]["posterior_kwargs"] = {
            "input_dim": input_dims.get(task, 2),  # Default to 2 if unknown
            "context_dim": context_dims.get(task, 2),  # Default to 2 if unknown
            "num_flow_steps": 8,
            "base_transform_kwargs": {
                "hidden_dim": 256,
                "num_transform_blocks": 4,
                "activation": "relu",
                "dropout_probability": 0.1,
                "batch_norm": True,
                "num_bins": 10,
                "tail_bound": 3.0,
                "base_transform_type": "rq-coupling"
            }
        }
    elif model_type == "flow_matching":
        # Example posterior_kwargs for flow_matching
        settings["model"]["posterior_kwargs"] = {
            "activation": "gelu",
            "batch_norm": False,
            "context_with_glu": False,
            "dropout": 0.0,
            "hidden_dims": [32, 64, 128, 256, 512, 1024, 1024, 1024, 512, 128, 64, 32],
            "sigma_min": 0.0001,
            "theta_with_glu": False,
            "time_prior_exponent": 4,
            "type": "DenseResidualNet"
        }
    
    # Save settings to the experiment directory
    settings_path = os.path.join(exp_dir, "settings.yaml")
    with open(settings_path, "w") as f:
        yaml.dump(settings, f, default_flow_style=False)
    
    return settings_path

def clean_previous_outputs():
    """Remove any potential conflicting files in the current directory"""
    # Files that might cause conflicts
    conflict_files = [
        "history.txt", 
        "best_model.pt", 
        "model_latest.pt",
        "c2st.csv"
    ]
    
    # Also remove any posterior_*.png files
    posterior_files = glob.glob("posterior_*.png")
    
    for file in conflict_files + posterior_files:
        if os.path.exists(file):
            print(f"Removing potentially conflicting file: {file}")
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove {file}: {str(e)}")

def run_experiment(exp_dir, settings_path):
    """Run the training and evaluation for this experiment"""
    # Clean any previous output files that might conflict
    clean_previous_outputs()
    
    # Construct the run command with the experiment directory
    train_cmd = f"python run_sbibm.py --train_dir {exp_dir}"
    
    # Run the training
    print(f"Running training: {train_cmd}")
    subprocess.run(train_cmd, shell=True, check=True)
    
    # Move any output files that might have been created in the current directory
    output_files = [
        "history.txt", 
        "best_model.pt", 
        "model_latest.pt",
        "c2st.csv"
    ] 
    
    # Check for posterior plots
    posterior_files = glob.glob("posterior_*.png")
    output_files.extend(posterior_files)
    
    for file in output_files:
        if os.path.exists(file):
            dest_file = os.path.join(exp_dir, file)
            print(f"Moving {file} to {dest_file}")
            shutil.move(file, dest_file)
    
    # Run evaluation again to ensure we get the plots
    # This is important because the first run may not generate plots in some cases
    eval_cmd = f"python evaluate_sbibm.py --train_dir {exp_dir}"
    print(f"Running evaluation: {eval_cmd}")
    subprocess.run(eval_cmd, shell=True, check=True)
    
    # Check for any new files created by evaluation (especially posterior plots)
    posterior_files = glob.glob("posterior_*.png")
    other_files = ["c2st.csv"]
    
    for file in posterior_files + other_files:
        if os.path.exists(file):
            dest_file = os.path.join(exp_dir, file)
            print(f"Moving {file} to {dest_file}")
            shutil.move(file, dest_file)
    
    # Verify that posterior plots exist in the experiment directory
    exp_posterior_files = glob.glob(os.path.join(exp_dir, "posterior_*.png"))
    if not exp_posterior_files:
        print(f"WARNING: No posterior plots found in {exp_dir}")
        
        # Try to generate them explicitly
        print("Attempting to generate posterior plots explicitly...")
        
        # Get task and model information from settings
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
            
        task_name = settings["task"]["name"]
        model_type = settings["model"]["type"]
        
        print(f"Generating posterior plots for task={task_name}, model={model_type}")
        
        # Run evaluation again with explicit plotting
        eval_plot_cmd = f"python evaluate_sbibm.py --train_dir {exp_dir}"
        subprocess.run(eval_plot_cmd, shell=True, check=True)
        
        # Move any newly created plots
        posterior_files = glob.glob("posterior_*.png")
        for file in posterior_files:
            if os.path.exists(file):
                dest_file = os.path.join(exp_dir, file)
                print(f"Moving {file} to {dest_file}")
                shutil.move(file, dest_file)

def main():
    parser = argparse.ArgumentParser(description="Run experiments across various tasks and model types")
    parser.add_argument("--base_dir", type=str, default=".",
                        help="Base directory to store experiment results (default: current directory)")
    parser.add_argument("--tasks", type=str, nargs="+", default=TASKS,
                        help=f"Tasks to run experiments on. Available: {', '.join(TASKS)}")
    parser.add_argument("--models", type=str, nargs="+", default=MODEL_TYPES,
                        help=f"Model types to use. Available: {', '.join(MODEL_TYPES)}")
    args = parser.parse_args()
    
    # Ensure base directory exists
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Iterate through tasks and model types
    for task in args.tasks:
        for model_type in args.models:
            print(f"\n{'='*50}")
            print(f"Running experiment: Task={task}, Model={model_type}")
            print(f"{'='*50}\n")
            
            # Create experiment directory
            exp_dir = create_experiment_dir(args.base_dir, task, model_type)
            
            # Create settings for this experiment
            settings_path = create_settings(task, model_type, exp_dir)
            
            # Run the experiment
            try:
                run_experiment(exp_dir, settings_path)
                print(f"Experiment completed successfully: {exp_dir}")
            except Exception as e:
                print(f"Experiment failed: {str(e)}")
                # Continue with next experiment

if __name__ == "__main__":
    main() 