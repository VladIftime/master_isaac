import os

def extract_optuna_logs(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"Input file {input_file} not found.")
        return

    print(f"Extracting updates from {input_file}...")
    
    keywords = [
        "TRIAL",
        "min_goal_dist",
        "goal_margin",
        "bob_completion",
        "rot_threshold",
        "alpha_decay_steps",
        "learning_rate",
        "entropy_coef",
        "clip_param",
        "demo_batch_ratio",
        "nsteps(auto)",
        "max_steps_bailout",
        "Alice Update",
        "Bob Update",
        "Bob Storage",
        "Alice Throttled",
        "FINAL Bob Success Rate",
        "OPTUNA SWEEP"
    ]
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            line_stripped = line.strip()
            # Skip Isaac Lab startup pipes
            if line_stripped.startswith("|"):
                continue
                
            if any(key in line for key in keywords):
                f_out.write(line)
            elif "============================================================" in line:
                # Keep the clean separators that don't have pipes
                f_out.write(line)

    print(f"Extraction complete! Saved to {output_file}")

if __name__ == "__main__":
    input_log = "/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/optuna_64.out"
    output_log = "/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/optuna_64_updates.out"
    
    extract_optuna_logs(input_log, output_log)
