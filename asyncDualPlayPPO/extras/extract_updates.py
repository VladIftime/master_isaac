import os

log_file = '/home/vlad/IsaacLab/dual_arm_Isaacgym/asyncDualPlayPPO/logs/25_rot.out'
output_file = '/home/vlad/IsaacLab/dual_arm_Isaacgym/asyncDualPlayPPO/logs/25_rot_updates_only.out'

inside_update = False
current_block = []
separator_count = 0

with open(log_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        # Check if line indicates a separator
        if "============================================================" in line:
            if separator_count == 0:
                # Start of a new block
                inside_update = True
                current_block = [line]
                separator_count = 1
            elif separator_count == 1:
                # Middle separator (after title)
                current_block.append(line)
                separator_count = 2
            elif separator_count == 2:
                # End of the entire block
                current_block.append(line)
                # Check if it was an update block
                if any("UPDATE" in l for l in current_block):
                    f_out.writelines(current_block)
                    f_out.write("\n")
                
                # Reset for the next block
                inside_update = False
                current_block = []
                separator_count = 0
        elif inside_update:
            current_block.append(line)

print(f"Extraction complete! Saved to {output_file}")
