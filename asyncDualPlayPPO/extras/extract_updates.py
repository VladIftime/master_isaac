import os

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
name_of_file = 'slurm-27623781-bco_med'
name_of_output= name_of_file + '_updates'
log_file = os.path.join(base_dir, 'logs', name_of_file+'.out')
output_file = os.path.join(base_dir, 'logs', name_of_output+'.out')

state = 0
block = []

with open(log_file, 'r') as f_in, open(output_file, 'w') as f_out:
    for line in f_in:
        is_sep = "============================================================" in line
        if state == 0:
            if is_sep:
                block = [line]
                state = 1
        elif state == 1:
            if "UPDATE" in line and not is_sep:
                block.append(line)
                state = 2
            else:
                state = 1 if is_sep else 0
                block = [line] if is_sep else []
        elif state == 2:
            if is_sep:
                block.append(line)
                state = 3
            else:
                state = 1 if is_sep else 0
                block = [line] if is_sep else []
        elif state == 3:
            block.append(line)
            if is_sep:
                # End of block!
                f_out.writelines(block)
                f_out.write("\n")
                state = 0
                block = []

print(f"Extraction complete! Saved to {output_file}")
