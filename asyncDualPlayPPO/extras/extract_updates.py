import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
name_of_file = "local_test copy"
name_of_output = name_of_file + "_updates"
log_file = os.path.join(base_dir, "logs", name_of_file + ".txt")
output_file = os.path.join(base_dir, "logs", name_of_output + ".txt")

# Patterns to extract
patterns = [
    re.compile(r"\[Alice Update \d+\]"),  # Alice update with stats
    re.compile(r"\[Bob Update \d+\]"),  # Bob update with stats
    re.compile(r"\[Alice Update\] Skipped"),  # Alice skipped updates
    re.compile(r"^Iteration \d+:"),  # Iteration summary
]

count = 0
with open(log_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        if any(p.search(line) for p in patterns):
            f_out.write(line)
            count += 1

print(f"Extracted {count} lines to {output_file}")
