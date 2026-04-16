import os
import glob
import subprocess

INPUT_DIR = "/home/vlad/IsaacLab/vlad/master_isaac/asyncDualPlayPPO/paper-async"
OUTPUT_DIR = os.path.join(INPUT_DIR, "md_files")

def convert_pdfs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pdfs = glob.glob(os.path.join(INPUT_DIR, "*.pdf"))
    print(f"Find {len(pdfs)} PDF")

    for pdf in pdfs:
        print(f"Read {pdf}")
        try:
            cmd = ["marker_single", pdf, "--output_dir", OUTPUT_DIR]
            subprocess.run(cmd, check=True)
            print(f"Done {pdf}")
        except subprocess.CalledProcessError as e:
            print(f"Fail {pdf}: {e}")
        except FileNotFoundError:
            print("Error: marker_single not found. Ensure environment active.")
            break
        
if __name__ == "__main__":
    convert_pdfs()
