import os
import subprocess
import datetime
import multiprocessing
import shutil
import argparse

# Load necessary libraries
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Run Skyline with input arguments.")
    parser.add_argument("--raw_file_loc", type=str, required=True, help="Location of raw file.")
    parser.add_argument("--raw_file", type=str, required=True, help="Raw file name.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the result file.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file.")
    args = parser.parse_args()

    # Set parameters
    time_out = 30
    max_crashes = 1
    raw_file_loc = args.raw_file_loc
    raw_file = args.raw_file
    result_file = args.result_file
    log_file = args.log_file

    # Get number of CPU cores
    num_cpu = multiprocessing.cpu_count()

    print("-----------")
    print("RUN SKYLINE")
    print("-----------")
    print("RUNNING AUTOMATED SKYLINE IN DOCKER")

    # Copy necessary files
    os.makedirs(raw_file_loc, exist_ok=True)
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    shutil.copy("/data/Prashanth/Skyline/_Skyline-template.sky", raw_file_loc)
    shutil.copy("/data/Prashanth/Skyline/_Skyline-report.skyr", raw_file_loc)

    # Set current date
    date = datetime.datetime.now().strftime("%Y%m%d")

    # Define Skyline command
    skyline_command = f"docker run -i --rm -v {raw_file_loc}:/data chambm/pwiz-skyline-i-agree-to-the-vendor-licenses wine SkylineCmd --timestamp --dir=/data --in=_Skyline-template.sky --save --out=output_{date}.sky --import-file={raw_file} --import-threads={num_cpu} --refine-auto-select-peptides --refine-auto-select-transitions --refine-auto-select-precursors --refine-min-peptides=0 --refine-min-transitions=0 --refine-min-peak-found-ratio=0 --refine-max-peak-found-ratio=1 --refine-minimum-detections=0 --refine-min-dotp=0 --refine-min-idotp=0 --tran-precursor-ion-charges=1,2,3,4,5,6 --tran-product-ion-types=y,p --full-scan-rt-filter-tolerance=0.5 --chromatogram-file=TICs.tsv --chromatogram-tics"

    # Run Skyline in Docker
    print("running Skyline (this can take up to an hour) ....")
    no_crashes = 0
    while no_crashes <= max_crashes:
        try:
            subprocess.run(skyline_command, shell=True, timeout=time_out * 60, check=True)
            shutil.copy(os.path.join(raw_file_loc, "TICs.tsv"), result_file)
            shutil.copy(os.path.join(raw_file_loc, f"output_{date}.sky"), os.path.dirname(result_file))
            break
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            no_crashes += 1

    if no_crashes > max_crashes:
        print("Cannot run Skyline")
    else:
        print("Finished Skyline!")
        print("Success! All data are in place to continue")

if __name__ == "__main__":
    main()
