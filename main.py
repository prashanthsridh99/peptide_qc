import argparse
import yaml
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_skyline_script(raw_file_location, raw_file_name, tic_result_file, tic_log_file):
    """
    Runs the Skyline script with the given parameters.

    Parameters:
    - raw_file_location (str): Path to the raw file location.
    - raw_file_name (str): Name of the raw file.
    - tic_result_file (str): Path to the output TIC result file.
    - tic_log_file (str): Path to the log file.

    Returns:
    - None
    """
    script_command = [
        "python3", "skyline.py",
        "--raw_file_loc", raw_file_location,
        "--raw_file", raw_file_name,
        "--result_file", tic_result_file,
        "--log_file", tic_log_file
    ]
    print("Running script.py with arguments:", " ".join(script_command))
    
    try:
        subprocess.run(script_command, check=True)
        print("Skyline script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running the Skyline script: {e}")

# Example usage:
# run_skyline_script("/path/to/raw/files", "sample.raw", "output.tsv", "log.txt")

def plot_tic_chromatograms(tic_result_file, tic_plot_file, raw_file_name):
    """
    Reads a TIC file and plots all chromatograms.
    
    Parameters:
        tic_result_file (str): Path to the TIC file (tab-separated values)
        tic_plot_file (str): Output path for saving the TIC plot
    """
    # Read TIC data
    TIC = pd.read_csv(tic_result_file, sep='\t')
    
    # Ensure column names are clean
    TIC.columns = TIC.columns.str.strip()
    
    print("---------------------")
    print("CHROMATOGRAM PLOTTING")
    print("---------------------")
    
    # Create a figure for plotting
    plt.figure(figsize=(12, 10))
    plt.suptitle(raw_file_name)  # Set the overall title to raw file name
    plot_index = 1
    for i, row in TIC.iterrows():
        times = np.array(row['Times'].split(','), dtype=float)
        intensities = np.array(row['Intensities'].split(','), dtype=float)
        
        plt.subplot(2, 2, plot_index)
        plt.plot(times, intensities, label="Total Ion Chromatogram")
        
        plt.xlabel("RT [min]")
        plt.ylabel("Intensity")
        plot_index += 1
    
    plt.tight_layout()
    plt.savefig(tic_plot_file)
    plt.show()
    
    print("All chromatograms successfully plotted!....")

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def run_msfragger_script(fragger_params, raw_file_folder, output_folder, proteome, fragger_path, contams_db):
    command = [
        "python3", "msfragger.py",
        "--fragger_params", fragger_params,
        "--raw_file_folder", raw_file_folder,
        "--output_folder", output_folder,
        "--proteome", proteome,
        "--fragger_path", fragger_path,
        "--contams_db", contams_db
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"MSFragger script failed with error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process configuration for MSFragger and XIC Plotting.")
    parser.add_argument(
        "--config-file", type=str, default="config.yaml", help="Path to the configuration YAML file. Default: config.yaml"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = args.config_file
    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return

    config = load_config(config_path)
    print("Loaded configuration:")
    print(yaml.dump(config, default_flow_style=False))

    # Extract config values
    input_csv = config.get("input_csv_file")
    raw_file_name = config.get("raw_file_name")
    raw_file_type = config.get("raw_file_type")
    raw_file_location = config.get("raw_file_location")
   
    tic_plotting = config.get("TIC_Plotting", {})
    tic_result_file = tic_plotting.get("tic_result_file")
    tic_log_file = tic_plotting.get("tic_log_file")
    tic_plot_file = tic_plotting.get("tic_plot_file")
   
    msfragger_part = config.get("MSFraggerPart", {})
    quality_filter = msfragger_part.get("Qualuty_Filter_type")
    q_value_cutoff = msfragger_part.get("q_value_cutoff", 0.01)
    engine_score_cutoff = msfragger_part.get("engine_score_cutoff", 10)
    msfragger_inputs = msfragger_part.get("MSFraggerFunctionInputs", {})
    fragger_params = msfragger_inputs.get("fragger_params")
    raw_file_folder = msfragger_inputs.get("raw_file_folder")
    output_folder = msfragger_inputs.get("output_folder")
    proteome = msfragger_inputs.get("proteome")
    fragger_path = msfragger_inputs.get("fragger_path")
    contams_db = msfragger_inputs.get("contams_db")


    xic_plotting = config.get("XIC_Plotting", {})
    xic_result_file = xic_plotting.get("xic_result_file")
    rt_cutoff = xic_plotting.get("RT_Cuttoff", 60)
    
    

    # Print extracted values
    print(f"Input CSV File: {input_csv}")
    print(f"Raw File Name: {raw_file_name}")
    print(f"Raw File Type: {raw_file_type}")
    print(f"Raw File Location: {raw_file_location}")
    print(f"TIC Result File: {tic_result_file}")
    print(f"TIC Log File: {tic_log_file}")
    print(f"TIC Plot File: {tic_plot_file}")
    print(f"MSFragger Quality Filter Type: {quality_filter}")
    print(f"Q Value Cutoff: {q_value_cutoff}")
    print(f"Engine Score Cutoff: {engine_score_cutoff}")
    print(f"fragger_params: {fragger_params}")
    print(f"Raw File Folder: {raw_file_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Proteome: {proteome}")
    print(f"Fragger Path: {fragger_path}")
    print(f"Contaminants Database: {contams_db}")
    print(f"XIC Plotting Result File: {xic_result_file}")
    print(f"XIC Plotting RT Cutoff: {rt_cutoff}")
    print(f"XIC Result File: {xic_result_file}")

    #run_skyline_script(raw_file_location, raw_file_name, tic_result_file, tic_log_file)
    #plot_tic_chromatograms(tic_result_file, tic_plot_file, raw_file_name)
    run_msfragger_script(fragger_params,raw_file_folder, output_folder, proteome, fragger_path, contams_db)

if __name__ == "__main__":
    main()