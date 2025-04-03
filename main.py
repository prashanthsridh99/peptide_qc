import argparse
import yaml
import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from pathlib import Path
import platform
from datetime import datetime

import subprocess
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages


def run_skyline_docker_xic(raw_file_folder, proteome):
    """
    Runs SkylineCmd inside a Docker container with the provided raw_file_folder mounted as /data.
    """
    # Check if peptide_qc_results folder exists in the raw_file_folder and delete it
    peptide_qc_results_folder = os.path.join(raw_file_folder, "peptide_qc_results")
    if os.path.exists(peptide_qc_results_folder):
        try:
            print(f"Deleting existing folder: {peptide_qc_results_folder}...")
            subprocess.run(["rm", "-rf", peptide_qc_results_folder], check=True)
            print("Folder deleted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while deleting the folder: {e}")
    
    # Check if _Skyline-template.skyd exists in the raw_file_folder and delete it
    skyline_template_skyd = os.path.join(raw_file_folder, "_Skyline-template.skyd")
    if os.path.exists(skyline_template_skyd):
        try:
            print(f"Deleting existing file: {skyline_template_skyd}...")
            os.remove(skyline_template_skyd)
            print("File deleted successfully.")
        except OSError as e:
            print(f"Error occurred while deleting the file: {e}")
    # Copy the proteome file to the raw_file_folder
    proteome_file_name = os.path.basename(proteome)
    destination_path = os.path.join(raw_file_folder, os.path.basename(proteome))
    if not os.path.exists(destination_path):
        try:
            print(f"Copying {proteome} to {destination_path}...")
            subprocess.run(["cp", proteome, destination_path], check=True)
            print("Proteome file copied successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while copying the proteome file: {e}")
    else:
        print(f"Proteome file already exists at {destination_path}.")
    # Get timestamp for output naming
    timestamp = datetime.now().strftime("%Y%m%d")

    # Define output paths (inside the container /data maps to raw_file_folder)
    output_sky = f"/data/peptide_qc_results/Skyline_Project_{timestamp}.sky"

    # Ensure the output directory exists on the host system
    output_dir = os.path.join(raw_file_folder, "peptide_qc_results")
    os.makedirs(output_dir, exist_ok=True)

    # Define the SkylineCmd command to run inside the container
    skyline_cmd = [
        "wine", "SkylineCmd", "--timestamp",
        "--dir=/data",
        "--in=/data/_Skyline-template.sky",
        "--save",
        f"--out={output_sky}",
        "--import-search-file=/data/ssl_output.ssl",
        "--import-search-add-mods",
        "--import-search-include-ambiguous",
        f"--import-fasta=/data/{proteome_file_name}",
        "--keep-empty-proteins",
        "--import-threads=4",
        "--refine-auto-select-peptides",
        "--refine-auto-select-transitions",
        "--refine-auto-select-precursors",
        "--chromatogram-precursors",
        "--chromatogram-file=/data/peptide_qc_results/XICs.tsv",
        "--report-add=/data/_Skyline-report.skyr"
    ]

    # Construct the full Docker command
    docker_cmd = [
        "docker", "run", "-i", "--rm",
        "-v", f"{raw_file_folder}:/data",
        "chambm/pwiz-skyline-i-agree-to-the-vendor-licenses",
        "bash", "-c", " ".join(skyline_cmd)
    ]

    # Run the command
    try:
        print("🚀 Launching SkylineCmd in Docker...")
        subprocess.run(docker_cmd, check=True)
        print("✅ SkylineCmd finished successfully.")
    except subprocess.CalledProcessError as e:
        print("❌ Error while running SkylineCmd inside Docker:")
        print(e)



def create_ssl_data(raw_file_folder, raw_file_name):
    raw_file = os.path.join(raw_file_folder, raw_file_name)
    filtered_hits_csv_file = raw_file.replace(".raw", "_filtered_hits.csv")
    filtered_hits_csv_file_docker_version = f"Z:\\\\data\\\\{raw_file_name}"
    filtered_hits_df = pd.read_csv(filtered_hits_csv_file)
    # Create ssl_df with required columns
    ssl_df = pd.DataFrame({
        "file": filtered_hits_csv_file_docker_version,
        "scan": filtered_hits_df["scan"],
        "charge": filtered_hits_df["charge"],
        "sequence": filtered_hits_df["peptide"]
    })

    # Display top 5 rows and dimensions of ssl_df
    ssl_df_head = ssl_df.head()
    ssl_df_shape = ssl_df.shape

    # Save ssl_df as a .ssl file
    output_path = os.path.join(raw_file_folder, "ssl_output.ssl")
    ssl_df.to_csv(output_path, sep="\t", index=False, header=True)
    print(ssl_df_head)
    print(ssl_df_shape)

def convert_raw_to_mgf(scans_folder, n_cores=4):
    ENDC_TEXT = '\033[0m'
    OKCYAN_TEXT = '\033[96m'
    """
    Converts RAW files to MGF format using ThermoRawFileParser.

    Parameters
    ----------
    scans_folder : str
        Path to the folder containing RAW files.
    n_cores : int, optional
        Number of CPU cores to use for parallel processing, default is 4.
    """
    # Ensure the input folder exists
    if not os.path.isdir(scans_folder):
        raise FileNotFoundError(f"The folder '{scans_folder}' does not exist.")

    # Get the list of .raw files in the folder
    raw_files = [
        scan_f for scan_f in os.listdir(scans_folder) if scan_f.lower().endswith('.raw')
    ]
    print(
        OKCYAN_TEXT + f'\tFound {len(raw_files)} RAW files to be converted.' + ENDC_TEXT
    )

    # Ensure ThermoRawFileParser is available
    home = str(Path.home())
    thermo_parser_dir = f'{home}/inSPIRE_models/ThermoRawFileParser'
    thermo_path = f'{thermo_parser_dir}/ThermoRawFileParser.exe'

    if not os.path.isfile(thermo_path):
        print("ThermoRawFileParser not found. Downloading...")
        download_thermo_raw_file_parser()

    # Platform-specific prefix
    prefix = ''
    if platform.system() != 'Windows':
        prefix = 'mono '

    # Build commands for RAW to MGF conversion
    convert_commands = [
        f'{prefix}{thermo_path} -o={scans_folder} -f=0 -i={scans_folder}/{raw_file} -l 4'
        for raw_file in raw_files if not os.path.exists(
            f'{scans_folder}/{raw_file.replace(".raw", ".mgf")}'
        )
    ]

    # Perform the conversions in parallel
    if convert_commands:
        print(OKCYAN_TEXT + "Starting RAW to MGF conversion..." + ENDC_TEXT)
        with Pool(processes=n_cores) as pool:
            pool.map(os.system, convert_commands)
        print(OKCYAN_TEXT + "Conversion completed." + ENDC_TEXT)
    else:
        print(OKCYAN_TEXT + "No new files to convert." + ENDC_TEXT)

def run_skyline_script(skyline_path, raw_file_folder, raw_file_name, tic_result_file, tic_log_file):
    """
    Runs the Skyline script with the given parameters.

    Parameters:
    - raw_file_folder (str): Path to the raw file location.
    - raw_file_name (str): Name of the raw file.
    - tic_result_file (str): Path to the output TIC result file.
    - tic_log_file (str): Path to the log file.

    Returns:
    - None
    """
    script_command = [
        "python3", "skyline.py",
        "--skyline_path", skyline_path,
        "--raw_file_loc", raw_file_folder,
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

def read_process_msfragger_results(raw_file_folder, input_csv, raw_file_name, quality_filter, q_value_cutoff, engine_score_cutoff):
    command = [
        "python3", "read_process_msfragger_results.py",
        "--raw_file_folder", raw_file_folder,
        "--input_csv_file", input_csv,
        "--raw_file_name", raw_file_name,
        "--quality_filter", quality_filter,
        "--q_value_cutoff", str(q_value_cutoff),
        "--engine_score_cutoff", str(engine_score_cutoff)
    ]
    try:    
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error in processing MSFragger results: {e}")

def convert_raw_to_mzml(raw_file_folder, raw_file_name):
    command = [
        "docker", "run", "-it", "--rm",
        "-v", f"{raw_file_folder}:/data",
        "proteowizard/pwiz-skyline-i-agree-to-the-vendor-licenses",
        "wine", "msconvert", f"/data/{raw_file_name}", "-o", "/data"
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Successfully converted {raw_file_name} to mzML format.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while converting RAW to mzML: {e}")

def plot_ms2_spectra(proteome, raw_file_folder, raw_file_name):
    command = [
        "python3", "plot_ms2_spectra.py",
        "--proteome", proteome,
        "--raw_file_folder", raw_file_folder,
        "--raw_file_name", raw_file_name
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while plotting MS2 spectra: {e}")

def plot_ms1_spectra(unimod_file, proteome, raw_file_folder, raw_file_name):
    command = [
        "python3", "plot_ms1_spectra.py",
        "--unimod_file", unimod_file,
        "--proteome", proteome,
        "--raw_file_folder", raw_file_folder,
        "--raw_file_name", raw_file_name
    ]
    try:
        subprocess.run(command)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while plotting MS1 spectra: {e}")


def plot_xic_chromatograms(raw_file_folder):
    xic_file_path = os.path.join(raw_file_folder, "peptide_qc_results", "XICs.tsv")
    if not os.path.exists(xic_file_path):
        print(f"XIC file not found: {xic_file_path}")
        return
    print("Plotting XIC chromatograms...")
    
    # Reload the data with proper column names
    columns = [
        "FileName", "PeptideModifiedSequence", "PrecursorCharge", "ProductMz",
        "FragmentIon", "ProductCharge", "IsotopeLabelType", "TotalArea", "Times", "Intensities"
    ]

    # Load and parse
    data = pd.read_csv(xic_file_path, sep="\t", names=columns, skiprows=1)
    data['Times'] = [list(map(float, str(x).split(','))) for x in data['Times']]
    data['Intensities'] = [list(map(float, str(x).split(','))) for x in data['Intensities']]

    # Group by peptide, charge, and mz
    grouped = data.groupby(["PeptideModifiedSequence", "PrecursorCharge", "ProductMz"])

    # Get unique peptides
    unique_peptides = data["PeptideModifiedSequence"].unique()

    # Create a PDF to store all plots
    pdf_path = os.path.join(raw_file_folder, "XIC_chromatograms.pdf")
    with PdfPages(pdf_path) as pdf:
        for peptide in unique_peptides:
            peptide_data = grouped.filter(lambda x: x["PeptideModifiedSequence"].iloc[0] == peptide)

            plt.figure(figsize=(10, 6))
            for (pep, charge, mz), group in peptide_data.groupby(["PeptideModifiedSequence", "PrecursorCharge", "ProductMz"]):
                times = np.concatenate(group["Times"].values)
                intensities = np.concatenate(group["Intensities"].values)
                plt.plot(times, intensities, label=f"z={charge}, m/z={mz:.2f}")

            plt.title(f"XICs for Peptide: {peptide}", fontsize=14)
            plt.xlabel("Retention Time (min)", fontsize=12)
            plt.ylabel("Intensity", fontsize=12)
            plt.legend(fontsize=10, loc='upper right')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(pdf_path)
    print("XIC chromatograms plotted and saved to PDF.")


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
    skyline_path = config.get("skyline_path")
    input_csv = config.get("input_csv_file")
    raw_file_name = config.get("raw_file_name")
    raw_file_type = config.get("raw_file_type")
    raw_file_folder = config.get("raw_file_folder")
    unimod_file = config.get("unimod_file")
   
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
    output_folder = msfragger_inputs.get("output_folder")
    proteome = msfragger_inputs.get("proteome")
    fragger_path = msfragger_inputs.get("fragger_path")
    contams_db = msfragger_inputs.get("contams_db")


    xic_plotting = config.get("XIC_Plotting", {})
    xic_result_file = xic_plotting.get("xic_result_file")
    rt_cutoff = xic_plotting.get("RT_Cuttoff", 60)
    
    

    # Print extracted values
    print(f"skyline_path: {skyline_path}")
    print(f"Input CSV File: {input_csv}")
    print(f"Raw File Name: {raw_file_name}")
    print(f"Raw File Type: {raw_file_type}")
    print(f"Raw File Folder: {raw_file_folder}")
    print(f"Unimod File: {unimod_file}")
    print(f"TIC Result File: {tic_result_file}")
    print(f"TIC Log File: {tic_log_file}")
    print(f"TIC Plot File: {tic_plot_file}")
    print(f"MSFragger Quality Filter Type: {quality_filter}")
    print(f"Q Value Cutoff: {q_value_cutoff}")
    print(f"Engine Score Cutoff: {engine_score_cutoff}")
    print(f"fragger_params: {fragger_params}")
    print(f"Output Folder: {output_folder}")
    print(f"Proteome: {proteome}")
    print(f"Fragger Path: {fragger_path}")
    print(f"Contaminants Database: {contams_db}")
    print(f"XIC Plotting Result File: {xic_result_file}")
    print(f"XIC Plotting RT Cutoff: {rt_cutoff}")
    print(f"XIC Result File: {xic_result_file}")

    run_skyline_script(skyline_path, raw_file_folder, raw_file_name, tic_result_file, tic_log_file)
    plot_tic_chromatograms(tic_result_file, tic_plot_file, raw_file_name)
    run_msfragger_script(fragger_params,raw_file_folder, output_folder, proteome, fragger_path, contams_db)
    read_process_msfragger_results(raw_file_folder, input_csv, raw_file_name, quality_filter, q_value_cutoff, engine_score_cutoff)
    convert_raw_to_mgf(raw_file_folder)
    convert_raw_to_mzml(raw_file_folder, raw_file_name)
    plot_ms2_spectra(proteome, raw_file_folder, raw_file_name)
    plot_ms1_spectra(unimod_file, proteome, raw_file_folder, raw_file_name)
    create_ssl_data(raw_file_folder, raw_file_name)
    run_skyline_docker_xic(raw_file_folder, proteome)
    plot_xic_chromatograms(raw_file_folder)

if __name__ == "__main__":
    main()