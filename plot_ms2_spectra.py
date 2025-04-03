import os
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pyopenms import MSExperiment, MzMLFile
from Bio import SeqIO
import PyPDF2
import argparse
import multiprocessing, shutil, subprocess, tempfile


# ==============================================================
# PARAMETERS & CONSTANTS (MS1 part)
# ==============================================================

RT_WINDOW = 60.0        # seconds (± window around the MS2 retention time)
TOL_SUBSTRATE = 0.01    # Da tolerance to decide if a peak matches the substrate m/z
TOL_UNIMOD = 0.05       # Da tolerance for matching delta masses to Unimod modifications

WATER_MASS = 18.01056
PROTON_MASS = 1.007825

# Monoisotopic masses for 20 amino acids
MONOISOTOPIC_MASSES = {
    "A": 71.037114, "C": 103.009185, "D": 115.026943, "E": 129.042593,
    "F": 147.068414, "G": 57.021464,  "H": 137.058912, "I": 113.084064,
    "K": 128.094963, "L": 113.084064, "M": 131.040485, "N": 114.042927,
    "P": 97.052764,  "Q": 128.058578, "R": 156.101111, "S": 87.032028,
    "T": 101.047679, "V": 99.068414,  "W": 186.079313, "Y": 163.06332
}

# ==============================================================
# MS1 FUNCTIONS
# ==============================================================

def compute_mw(peptide: str) -> float:
    """Compute the theoretical monoisotopic MW of a peptide (sum of AA masses plus water)."""
    return sum(MONOISOTOPIC_MASSES.get(aa, 0) for aa in peptide) + WATER_MASS

def compute_mz(mw: float, charge: int) -> float:
    """Compute theoretical m/z given MW and charge."""
    return (mw + charge * PROTON_MASS) / charge

def parse_scan_number(native_id: str) -> int:
    """
    Extract the scan number from a native ID string.
    For example: "controllerType=0 controllerNumber=1 scan=2274"
    """
    m = re.search(r"scan=(\d+)", native_id)
    return int(m.group(1)) if m else -1

def load_mzml(mzml_path: str) -> MSExperiment:
    """Load a .mzML file into a pyOpenMS MSExperiment object."""
    exp = MSExperiment()
    MzMLFile().load(mzml_path, exp)
    return exp

def separate_spectra_by_ms_level(exp: MSExperiment):
    """
    Return two lists: ms1_spectra and ms2_spectra.
    Each element is a tuple: (scan_num, rt (seconds), mz_array, intensity_array)
    """
    ms1_spectra = []
    ms2_spectra = []
    for spectrum in exp:
        ms_level = spectrum.getMSLevel()
        rt = spectrum.getRT()  # seconds
        native_id = spectrum.getNativeID()
        scan_num = parse_scan_number(native_id)
        mz_array, intensity_array = spectrum.get_peaks()
        entry = (scan_num, rt, mz_array, intensity_array)
        if ms_level == 1:
            ms1_spectra.append(entry)
        elif ms_level == 2:
            ms2_spectra.append(entry)
    return ms1_spectra, ms2_spectra

def plot_individual_ms1_scans(ms1_spectra, center_rt, ms2_scan, substrate_mz=None,
                              out_dir="ms1_spectra", rt_window=RT_WINDOW, tol_substrate=TOL_SUBSTRATE):
    """
    For the given MS2 scan (with retention time center_rt), find all MS1 scans within ±rt_window.
    Then, only keep those MS1 scans that contain at least one peak within tol_substrate of substrate_mz.
    Divide these scans into pages of 12 subplots per page and save each page as a separate PDF.
    """
    # Collect MS1 scans in the RT window
    ms1_in_window = [
        (scan_num, rt, mz_array, intensity_array)
        for (scan_num, rt, mz_array, intensity_array) in ms1_spectra
        if center_rt - rt_window <= rt <= center_rt + rt_window
    ]
    ms1_in_window.sort(key=lambda x: x[1])  # sort by RT

    # Filter to only include scans that have at least one peak within tol_substrate of substrate_mz
    if substrate_mz is not None:
        ms1_in_window = [scan for scan in ms1_in_window if any(abs(mz - substrate_mz) < tol_substrate for mz in scan[2])]

    if not ms1_in_window:
        print(f"  No MS1 scans with a substrate match (±{tol_substrate} Da) in ±{rt_window}s around RT={center_rt:.2f} for scan {ms2_scan}.")
        return

    # Divide the scans into pages (12 per page)
    n_per_page = 12
    pages = [ms1_in_window[i:i+n_per_page] for i in range(0, len(ms1_in_window), n_per_page)]
    page_num = 1

    for page in pages:
        n_scans = len(page)
        nrows, ncols = 3, 4
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))
        axes = np.ravel(axes)
        fig.suptitle(f"MS1 scans with substrate match (±{tol_substrate} Da) within ±{rt_window}s of MS2 scan {ms2_scan} (Page {page_num})\n(Total matching MS1 scans: {len(ms1_in_window)})", y=0.98)
        for i, (scan_num, rt, mz_array, intensity_array) in enumerate(page):
            ax = axes[i]
            ax.set_title(f"MS1 scan {scan_num} (RT={rt:.2f}s)")
            if len(mz_array) > 0:
                peaks = sorted(zip(mz_array, intensity_array), key=lambda x: x[0])
                sorted_mz = [p[0] for p in peaks]
                sorted_int = [p[1] for p in peaks]
                ax.vlines(sorted_mz, [0], sorted_int, color="black", linewidth=0.8)
                # Highlight substrate peak(s)
                for mz_val, int_val in peaks:
                    if abs(mz_val - substrate_mz) < tol_substrate:
                        ax.vlines(mz_val, 0, int_val, color="red", linewidth=1.2)
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity")
        # Hide unused subplots on the page
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        plt.tight_layout()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ms1_scan_{ms2_scan}_page_{page_num}.pdf")
        plt.savefig(out_path)
        plt.close()
        print(f"  Saved {n_scans} matching MS1 scans to {out_path}")
        page_num += 1

def get_delta_masses_from_ms1_scan(mz_array, intensity_array, substrate_mz, charge, tol_substrate=TOL_SUBSTRATE):
    """
    For one MS1 scan, find the substrate peak (within ±tol_substrate of substrate_mz).
    Then, if the substrate peak is not the most intense, compute delta masses for peaks with higher intensity.
    Otherwise, compute delta masses for peaks with intensity >= 50% of substrate (excluding the substrate).
    Returns a list of delta masses computed as (peak_mz - substrate_mz) * charge.
    """
    substrate_indices = [i for i, mz in enumerate(mz_array) if abs(mz - substrate_mz) < tol_substrate]
    if not substrate_indices:
        return []
    sub_idx = substrate_indices[0]
    sub_int = intensity_array[sub_idx]
    max_idx = np.argmax(intensity_array)
    delta_list = []
    if max_idx != sub_idx:
        for i, (mz, inten) in enumerate(zip(mz_array, intensity_array)):
            if inten > sub_int:
                delta_list.append(charge * (mz - substrate_mz))
    else:
        threshold = 0.5 * sub_int
        for i, (mz, inten) in enumerate(zip(mz_array, intensity_array)):
            if i != sub_idx and inten >= threshold:
                delta_list.append(charge * (mz - substrate_mz))
    return delta_list

def collect_delta_masses_in_window(ms1_spectra, center_rt, substrate_mz, charge,
                                   rt_window=RT_WINDOW, tol_substrate=TOL_SUBSTRATE):
    """
    Gather delta masses from all MS1 scans (in ±rt_window around center_rt).
    """
    deltas = []
    for (scan_num, rt, mz_array, intensity_array) in ms1_spectra:
        if center_rt - rt_window <= rt <= center_rt + rt_window:
            these_deltas = get_delta_masses_from_ms1_scan(mz_array, intensity_array, substrate_mz, charge, tol_substrate)
            deltas.extend(these_deltas)
    return deltas

def match_delta_to_unimod(delta_value, unimod_df, tol=TOL_UNIMOD):
    """
    Return rows in unimod_df where |MonoMass - delta_value| < tol.
    """
    return unimod_df[np.abs(unimod_df["MonoMass"] - delta_value) < tol]













# ==============================================================
# MS2 (MGF) PLOTTING FUNCTIONS & CONSTANTS
# ==============================================================

# Constants for MS2 fragment plotting
PROTON = 1.007276466622
ELECTRON = 0.00054858
H = 1.007825035
C = 12.0
O = 15.99491463
N = 14.003074

N_TERMINUS = H
C_TERMINUS = O + H
CO = C + O
CHO = C + H + O
NH2 = N + (H * 2)
H2O = (H * 2) + O
NH3 = N + (H * 3)

NEUTRAL_LOSSES = [NH3, H2O]
LOSS_NAMES = {'': '', 'NH3': '*', 'H2O': '°'}
LOSS_WEIGHTS = {'': 0, 'NH3': NH3, 'H2O': H2O}
RESIDUE_WEIGHTS = {
    'A': 71.037114, 'R': 156.101111, 'N': 114.042927, 'D': 115.026943,
    'C': 103.009185, 'E': 129.042593, 'Q': 128.058578, 'G': 57.021464,
    'H': 137.058912, 'I': 113.084064, 'L': 113.084064, 'K': 128.094963,
    'M': 131.040485, 'F': 147.068414, 'P': 97.052764,  'S': 87.032028,
    'T': 101.047679, 'W': 186.079313, 'Y': 163.06332,  'V': 99.068414
}

KNOWN_PTM_WEIGHTS = {
    'Deamidated (N)': 0.984016,
    'Deamidated (NQ)': 0.984016,
    'Deamidation (NQ)': 0.984016,
    'Deamidation (N)': 0.984016,
    'Deamidation (Q)': 0.984016,
    'Oxidation (M)': 15.994915,
    'Acetyl (N-term)': 42.010565,
    'Acetylation (N-term)': 42.010565,
    'Acetyl (Protein N-term)': 42.010565,
    'Phospho (Y)': 79.966331,
    'Phospho (ST)': 79.966331,
    'Phospho (STY)': 79.966331,
    'Phosphorylation (STY)': 79.966331,
    'Carbamidomethyl (C)': 57.021464,
    'Carbamidomethylation': 57.021464,
    'unknown': 0.0
}

ION_OFFSET = {
    'a': N_TERMINUS - CHO,
    'b': N_TERMINUS - H,
    'c': N_TERMINUS + NH2,
    'x': C_TERMINUS + CO - H,
    'y': C_TERMINUS + H,
    'z': C_TERMINUS - NH2,
}

def compute_potential_mws(sequence, modifications, reverse, ptm_id_weights):
    sequence_length = len(sequence)
    n_fragments = sequence_length - 1
    mzs = np.empty(n_fragments)

    if modifications and isinstance(modifications, str) and modifications != 'nan' and modifications != 'unknown':
        ptms_list = modifications.split(".")
        mods_list = [int(mod) for mod in ptms_list[1]]
        if reverse:
            ptm_start = int(ptms_list[2])
            ptm_end = int(ptms_list[0])
            mods_list = mods_list[::-1]
        else:
            ptm_start = int(ptms_list[0])
            ptm_end = int(ptms_list[2])
    else:
        mods_list = None
        ptm_start = 0
        ptm_end = 0

    if reverse:
        sequence = sequence[::-1]

    if ptm_start:
        tracking_mw = ptm_id_weights[ptm_start]
    else:
        tracking_mw = 0

    for idx in range(n_fragments):
        tracking_mw += RESIDUE_WEIGHTS[sequence[idx]]
        if mods_list is not None and mods_list[idx]:
            tracking_mw += ptm_id_weights[mods_list[idx]]
        mzs[idx] = tracking_mw

    tracking_mw += RESIDUE_WEIGHTS[sequence[n_fragments]]
    if mods_list is not None and mods_list[n_fragments]:
        tracking_mw += ptm_id_weights[mods_list[n_fragments]]
    if ptm_end:
        tracking_mw += ptm_id_weights[ptm_end]

    return mzs, tracking_mw

def get_ion_masses(sequence, ptm_id_weights, modifications=None):
    sub_seq_mass, total_residue_mass = compute_potential_mws(
        sequence=sequence,
        modifications=modifications,
        reverse=False,
        ptm_id_weights=ptm_id_weights
    )

    rev_sub_seq_mass, _ = compute_potential_mws(
        sequence=sequence,
        modifications=modifications,
        reverse=True,
        ptm_id_weights=ptm_id_weights
    )

    ion_data = {}
    ion_data['b'] = ION_OFFSET['b'] + sub_seq_mass
    ion_data['y'] = ION_OFFSET['y'] + rev_sub_seq_mass

    return ion_data, (total_residue_mass + N_TERMINUS + C_TERMINUS)

def match_mz(base_mass, frag_z, observed_mzs, loss=0.0):
    fragment_mz = (base_mass + (frag_z * PROTON) - loss) / frag_z
    matched_mz_ind = np.argmin(np.abs(observed_mzs - fragment_mz))
    return observed_mzs[matched_mz_ind] - fragment_mz, matched_mz_ind

def extract_spectrum_for_mgf_file(mgf_file_path, mgf_dict):
    """
    Extract spectra from an MGF file. Only spectra with scan numbers
    present in mgf_dict[<basename(mgf_file_path)>] are kept.
    Returns a dictionary: scan number -> {'mz_values', 'intensities', 'charge'}.
    """
    spectra = {}
    scan_pattern = re.compile(r'scan=(\d+)')
    charge_pattern = re.compile(r'(\d+)\+')
    data_pattern = re.compile(r"(\d+\.\d+)\s+(\d+)")
    with open(mgf_file_path, 'r') as file:
        current_scan_number = None
        mz_values = []
        intensities = []
        charge = None
        is_reading_spectrum = False
        for line in file:
            if 'TITLE' in line or 'NativeID' in line:
                match = scan_pattern.search(line)
                if match:
                    if current_scan_number is not None:
                        spectra[current_scan_number] = {
                            'mz_values': mz_values,
                            'intensities': intensities,
                            'charge': charge
                        }
                    current_scan_number = int(match.group(1))
                    # Only continue if this scan is in our list of interest:
                    if current_scan_number not in mgf_dict.get(os.path.basename(mgf_file_path), set()):
                        current_scan_number = None
                        is_reading_spectrum = False
                        continue
                    mz_values = []
                    intensities = []
                    charge = None
                    is_reading_spectrum = True
            if is_reading_spectrum:
                if 'CHARGE' in line:
                    charge_match = charge_pattern.search(line)
                    if charge_match:
                        charge = int(charge_match.group(1))
                data_match = data_pattern.match(line)
                if data_match:
                    mz_values.append(float(data_match.group(1)))
                    intensities.append(int(data_match.group(2)))
                if line.strip() == 'END IONS':
                    if current_scan_number is not None:
                        spectra[current_scan_number] = {
                            'mz_values': mz_values,
                            'intensities': intensities,
                            'charge': charge
                        }
                    current_scan_number = None
                    is_reading_spectrum = False
        if current_scan_number is not None:
            spectra[current_scan_number] = {
                'mz_values': mz_values,
                'intensities': intensities,
                'charge': charge
            }
    return spectra

def plot_matched_ions(peptide, scan_number, mgf_file_path, mz_values, intensities, charge, ax2):
    """
    Plot the MS2 spectrum (in ax2) with an inset (ax1) showing the peptide fragment pattern.
    """
    # Create an inset axis for the peptide fragment diagram
    ax1 = ax2.inset_axes([0, 0.80, 1, 0.20])  # Top 20% for peptide fragment pattern
    modifications = None
    ion_data, _ = get_ion_masses(peptide, KNOWN_PTM_WEIGHTS, modifications)
    proton_charge = PROTON

    # Find matched b-ions (example: only matching within 0.02 Da)
    matched_b_ions = [
        (i + 1, ion_mass, z, loss_name,
         (ion_mass + (z * proton_charge) - loss_weight) / z,
         np.interp((ion_mass + (z * proton_charge) - loss_weight) / z, mz_values, intensities))
        for ion_type, ions in ion_data.items()
        for i, ion_mass in enumerate(ions)
        for z in range(1, 3)
        for loss_name, loss_weight in LOSS_WEIGHTS.items()
        if ion_type == 'b' and abs(match_mz(ion_mass, z, np.array(mz_values), loss=loss_weight)[0]) < 0.02
    ]
    matched_y_ions = [
        (i + 1, ion_mass, z, loss_name,
         (ion_mass + (z * proton_charge) - loss_weight) / z,
         np.interp((ion_mass + (z * proton_charge) - loss_weight) / z, mz_values, intensities))
        for ion_type, ions in ion_data.items()
        for i, ion_mass in enumerate(ions)
        for z in range(1, 3)
        for loss_name, loss_weight in LOSS_WEIGHTS.items()
        if ion_type == 'y' and abs(match_mz(ion_mass, z, np.array(mz_values), loss=loss_weight)[0]) < 0.02
    ]
    mgf_file_name = os.path.basename(mgf_file_path)
    title = f'Peptide: {peptide}\nScan: {scan_number}, File: {mgf_file_name}, Charge: {charge}'
    ax1.set_title(title, fontweight='bold', fontsize=10)
    # Draw the peptide sequence and mark matched ions
    for i, aa in enumerate(peptide):
        ax1.text(i + 1, 1, aa, ha='center', va='center', color='black', fontsize=10)
        if any(b[0] == i + 1 for b in matched_b_ions):
            ax1.plot([i + 1.5, i + 1.5], [0.8, 1], color='blue', linewidth=2)
            ax1.plot([i + 1.3, i + 1.5], [0.8, 0.8], color='blue', linewidth=2)
            ax1.text(i + 1.5, 0.75, f'b{i+1}', ha='center', va='top', color='blue', fontsize=8)
        if any(y[0] == len(peptide) - i for y in matched_y_ions):
            pos_y = i + 0.5
            ax1.plot([pos_y, pos_y], [1, 1.2], color='red', linewidth=2)
            ax1.plot([pos_y, pos_y + 0.2], [1.2, 1.2], color='red', linewidth=2)
            ax1.text(pos_y + 0.2, 1.25, f'y{len(peptide) - i}', ha='center', va='bottom', color='red', fontsize=8)
    ax1.set_xlim(0, len(peptide) + 1)
    ax1.set_ylim(0.7, 1.3)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # Plot the MS2 spectrum
    max_intensity = max(intensities) if intensities else 1
    ax2.set_ylim(0, max_intensity * 1.4)
    ax2.stem(mz_values, intensities, linefmt="k-", markerfmt=" ", basefmt="k-")
    # Mark matched ions on the spectrum
    for b_ion in matched_b_ions:
        ax2.stem([b_ion[4]], [b_ion[5]], linefmt="b-", markerfmt=" ", basefmt="b-")
        ax2.annotate(f'b{b_ion[0]}{LOSS_NAMES[b_ion[3]]}',
                     (b_ion[4], b_ion[5]),
                     textcoords="offset points",
                     xytext=(0, 2 + b_ion[0]),
                     ha='center', color='blue', fontsize=8, clip_on=False)
    for y_ion in matched_y_ions:
        ax2.stem([y_ion[4]], [y_ion[5]], linefmt="r-", markerfmt=" ", basefmt="r-")
        ax2.annotate(f'y{y_ion[0]}{LOSS_NAMES[y_ion[3]]}',
                     (y_ion[4], y_ion[5]),
                     textcoords="offset points",
                     xytext=(0, y_ion[0]),
                     ha='center', color='red', fontsize=8, clip_on=False)
    ax2.set_xlabel('m/z', fontsize=12)
    ax2.set_ylabel('Intensity', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)

def process_spectra(mgf_file_name, mgf_folder, proteasome_db):
    """
    For a given MGF file, extract the scan numbers from the file and then select
    the entries in proteasome_db that have matching scan numbers.
    Returns a list of tuples: (peptide, [scan_numbers], mgf_file_path)
    """
    mgf_file_path = os.path.join(mgf_folder, mgf_file_name)
    mgf_scan_numbers = set()
    with open(mgf_file_path, 'r') as mgf_file:
        for line in mgf_file:
            match = re.search(r'scan=(\d+)', line)
            if match:
                scan_number = int(match.group(1))
                mgf_scan_numbers.add(scan_number)
    matching_entries = proteasome_db[proteasome_db['MSfile'] == mgf_file_name]
    matching_entries = matching_entries[matching_entries['scanNum'].isin(mgf_scan_numbers)]
    if not matching_entries.empty:
        print(f"Processing {mgf_file_name}...")
        grouped = matching_entries.groupby('pepSeq')['scanNum'].apply(list)
        return [(peptide, scan_numbers, mgf_file_path) for peptide, scan_numbers in grouped.items()]
    else:
        print(f"No matching entry found for scans in {mgf_file_name}. Skipping...")
        return []

def convert_spectra_data_to_dict(spectra_data):
    """
    Convert spectra_data (list of tuples) to a dictionary keyed by peptide.
    Also returns a dictionary (mgf_dict) mapping mgf filename -> set(scan_numbers).
    """
    spectra_dict = {}
    mgf_dict = {}
    for peptide, scan_numbers, mgf_file_path in spectra_data:
        if peptide in spectra_dict:
            spectra_dict[peptide].append((scan_numbers, mgf_file_path))
        else:
            spectra_dict[peptide] = [(scan_numbers, mgf_file_path)]
        mgf_filename = os.path.basename(mgf_file_path)
        if mgf_filename in mgf_dict:
            mgf_dict[mgf_filename].update(scan_numbers)
        else:
            mgf_dict[mgf_filename] = set(scan_numbers)
    return spectra_dict, mgf_dict

def plot_single_peptide(peptide, spectra_dict, mgf_data, pdf_filename, temp_folder):
    """
    For a given peptide, generate a PDF (using PdfPages) with MS2 spectra plots.
    Each page contains up to 2 spectra.
    """
    pdf_path = os.path.join(temp_folder, "temp_" + peptide + "_" + pdf_filename)
    with PdfPages(pdf_path) as pdf:
        for scan_numbers, mgf_file_path in spectra_dict[peptide]:
            num_plots = len(scan_numbers)
            num_pages = (num_plots + 1) // 2
            for page in range(num_pages):
                fig, axs = plt.subplots(2, 1, figsize=(16, 10))
                fig.subplots_adjust(hspace=0.8)
                for i in range(2):
                    plot_index = page * 2 + i
                    if plot_index >= num_plots:
                        axs[i].axis('off')
                        continue
                    scan_number = scan_numbers[plot_index]
                    mgf_filename = os.path.basename(mgf_file_path)
                    spectrum = mgf_data[mgf_filename].get(scan_number, None)
                    if spectrum is None:
                        axs[i].axis('off')
                        continue
                    mz_values = spectrum['mz_values']
                    intensities = spectrum['intensities']
                    charge = spectrum['charge']
                    if mz_values and intensities:
                        plot_matched_ions(peptide, scan_number, mgf_file_path, mz_values, intensities, charge, axs[i])
                    else:
                        axs[i].axis('off')
                pdf.savefig(fig)
                plt.close(fig)
    print(f"Spectra plots saved as '{pdf_path}'")

def combine_pdfs_into_single_pdf(input_folder, output_file):
    """
    Combine all PDF files in input_folder (that start with 'temp_') into a single PDF.
    """
    pdf_merger = PyPDF2.PdfMerger()
    pdf_files = [file for file in os.listdir(input_folder) if file.startswith('temp_') and file.lower().endswith('.pdf')]
    pdf_files.sort()
    for pdf_file in pdf_files:
        pdf_merger.append(os.path.join(input_folder, pdf_file))
    with open(output_file, 'wb') as merged_pdf_file:
        pdf_merger.write(merged_pdf_file)
    print(f"Combined {len(pdf_files)} PDFs into {output_file}")

def plot_ms2_spectra(raw_file_folder, filtered_hits_df, mgf_file):
    """
    From the filtered_hits_df (with query peptides) and a given MGF file,
    group by peptide and output MS2 spectra (with fragment annotation) as PDFs.
    """
    mgf_folder = os.path.dirname(mgf_file)
    mgf_file_name = os.path.basename(mgf_file)
    # Create a proteasome_db dataframe from filtered_hits_df:
    proteasome_db = filtered_hits_df.copy()
    proteasome_db.rename(columns={'peptide': 'pepSeq', 'scan': 'scanNum'}, inplace=True)
    proteasome_db['MSfile'] = mgf_file_name

    # Process the MGF file to get spectra data for the scans in proteasome_db:
    spectra_data = process_spectra(mgf_file_name, mgf_folder, proteasome_db)
    if not spectra_data:
        print("No matching MS2 scans found in the MGF file.")
        return

    # Convert spectra_data into dictionaries:
    spectra_dict, mgf_dict = convert_spectra_data_to_dict(spectra_data)

    # Extract spectra from the MGF file using mgf_dict:
    mgf_data = {}
    mgf_data[mgf_file_name] = extract_spectrum_for_mgf_file(mgf_file, mgf_dict)

    # Create a temporary folder for individual peptide PDFs:
    temp_folder = os.path.join(raw_file_folder, "ms2_spectra")
    os.makedirs(temp_folder, exist_ok=True)
    pdf_filename = "ms2_spectra.pdf"
    for peptide in spectra_dict:
        plot_single_peptide(peptide, spectra_dict, mgf_data, pdf_filename, temp_folder)
    # Optionally, combine all peptide PDFs into one file:
    combined_pdf = os.path.join(raw_file_folder, "ms2_spectra_combined.pdf")
    combine_pdfs_into_single_pdf(temp_folder, combined_pdf)
    print("MS2 spectra analysis complete.")
    
    
    



# ==============================================================
# MAIN WORKFLOW
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run plot ms2 spectra with provided inputs.")
    parser.add_argument("--proteome", required=True, help="Path to proteome FASTA file")
    parser.add_argument("--raw_file_folder", required=True, help="Folder containing raw or mzML files")
    parser.add_argument("--raw_file_name", required=True, help="Name of the raw file.")
    args = parser.parse_args()
    os.makedirs(args.raw_file_folder, exist_ok=True)
    
    """
    Workflow:
      - Load query peptides from FASTA.
      - Load filtered MSFragger hits and keep those matching query peptides.
      - Load the mzML file and separate MS1 and MS2 spectra.
      - For each MSFragger hit:
            * Compute theoretical MW and m/z.
            * Use the MS2 scan's RT to find all MS1 scans within ±RT_WINDOW.
            * Plot matching MS1 scans (with substrate match highlighted).
            * Compute delta masses and match them to Unimod modifications.
      - Then, using an MGF file, plot the MS2 spectra with peptide fragment patterns.
    """
    # ----- Part 1: MS1 Analysis -----
    #Load query peptides from FASTA
    query_fasta = args.proteome
    query_peptides = [str(rec.seq) for rec in SeqIO.parse(query_fasta, "fasta")]
    print(f"Loaded {len(query_peptides)} query peptides from FASTA.")

    # # Load filtered MSFragger hits (dummy example DataFrame shown here)
    pep_xml_file = f"{args.raw_file_folder}/{args.raw_file_name.replace('.raw', '')}.pepXML"

    filtered_hits_csv_file = pep_xml_file.replace(".pepXML", "_filtered_hits.csv")

    filtered_hits_df = pd.read_csv(filtered_hits_csv_file)
    # Only consider hits whose peptide is in the query FASTA
    filtered_hits_df = filtered_hits_df[filtered_hits_df["peptide"].isin(query_peptides)]
    print(f"{len(filtered_hits_df)} MSFragger hits match the query peptides.")

    # # File paths for mzML and Unimod:
    mzml_file = pep_xml_file.replace(".pepXML",".mzML")
   
    # Load mzML and separate spectra:
    print(f"Loading mzML file: {mzml_file}")
    exp = load_mzml(mzml_file)
    ms1_spectra, ms2_spectra = separate_spectra_by_ms_level(exp)
    print(f"Found {len(ms1_spectra)} MS1 spectra and {len(ms2_spectra)} MS2 spectra.")

    # Create a dictionary for MS2 scans: scan number -> (rt, mz_array, intensity_array)
    ms2_dict = {scan_num: (rt, mz_arr, int_arr) for (scan_num, rt, mz_arr, int_arr) in ms2_spectra}



    results = {}
    for idx, row in filtered_hits_df.iterrows():
        ms2_scan = int(row["scan"])
        pep = row["peptide"]
        charge = int(row["charge"])
        theo_mw = compute_mw(pep)
        theo_mz = compute_mz(theo_mw, charge)
        print(f"\nRow {idx}: MS2 scan {ms2_scan}, peptide: {pep}, charge: {charge}, MW: {theo_mw:.2f}, m/z: {theo_mz:.2f}")

        if ms2_scan not in ms2_dict:
            print(f"  MS2 scan {ms2_scan} not found in mzML; skipping.")
            continue
        ms2_rt, ms2_mz_arr, ms2_int_arr = ms2_dict[ms2_scan]



    # ----- Part 2: MS2 (MGF) Analysis -----
    mgf_file = pep_xml_file.replace(".pepXML", ".mgf")  # adjust the path as needed
    print("\nStarting MS2 spectra analysis from MGF file...")
    plot_ms2_spectra(args.raw_file_folder, filtered_hits_df, mgf_file)
    print("All analyses complete.")

if __name__ == "__main__":
    main()

