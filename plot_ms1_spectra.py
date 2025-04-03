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
import multiprocessing, shutil, subprocess, tempfile
import argparse
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

def combine_pdfs_into_single_pdf(input_folder):
    """
    Combine all PDF files in input_folder (of the form ms1_scan_A_pageB.pdf) into separate combined PDFs for each A.
    """
    pdf_files = [file for file in os.listdir(input_folder) if file.startswith('ms1_scan_') and file.lower().endswith('.pdf')]
    pdf_files.sort()

    # Group files by scan number (A in ms1_scan_A_pageB.pdf)
    grouped_files = {}
    for pdf_file in pdf_files:
        match = re.match(r"ms1_scan_(\d+)_page_\d+\.pdf", pdf_file)
        if match:
            scan_num = match.group(1)
            grouped_files.setdefault(scan_num, []).append(pdf_file)

    # Combine PDFs for each scan number
    for scan_num, files in grouped_files.items():
        pdf_merger = PyPDF2.PdfMerger()
        for pdf_file in files:
            pdf_path = os.path.join(input_folder, pdf_file)
            try:
                with open(pdf_path, 'rb') as f:
                    PyPDF2.PdfReader(f)  # Validate the PDF file
                pdf_merger.append(pdf_path)
            except PyPDF2.errors.PdfReadError:
                print(f"Skipping invalid or corrupted PDF file: {pdf_file}")
        output_file = os.path.join(input_folder, f"combined_ms1_scan_{scan_num}.pdf")
        with open(output_file, 'wb') as merged_pdf_file:
            pdf_merger.write(merged_pdf_file)
        print(f"Combined {len(files)} PDFs for scan {scan_num} into {output_file}")

def process_page(page, page_num, out_dir, ms2_scan, tol_substrate, rt_window, ms1_in_window, substrate_mz):
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


def plot_individual_ms1_scans(raw_file_folder, ms1_spectra, center_rt, ms2_scan, substrate_mz=None, rt_window=RT_WINDOW, tol_substrate=TOL_SUBSTRATE):
    """
    For the given MS2 scan (with retention time center_rt), find all MS1 scans within ±rt_window.
    Then, only keep those MS1 scans that contain at least one peak within tol_substrate of substrate_mz.
    Divide these scans into pages of 12 subplots per page and save each page as a separate PDF.
    """
    out_dir=os.path.join(raw_file_folder,"ms1_spectra")
    
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
    # Use multiprocessing to process pages in parallel with a maximum of 10 processes
    os.makedirs(out_dir, exist_ok=True)
    with multiprocessing.Pool(processes=10) as pool:
        pool.starmap(process_page, [(page, page_num, out_dir, ms2_scan, tol_substrate, rt_window, ms1_in_window, substrate_mz) for page_num, page in enumerate(pages, start=1)])
    combine_pdfs_into_single_pdf(out_dir)
    print("MS1 spectra analysis complete.")

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
# MAIN WORKFLOW
# ==============================================================

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run plot ms2 spectra with provided inputs.")
    parser.add_argument("--proteome", required=True, help="Path to proteome FASTA file")
    parser.add_argument("--raw_file_folder", required=True, help="Folder containing raw or mzML files")
    parser.add_argument("--raw_file_name", required=True, help="Name of the raw file.")
    parser.add_argument("--unimod_file", required=True, help="Path to Unimod CSV file")
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
    query_fasta =args.proteome
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
    unimod_file = args.unimod_file

    # Load mzML and separate spectra:
    print(f"Loading mzML file: {mzml_file}")
    exp = load_mzml(mzml_file)
    ms1_spectra, ms2_spectra = separate_spectra_by_ms_level(exp)
    print(f"Found {len(ms1_spectra)} MS1 spectra and {len(ms2_spectra)} MS2 spectra.")

    # Create a dictionary for MS2 scans: scan number -> (rt, mz_array, intensity_array)
    ms2_dict = {scan_num: (rt, mz_arr, int_arr) for (scan_num, rt, mz_arr, int_arr) in ms2_spectra}

    # Load Unimod CSV (must contain a column "MonoMass"):
    unimod_df = pd.read_csv(unimod_file)

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

        # Plot MS1 scans (with substrate match highlighted)
        plot_individual_ms1_scans(
            args.raw_file_folder,
            ms1_spectra, center_rt=ms2_rt, ms2_scan=ms2_scan,
            substrate_mz=theo_mz,
            rt_window=RT_WINDOW, tol_substrate=TOL_SUBSTRATE
        )

        # Compute delta masses from MS1 scans
        deltas = collect_delta_masses_in_window(
            ms1_spectra, center_rt=ms2_rt, substrate_mz=theo_mz, charge=charge,
            rt_window=RT_WINDOW, tol_substrate=TOL_SUBSTRATE
        )
        # Filter delta masses to plausible values:
        deltas = [d for d in deltas if -400 <= d <= 400 and abs(d) >= 0.9]
        deltas = sorted(set(round(d, 2) for d in deltas))
        #print(f"  Delta masses: {deltas}")

        # Match each delta to Unimod modifications:
        matched_ptms = []
        for d in deltas:
            hits = match_delta_to_unimod(d, unimod_df, tol=TOL_UNIMOD)
            if not hits.empty:
                matched_ptms.append((d, hits))
                print(f"    Delta {d} matches:")
                print(hits[["Name", "Description", "MonoMass"]])
            #else:
            #    print(f"    Delta {d} has no Unimod match within ±{TOL_UNIMOD} Da.")
        results[ms2_scan] = {"deltas": deltas, "ptm_matches": matched_ptms}

    print("\nMS1 analysis complete.")

    # ----- Part 2: MS2 (MGF) Analysis -----
    # mgf_file = "/data/Prashanth/raw_file_substrate/HRoetschke_020124_020124_Lu_TSN200_40uM.mgf"  # adjust the path as needed
    # print("\nStarting MS2 spectra analysis from MGF file...")
    # plot_ms2_spectra(filtered_hits_df, mgf_file)
    # print("All analyses complete.")

if __name__ == "__main__":
    main()