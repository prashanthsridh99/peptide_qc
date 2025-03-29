import argparse

""" Functions for reading in MSFragger search results.
"""
import multiprocessing

import pandas as pd
import polars as pl
from pyteomics import pepxml


ACCESSION_KEY = "proteins"
CHARGE_KEY = "charge"
DELTA_SCORE_KEY = "deltaScore"
ENGINE_SCORE_KEY = "engineScore"
KNOWN_PTM_WEIGHTS = {
    "Deamidated (N)": 0.984016,
    "Deamidated (NQ)": 0.984016,
    "Deamidation (NQ)": 0.984016,
    "Deamidation (N)": 0.984016,
    "Deamidation (Q)": 0.984016,
    "Oxidation (M)": 15.994915,
    "Acetyl (N-term)": 42.010565,
    "Acetylation (N-term)": 42.010565,
    "Acetyl (Protein N-term)": 42.010565,
    "Phospho (Y)": 79.966331,
    "Phospho (ST)": 79.966331,
    "Phospho (STY)": 79.966331,
    "Phosphorylation (STY)": 79.966331,
    "Carbamidomethyl (C)": 57.021464,
    "Carbamidomethylation": 57.021464,
    "unknown": 0.0,
}
LABEL_KEY = "Label"
MASS_DIFF_KEY = "massDiff"
PEPTIDE_KEY = "peptide"
PTM_ID_KEY = "Identifier"
PTM_IS_VAR_KEY = "isVar"
PTM_NAME_KEY = "Name"
PTM_SEQ_KEY = "ptm_seq"
PTM_WEIGHT_KEY = "Delta"
RT_KEY = "retentionTime"
SCAN_KEY = "scan"
SEQ_LEN_KEY = "sequenceLength"
SOURCE_KEY = "source"

def filter_for_prosit(search_df):
    """ Function to filter sequences not suitable for Prosit input (polars DataFrame).

    Parameters
    ----------
    search_df : pl.DataFrame
        A DataFrame of search results from an ms search engine.

    Returns
    -------
    search_df : pl.DataFrame
        The input DataFrame with sequences not suitable for Prosit input removed.
    """
    search_df = search_df.filter(
        pl.col(PEPTIDE_KEY).is_not_null() & (pl.col(CHARGE_KEY).is_not_null())
    )
    search_df = search_df.filter(
        pl.col(PEPTIDE_KEY).map_elements(
            lambda x: isinstance(x, str) and 'U' not in x and 6 < len(x) < 31,
            skip_nulls=False
        )
    )
    search_df = search_df.filter(
        pl.col(CHARGE_KEY).lt(7)
    )

    return search_df

# Define the relevant column names from MSFragger search results.
MSF_ACCESSION_KEY = 'Proteins'
MSF_DELTA_KEY = 'delta_hyperscore'
MSF_PPM_ERR_KEY = 'abs_ppm'
MSF_PEPTIDE_KEY = 'Peptide'
MSF_MASS_KEY = 'ExpMass'
MSF_LABEL_KEY = 'Label'
MSF_PEP_LEN_KEY = 'peptide_length'
MSF_SCORE_KEY = 'hyperscore'
MSF_RT_KEY = 'retentiontime'
MSF_LOG10E_KEY = 'log10_evalue'
MSF_NTT_KEY = 'ntt'
MSF_NMC_KEY = 'nmc'

MSF_IGNORE_COLS = [
    MSF_LOG10E_KEY,
    MSF_NMC_KEY,
    MSF_NTT_KEY,
]

ID_NUMBERS = {
    'Deamidation (N)': 8,
    'Deamidation (Q)': 7,
    'Phospho (S)': 6,
    'Phospho (T)': 5,
    'Phospho (Y)': 4,
    'Acetyl (N-term)': 3,
    'Oxidation (M)': 2,
    'Carbamidomethyl (C)': 1,
}

MSF_MAPPING_PEP_XML = {
    '115': 'Deamidation (N)',
    '129': 'Deamidation (Q)',
    '167': 'Phospho (S)',
    '181': 'Phospho (T)',
    '243': 'Phospho (Y)',
    '43': 'Acetyl (N-term)',
    '147': 'Oxidation (M)',
    '160': 'Carbamidomethyl (C)',
}

MSF_MAPPING_PEP_XML_REV = {
    item: key for key, item in MSF_MAPPING_PEP_XML.items()
}

def _extract_psm_id_data(spec_id):
    """ Function to extract data from the PSM ID column to standard inSPIRE format.
    """
    results = {}
    spec_id_data = spec_id.split('.')
    results[SOURCE_KEY] = spec_id_data[0]
    results[SCAN_KEY] = int(spec_id_data[1])
    results[f'alt_{CHARGE_KEY}'] = int(spec_id_data[-1].split('_')[0])
    return results

def _extract_mod_weight(code):
    if code['mass'] is not None:
        return round(code['mass'])
    return None

def _get_msf_mods_pepxml(msf_df, fixed_modifications):
    """ Function to get the modifications based on the MS fragger pepXML columns.

    Parameters
    ----------
    msf_df : pl.DataFrame
        MSFragger results read in from pepXML format.
    fixed_modifications : list of str or None
        A list of the fixed modification used in the experiment.

    Returns
    -------
    mod_df : pd.DataFrame
        Small DataFrame of the modifications found in the data.
    msf_name_to_id : dict
        A dictionary mapping names of modifications in MSFragger to inSPIRE IDs.
    """
    try:
        msf_mods = sorted(
            msf_df['modifications'].explode().apply(
                _extract_mod_weight,
                skip_nulls=False,
            ).drop_nulls().unique().to_list()
        )
        msf_mods = [str(int(x)) for x in msf_mods if int(x) != -1]
    except:
        msf_mods = []

    mod_names = [MSF_MAPPING_PEP_XML[msf_mod] for msf_mod in msf_mods]
    mod_weights = [KNOWN_PTM_WEIGHTS[mod] for mod in mod_names]

    mod_df = pd.DataFrame({
        PTM_NAME_KEY: mod_names,
        PTM_WEIGHT_KEY: mod_weights,
        PTM_IS_VAR_KEY: [
            fixed_modifications is None or not nm in fixed_modifications for nm in mod_names
        ],
        'msfMod': msf_mods,
    })

    mod_df = mod_df.sort_values(by=PTM_NAME_KEY)
    mod_df = mod_df.reset_index(drop=True)
    mod_df[PTM_ID_KEY] = mod_df[PTM_NAME_KEY].apply(ID_NUMBERS.get)
    msf_name_to_id = dict(zip(
        mod_df['msfMod'].tolist(),
        [str(x) for x in mod_df[PTM_ID_KEY].tolist()],
    ))
    return mod_df, msf_name_to_id

def separate_msf_ptms(mod_pep, ms_frag_mappings, mod_pep_key):
    """ Function to separate MSFragger reported PTMs.

    Parameters
    ----------
    mod_pep : str
        Modified peptide string
    ms_frag_mappings : dict
        Dictionary mapping PTM weights to their inSPIRE ID.

    Returns
    -------
    str
        Formatted peptide sequence with modifications.
    """
    try:
        if mod_pep is None:
            return None

        peptide = mod_pep
        start_mod = '0'

        if peptide[0:5] == 'n[43]':
            start_mod = '3'
            peptide = peptide[5:]

        if '[' in peptide:
            pep_seq = ''
            ptm_seq = ''
            while peptide:
                if peptide[0] == '[':
                    # unexpected format
                    return None

                if len(peptide) == 1:
                    pep_seq += peptide[0]
                    ptm_seq += '0'
                    break

                if peptide[1] != '[':
                    pep_seq += peptide[0]
                    ptm_seq += '0'
                    peptide = peptide[1:]
                else:
                    pep_seq += peptide[0]
                    peptide = peptide[1:]  # skip current character

                    end_mod_str = peptide.index(']') + 1
                    mod_str = peptide[1:end_mod_str - 1]
                    mod_code = ms_frag_mappings.get(mod_str)

                    if mod_code is None:
                        return None  # unknown modification

                    ptm_seq += mod_code
                    peptide = peptide[end_mod_str:]

            return f'{start_mod}.{ptm_seq}.0'

        return None

    except Exception:
        return None


def flatten_protein_data(proteins):
    """ Function to extract protein accession data from pepXML dictionary.

    Parameters
    ----------
    proteins : list
        List of protein dictionaries from pepXML.

    Returns
    -------
    dict
        Dictionary with label and protein accession data.
    """
    label = -1
    results = {}
    new_proteins = []
    for protein in proteins:
        if not protein['protein'].startswith('rev'):
            label = 1
        new_proteins.append(protein['protein'])
    results[LABEL_KEY] = label
    results[ACCESSION_KEY] = ','.join(new_proteins)
    return results

def read_single_ms_fragger_data(df_loc, fixed_modifications):
    """ Function to read in MS Fragger search results from a single file.

    Parameters
    ----------
    df_loc : str
        A location of MS Fragger search results.
    fixed_modifications : list
        List of fixed modifications used in the experiment.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs found in the data.
    """
    all_psms = []
    print(df_loc)
    with pepxml.read(df_loc) as psms:
        for psm in psms:
            if len(psm['search_hit']) > 1:
                rank_2_score = psm['search_hit'][1]['search_score']['hyperscore']
            else:
                rank_2_score = 0

            for pep_psm in psm['search_hit']:
                pep_psm[CHARGE_KEY] = int(psm['assumed_charge'])
                pep_psm['SpecId'] = psm['spectrum']
                pep_psm[MSF_RT_KEY] = psm['retention_time_sec']
                pep_psm['hyperscore'] = pep_psm['search_score']['hyperscore']
                pep_psm[MSF_DELTA_KEY] = pep_psm['hyperscore'] - rank_2_score
                pep_psm['q-value'] = pep_psm['search_score']['expect']
                del pep_psm['search_score']
                all_psms.append(pep_psm)
    
    # Convert to Polars DataFrame for intermediate processing
    msf_df = pl.DataFrame(all_psms)
    msf_df = msf_df.with_columns(
        pl.col('SpecId').map_elements(_extract_psm_id_data).alias('results')
    ).unnest('results')

    if CHARGE_KEY not in msf_df.columns:
        msf_df = msf_df.rename({f'alt_{CHARGE_KEY}': CHARGE_KEY})

    # Separate PTMs.
    var_mod_df, msf_name_to_id = _get_msf_mods_pepxml(msf_df, fixed_modifications)
    msf_df = msf_df.with_columns(
        pl.col('modified_peptide').map_elements(
            lambda mod_pep: separate_msf_ptms(mod_pep, msf_name_to_id, 'modified_peptide'),
            skip_nulls=False
        ).alias(PTM_SEQ_KEY)
    )

    msf_df = msf_df.with_columns(
        pl.col('proteins').map_elements(flatten_protein_data).alias('results')
    )
    msf_df = msf_df.drop('proteins')
    msf_df = msf_df.unnest('results')
    msf_df = msf_df.filter(pl.col('Label').eq(1))

    msf_df = msf_df.with_columns(
        pl.col(PEPTIDE_KEY).map_elements(len).alias(MSF_PEP_LEN_KEY),
    )

    # Rename to match inSPIRE naming scheme.
    msf_df = msf_df.rename({
        MSF_PEP_LEN_KEY: SEQ_LEN_KEY,
        'massdiff': MASS_DIFF_KEY,
        MSF_RT_KEY: RT_KEY,
        MSF_SCORE_KEY: ENGINE_SCORE_KEY,
        MSF_DELTA_KEY: DELTA_SCORE_KEY,
    })

    msf_df = msf_df.with_columns(
        pl.lit(0).alias('fromChimera'),
        pl.lit(0).alias('missedCleavages'),
        pl.col(MASS_DIFF_KEY).map_elements(abs).alias(MASS_DIFF_KEY),
    )

    # Filter for Prosit and add feature columns not present.
    msf_df = filter_for_prosit(msf_df)

    msf_df = msf_df.select(
        'source',
        'scan',
        'peptide',
        'Label',
        'proteins',
        'ptm_seq',
        'sequenceLength',
        'missedCleavages',
        'charge',
        'massDiff',
        'retentionTime',
        'engineScore',
        'deltaScore',
        'q-value'
    )

    # Convert Polars DataFrame to Pandas DataFrame before returning
    return msf_df.to_pandas(), var_mod_df

def read_ms_fragger_data(ms_fragger_data, fixed_modifications, n_cores=8):
    """ Function to read in MS Fragger search results from one or more files.

    Parameters
    ----------
    ms_fragger_data : str or list of str
        A single location of MS Fragger search results or a list of locations.
    fixed_modifications : list
        List of fixed modifications used in the experiment.
    n_cores : int
        Number of cores to use for multiprocessing.

    Returns
    -------
    hits_df : pd.DataFrame
        A DataFrame of all search results properly formatted for inSPIRE.
    mods_df : pd.DataFrame
        A small DataFrame detailing the PTMs found in the data.
    """
    if isinstance(ms_fragger_data, list):
        func_args = [(msf_file, fixed_modifications) for msf_file in ms_fragger_data]

        with multiprocessing.get_context('spawn').Pool(processes=n_cores) as pool:
            results = pool.starmap(read_single_ms_fragger_data, func_args)
        mods_dfs = [res[1] for res in results]

        # Combine DataFrames and validate that same PTMs are present.
        hits_df = pd.concat([res[0] for res in results], ignore_index=True)
        for i in range(len(mods_dfs)-1):
            if mods_dfs[i].shape[0] and mods_dfs[i+1].shape[0]:
                print(mods_dfs[i])
                print(mods_dfs[i+1])
                # assert mods_dfs[i].equals(mods_dfs[i+1])

        return hits_df, mods_dfs[0]

    return read_single_ms_fragger_data(ms_fragger_data, fixed_modifications)

def main():
    parser = argparse.ArgumentParser(description="Process MSFragger results.")
    parser.add_argument("--raw_file_folder", required=True, help="Path to the output folder.")
    parser.add_argument("--input_csv_file", required=True, help="Path to the input CSV file.")
    parser.add_argument("--raw_file_name", required=True, help="Name of the raw file.")
    parser.add_argument("--quality_filter", default="q_value", help="Quality filter to apply (default: q_value).")
    parser.add_argument("--q_value_cutoff", type=float, default=0.01, help="Q-value cutoff (default: 0.01).")
    parser.add_argument("--engine_score_cutoff", type=float, default=10, help="Engine score cutoff (default: 10).")

    args = parser.parse_args()

    raw_file_folder = args.raw_file_folder
    input_csv = args.input_csv_file
    raw_file_name = args.raw_file_name
    quality_filter = args.quality_filter
    q_value_cutoff = args.q_value_cutoff if quality_filter == "q_value" else None
    engine_score_cutoff = args.engine_score_cutoff if quality_filter == "engine_score" else None
    
    pep_xml_file = f"{raw_file_folder}/{raw_file_name.replace('.raw', '')}.pepXML"

    print(f"Output folder: {raw_file_folder}")
    print(f"Input CSV file: {input_csv}")
    print(f"Raw file name: {raw_file_name}")
    print(f"Quality filter: {quality_filter}")
    print(f"Q-value cutoff: {q_value_cutoff}")
    print(f"Engine score cutoff: {engine_score_cutoff}")
    print(f"PEP XML file: {pep_xml_file}")
    
    print("Reading MSFragger data...")
    hits_df, mods_df = read_ms_fragger_data(pep_xml_file, fixed_modifications=None)
    # Save hits_df and mods_df as CSV files
    hits_csv_file = pep_xml_file.replace(".pepXML", "_hits.csv")
    mods_csv_file = pep_xml_file.replace(".pepXML", "_mods.csv")

    hits_df.to_csv(hits_csv_file, index=False)
    mods_df.to_csv(mods_csv_file, index=False)

    print(f"Saved hits DataFrame to: {hits_csv_file}")
    print(f"Saved modifications DataFrame to: {mods_csv_file}")
    
    
    
if __name__ == "__main__":
    main()