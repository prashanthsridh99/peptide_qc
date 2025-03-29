import argparse
import os
import shutil


# Functions for executing MSFragger Searches

def get_proteins(protein_file, all_ids, all_proteins):
    """ Function to retrieve protein IDs and names from a file. """
    print(f"Reading proteins from {protein_file}")
    with open(protein_file, mode='r', encoding='UTF-8') as fasta_f:
        protein = ''
        while (line := fasta_f.readline()):
            if line.startswith('>'):
                all_ids.append(line[1:].strip('\n'))
                if protein:
                    all_proteins.append(protein)
                    protein = ''
            else:
                protein += line.strip('\n')
        all_proteins.append(protein)
    print(f"Proteins read: {len(all_ids)}")
    return all_ids, all_proteins


def write_search_proteome(config):
    """ Write new proteome file with decoys. """
    all_ids = []
    all_proteins = []

    # Read proteins from the proteome and contaminants databases
    all_ids, all_proteins = get_proteins(config.proteome, all_ids, all_proteins)
    all_ids, all_proteins = get_proteins(config.contams_db, all_ids, all_proteins)

    print(f"Proteome file: {config.proteome}")
    # Copy original proteome file to the output folder
    shutil.copyfile(config.proteome, f'{config.output_folder}/search_proteome.fasta')
    with open(f'{config.output_folder}/search_proteome.fasta', mode='a+', encoding='UTF-8') as search_file:
        with open(config.contams_db, mode='r', encoding='UTF-8') as cont_file:
            search_file.write(cont_file.read())

    print(f"Proteome file copied to {config.output_folder}/search_proteome.fasta")
    # Append decoys to the copied proteome file
    with open(f'{config.output_folder}/search_proteome.fasta', mode='a', encoding='UTF-8') as out_file:
        for prot_id, prot_seq in zip(all_ids, all_proteins):
            rev_sequence = prot_seq[::-1]
            out_file.write(f'>rev_{prot_id}\n{rev_sequence}\n')


def write_fragger_params(config):
    """ Function to write MSFragger parameters. """
    ms2_units = 0 if config.mz_units == 'Da' else 1

    print(f"MS2 units: {ms2_units} ({config.mz_units})")
    with open(config.fragger_params, mode='r', encoding='UTF-8') as frag_template_file:
        fragger_params = frag_template_file.read().format(
            search_database=f'{config.output_folder}/search_proteome.fasta',
            ncpus=config.n_cores,
            precursor_tolerance=config.ms1_accuracy,
            fragament_tolerance=config.mz_accuracy,
            fragment_units=ms2_units,
            top_n_candidates=10,
        )
    print(f"MSFragger parameters:\n{fragger_params}")
    with open(f'{config.output_folder}/fragger.params', mode='w', encoding='UTF-8') as params_file:
        params_file.write(fragger_params)


def clean_up_fragger(config):
    """ Function to clean up files after MSFragger execution. """
    with open(f'{config.output_folder}/fragger_searches.txt', mode='w', encoding='UTF-8') as frag_out:
        for s_file in os.listdir(config.scans_folder):
            if s_file.endswith('.pepXML'):
                frag_out.write(f'{config.scans_folder}/{s_file}\n')

    # Remove intermediate mzML files
    mzml_files = [f'{config.scans_folder}/{scan_f}' for scan_f in os.listdir(config.scans_folder)
                  if scan_f.lower().endswith('mzML')]
    for mzml_file in mzml_files:
        os.remove(mzml_file)


def execute_msfragger(config):
    """ Function to run an MSFragger search with inSPIRE default settings. """
    raw_files_for_fragger = ' '.join(
        [f'{config.scans_folder}/{scan_f}' for scan_f in os.listdir(config.scans_folder) if scan_f.lower().endswith('.raw')]
    )

    write_search_proteome(config)
    write_fragger_params(config)

    print("Running MSFragger search...")
    os.system(
        f'java -Xmx{config.fragger_memory}g -jar {config.fragger_path} ' +
        f'{config.output_folder}/fragger.params {raw_files_for_fragger} > ' +
        f'{config.output_folder}/fragger.log'
    )

    print("MSFragger search completed. Cleaning up files...")
    clean_up_fragger(config)

# Helper class to convert dictionary to object
class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

def main():
    parser = argparse.ArgumentParser(description="Wrapper to run MSFragger with provided inputs.")
    parser.add_argument("--fragger_params", required=True, help="Path to fragger params file")
    parser.add_argument("--raw_file_folder", required=True, help="Folder containing raw or mzML files")
    parser.add_argument("--output_folder", required=True, help="Output directory")
    parser.add_argument("--proteome", required=True, help="Path to proteome FASTA file")
    parser.add_argument("--fragger_path", required=True, help="Path to MSFragger .jar file")
    parser.add_argument("--contams_db", required=True, help="Path to contaminants FASTA file")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    # Configuration with all required parameters
    config = Config({
        'fragger_params': args.fragger_params,
        'scans_folder': args.raw_file_folder,
        'output_folder': args.output_folder,
        'proteome': args.proteome,
        'fragger_db_splits': 1,
        'fragger_memory': 8,
        'fragger_path': args.fragger_path,
        'contams_db': args.contams_db,
        # Added parameters
        'mz_units': 'Da',  # 'Da' or 'ppm'
        'ms1_accuracy': 20,  # e.g., 20 for the precursor mass tolerance
        'mz_accuracy': 0.02,  # e.g., 0.02 for the fragment mass tolerance
        'n_cores': 4  # Number of CPU cores to use
    })
    
    execute_msfragger(config)


if __name__ == "__main__":
    main()