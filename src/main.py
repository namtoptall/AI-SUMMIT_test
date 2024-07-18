from pathlib import Path
from processing import generate_report

def main():
    input_file = 'data/test-file.vtt'
    output_file = 'results/meeting_summary.txt'
    reference_summary_file = 'data/references_summary.txt'

    input_path = Path(input_file)
    output_path = Path(output_file)
    reference_summary_path = Path(reference_summary_file)

    if not input_path.is_file():
        print(f"Error: The input file {input_file} does not exist.")
        return

    reference_summary = reference_summary_path if reference_summary_path.is_file() else None
    if reference_summary is None:
        print(f"Warning: The reference summary file {reference_summary_file} does not exist. Skipping BLEU score calculation.")

    bleu_score = generate_report(str(input_path), str(output_path), str(reference_summary) if reference_summary else None)
if __name__ == "__main__":
    main()
