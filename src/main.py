from processing import generate_report

input_file = 'data/test-file.vtt' 
output_file = 'results/meeting_summary.txt'  
reference_summary = 'data/references_summary.txt'

generate_report(input_file, output_file, reference_summary)

# Load and print the generated summary
with open(output_file, 'r', encoding='utf-8') as file:
    summary = file.read()
    print(summary)
