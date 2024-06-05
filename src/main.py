from S3_operate import upload_file_to_s3, download_file_from_s3
from processing import convert_file, summarize_file, evaluate_summaries
import time

if __name__ == "__main__":
    input_vtt_path = '../data/test-file.vtt'
    processed_json_path = '../data/processed_file.json'
    summarized_json_path = '../data/summarized_file.json'
    reference_json_path = '../data/references/reference_summaries.json'

    # Model ID for Amazon Bedrock
    model_id = 'amazon.titan-text-premier-v1:0'

    # Upload VTT to S3
    upload_file_to_s3(input_vtt_path, 'test-file.vtt')

    # Download VTT from S3
    download_file_from_s3('test-file.vtt', input_vtt_path)

    # Convert VTT to JSON
    convert_file(input_vtt_path, processed_json_path)

    # Summarize JSON
    summarize_file(processed_json_path, summarized_json_path, model_id)

    # Evaluate summaries using BLEU score
    evaluate_summaries(summarized_json_path, reference_json_path)

    #print 'i'm still running' each 30sec with a for loop 
    for i in range(10):
        print("i'm still running")
        time.sleep(30)

