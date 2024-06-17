import json
import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
import webvtt
from langchain.llms import Bedrock

def convert_file(vtt_file, json_file):
    captions = []
    for caption in webvtt.read(vtt_file):
        captions.append({"start": caption.start, "end": caption.end, "text": caption.text})
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(captions, f, ensure_ascii=False, indent=4)
    print(f"VTT file {vtt_file} converted to JSON file {json_file}")

if __name__ == "__main_":
    input_vtt_path = '../data/input_file.vtt'
    processed_json_path = '../data/processed_file.json'
    convert_file(input_vtt_path, processed_json_path)

def process_captions(captions):
  processed_captions = []
  for caption in captions:
    # Split the caption by '/'
    parts = caption.split('/')
    # Extract speaker number from the last part
    speaker_number = parts[-1].split('-')[-1]
    # Extract content by removing timestamp information
    content = caption.split(' --> ')[1].strip()
    processed_captions.append({"speaker": speaker_number, "content": content})
  return processed_captions

def summarize_text(text, model_id):
  # Use LangChain's Bedrock class for easier model interaction
  bedrock_model = Bedrock(model_id=model_id, region=AWS_REGION,
                           aws_access_key_id=AWS_ACCESS_KEY_ID,
                           aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

  # Split text if it exceeds the maximum token limit
  max_tokens = bedrock_model.max_tokens  # Access model's max token limit
  if len(text.split()) > max_tokens:
    split_text = LLModel.split_text(text, max_tokens=max_tokens)
    summaries = []
    for chunk in split_text:
      summary = bedrock_model.generate_text(chunk)
      summaries.append(summary)
    # Combine or process summaries as needed (e.g., ensure coherence)
    return " ".join(summaries)  # Example: Combine summaries with spaces
  else:
    # If text is within limit, call original functionality (existing approach)
    client = boto3.client('bedrock-runtime', 
                          region_name=AWS_REGION, 
                          aws_access_key_id=AWS_ACCESS_KEY_ID, 
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    payload = {
        "inputText": text,
        "textGenerationConfig": {
            "maxTokenCount": 3072,
            "stopSequences": [],
            "temperature": 0.7,
            "topP": 0.9
        }
    }
    response = client.invoke_model(
        body=json.dumps(payload),
        contentType='application/json',
        accept='application/json',
        modelId=model_id
    )
    response_body = json.loads(response['body'].read().decode('utf-8'))
    return response_body.get('generatedText', '')

def summarize_file(input_json, output_json, model_id):
  # Read captions from JSON file
  with open(input_json, 'r') as f:
    data = json.load(f)
  captions = data['captions']

  # Process captions (extract speaker and content)
  processed_captions = process_captions(captions)

  # Summarize each caption (assuming processed_captions has speaker and content)
  for caption in processed_captions:
    caption['summary'] = summarize_text(caption['content'], model_id)

  # Write summaries back to JSON file
  with open(output_json, 'w') as f:
    json.dump(captions, f, ensure_ascii=False, indent=4)
    print(f"JSON file {input_json} summarized to JSON file {output_json}")

def calculate_bleu(reference, hypothesis):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)

def evaluate_summaries(summarized_json, reference_json):
    with open(summarized_json, 'r', encoding='utf-8') as f:
        summarized_data = json.load(f)
    with open(reference_json, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)

    bleu_scores = []
    for i in range(len(summarized_data)):
        hypothesis = summarized_data[i]['summary']
        reference = reference_data[i]['reference_summary']
        bleu_score = calculate_bleu(reference, hypothesis)
        bleu_scores.append(bleu_score)

    avg_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu_score}")
    return avg_bleu_score
