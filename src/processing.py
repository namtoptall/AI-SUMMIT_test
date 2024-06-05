import json
import boto3
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION
import webvtt

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
    
def summarize_text(text, model_id):
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
    with open(input_json, 'r', encoding='utf-8') as f:
        captions = json.load(f)
    
    for caption in captions:
        caption['summary'] = summarize_text(caption['text'], model_id)
    
    with open(output_json, 'w', encoding='utf-8') as f:
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
