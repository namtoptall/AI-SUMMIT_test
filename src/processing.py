import re
import nltk
from pathlib import Path
from langchain_aws import BedrockLLM
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, MODEL_ID

def preprocess_file(vtt_file, output_file):
    with open(vtt_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    conversation = []
    time_range = None
    speaker = None
    text = []
    attendees = set()

    for line in lines:
        line = line.strip()

        if re.match(r'\w{8}-\w{4}-\w{4}-\w{4}-\w{12}/\d+-\d+', line):
            continue

        time_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
        if time_match:
            if time_range and speaker and text:
                conversation.append(f"{speaker}|{time_range}|{' '.join(text)}")
            time_range = time_match.group(1)
            speaker = None
            text = []
            continue

        speaker_match = re.match(r'<v\s([^>]+)>(.+)', line)
        if speaker_match:
            if speaker and text:
                conversation.append(f"{speaker}|{time_range}|{' '.join(text)}")
            speaker = speaker_match.group(1)
            attendees.add(speaker)
            text = [speaker_match.group(2).replace('</v>', '').strip()]
            continue

        if line and time_range:
            text.append(line.replace('</v>', '').strip())

    if time_range and speaker and text:
        conversation.append(f"{speaker}|{time_range}|{' '.join(text)}")

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(conversation))

    return attendees


def save_to_file(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def load_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def summarize_chunk(chunk):
    llm = BedrockLLM(
        region_name=AWS_REGION,
        model_id=MODEL_ID,
    )
    prompt_template = """ 
    Please act as a meeting assistant,develop comprehensive meeting summarization including:
    - Attendees: List all participants along with their roles and departments.
    - Discussion Points: Summarize the key topics discussed.
    - Decisions Made: Record the main decisions taken.
    - Action Items: Specify tasks assigned, responsible individuals, and deadlines.
    - Follow-Up: Mention any agreed-upon follow-up meetings or checkpoints.
    
    Please generate a summary of the following meeting minutes in no more than 300 words. 
    Moreover, Avoid repeating information. 
    Format the summary as a report for me.
    {text}"""
    prompt = prompt_template.format(text=chunk)

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    summary = conversation.predict(input=prompt)
    return summary


def split_text(text, max_tokens, overlap_tokens):
    words = text.split()
    chunks = []
    start_idx = 0

    while start_idx < len(words):
        end_idx = min(start_idx + max_tokens, len(words))
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk)
        start_idx += max_tokens - overlap_tokens

    return chunks


def summarize_text(text):
    MAX_TOKENS = 8192
    OVERLAP_TOKENS = 1024
    # OUTPUT_MAX_TOKENS = 250

    chunks = split_text(text, MAX_TOKENS, OVERLAP_TOKENS)
    summaries = [summarize_chunk(chunk) for chunk in chunks]

    unique_summaries = []
    seen_sentences = set()
    # Remove duplicate sentences
    for summary in summaries:
        sentences = summary.split('. ')
        filtered_sentences = [s for s in sentences if s and s not in seen_sentences]
        seen_sentences.update(filtered_sentences)
        unique_summaries.append('. '.join(filtered_sentences))

    return ' '.join(unique_summaries)

def format_summary_as_report(summary, attendees):
    report_template = '''
    ### Meeting Summary Report

    **Attendees:**
    {attendees_count} attendees: {attendees_list}

    **Summary:**
    {summary}
    '''

    attendees_list = ', '.join(attendees)
    attendees_count = len(attendees)
    return report_template.format(
        attendees_count=attendees_count,
        attendees_list=attendees_list,
        summary=summary,
        
    )
    
def generate_report(input_file, output_file, reference_summary=None):
    attendees = preprocess_file(input_file, output_file)
    text = load_from_file(output_file)
    summary = summarize_text(text)

    formatted_summary = format_summary_as_report(summary, attendees)

    save_to_file(formatted_summary, output_file)
    
