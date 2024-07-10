import re
import json
from langchain_aws import BedrockLLM
from langchain.chains import ConversationChain
from langchain.memory.buffer import ConversationBufferMemory
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from config import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,MODEL_ID


def preprocess_file(vtt_file, output_file):
    # open the vtt file and read the lines
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
        # match the time range using regex 
        time_match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})', line)
        if time_match:
            if time_range and speaker and text:
                conversation.append(f"{speaker}|{time_range}|{' '.join(text)}")
            time_range = time_match.group(1)
            speaker = None
            text = []
            continue
        # match the speaker using regex
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
    # append the last conversation to the conversation list
    if time_range and speaker and text:
        conversation.append(f"{speaker}|{time_range}|{' '.join(text)}")

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("\n".join(conversation)) # viet vao thoi gian

    return attendees

def save_to_file(content, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

def load_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

def summarize_chunk(chunk):
    #create a bedrock instance 
    llm = BedrockLLM(
        region_name=AWS_REGION,
        model_id=MODEL_ID,
    )
    prompt_template = """ 
    Develop comprehensive meeting minutes including:
    Attendees: List all participants along with their roles and departments.
    Discussion Points: Detail the topics discussed, including any debates or alternate viewpoints.
    Decisions Made: Record all decisions, including who made them and the rationale.
    Action Items: Specify tasks assigned, responsible individuals, and deadlines.
    Data & Insights: Summarize any data presented or insights shared that influenced the meeting's course.
    Follow-Up: Note any agreed-upon follow-up meetings or checkpoints.
    Furhermore, please do no response to this prompt but only generate a sumaary of the following meeting minutes
    in no more than 350 words: 
    {text}"""
    prompt = prompt_template.format(text=chunk)

    conversation = ConversationChain(
        llm=llm, verbose=True, memory=ConversationBufferMemory()
    )

    summary = conversation.predict(input=prompt)
    return summary


def split_text(text, max_tokens):
    # split the text into 4 parts 
    words = text.split()
    num_tokens = len(words)
    tokens_per_part = num_tokens // 4

    chunks = []
    for i in range(4):
        start_idx = i * tokens_per_part
        end_idx = (i + 1) * tokens_per_part if i < 3 else num_tokens
        chunk = ' '.join(words[start_idx:end_idx])
        chunks.append(chunk)

    return chunks

def summarize_text(text):
    MAX_TOKENS = 8192
    OUTPUT_MAX_TOKENS = 250
    chunks = split_text(text, MAX_TOKENS - OUTPUT_MAX_TOKENS)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    
    unique_summaries = []
    seen_sentences = set()
    for summary in summaries:
        sentences = summary.split('. ')
        filtered_sentences = [s for s in sentences if s and s not in seen_sentences]
        seen_sentences.update(filtered_sentences)
        unique_summaries.append('. '.join(filtered_sentences))
    
    return ' '.join(unique_summaries)

def calculate_bleu_score(reference, hypothesis):
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    reference = preprocess_text(reference)
    hypothesis = preprocess_text(hypothesis)

    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)
    return bleu_score

def generate_report(input_file, output_file, reference_summary=None):
    attendees = preprocess_file(input_file, output_file)
    text = load_from_file(output_file)
    summary = summarize_text(text)
    
    formatted_summary = format_summary_as_report(summary, attendees)
    
    save_to_file(formatted_summary, output_file)
    
    if reference_summary:
        reference_text = reference_summary
        bleu_score = calculate_bleu_score(reference_text, formatted_summary)
        print(f"BLEU score: {bleu_score}")

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
        summary=summary
    )
