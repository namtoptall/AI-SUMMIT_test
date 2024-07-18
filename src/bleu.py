import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def calculate_bleu_score(reference, hypothesis):
    reference = preprocess_text(reference)
    hypothesis = preprocess_text(hypothesis)

    reference_tokens = [reference.split()]
    hypothesis_tokens = hypothesis.split()
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing_function)
    return bleu_score

# def calculate_rouge_scores(reference, hypothesis):
#     rouge = Rouge()
#     scores = rouge.get_scores(hypothesis, reference, avg=True)
#     return scores

def load_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
    return content

def main():
    generated_summary_file = 'results/meeting_summary.txt'
    reference_summary_file = 'data/references_summary.txt'
    public_summary_file = 'data/public_summary.txt'

    generated_summary = load_from_file(generated_summary_file)
    reference_summary = load_from_file(reference_summary_file)
    public_summary = load_from_file(public_summary_file)

    bleu_score_reference = calculate_bleu_score(reference_summary, generated_summary)
    bleu_score_public = calculate_bleu_score(public_summary, generated_summary)
    
    # rouge_scores_reference = calculate_rouge_scores(reference_summary, generated_summary)
    # rouge_scores_public = calculate_rouge_scores(public_summary, generated_summary)

    print(f"BLEU score for reference summary: {bleu_score_reference}")
    print(f"BLEU score for public summary: {bleu_score_public}")
    # print(f"ROUGE scores for reference summary: {rouge_scores_reference}")
    # print(f"ROUGE scores for public summary: {rouge_scores_public}")

if __name__ == "__main__":
    main()
