from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd

def calculate_bleu_score(reference_file, hypothesis_file):
    reference_df = pd.read_csv(reference_file)
    hypothesis_df = pd.read_csv(hypothesis_file)

    reference_texts = reference_df['text'].tolist()
    hypothesis_texts = hypothesis_df['summary'].tolist()

    smoothie = SmoothingFunction().method4

    bleu_scores = []
    for ref, hyp in zip(reference_texts, hypothesis_texts):
        reference = [ref.split()]
        hypothesis = hyp.split()
        score = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
        bleu_scores.append(score)

    average_bleu_score = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {average_bleu_score}")

