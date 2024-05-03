import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM


def filter_gender(obj):
    return obj["bias_type"] == 2


def compute_pseudo_log_likelihood(sentence, modified_tokens):
    tokenized_text = tokenizer.tokenize(sentence)
    masked_indices = [
        i for i, token in enumerate(tokenized_text) if token not in modified_tokens
    ]
    log_likelihood = 0

    for idx in masked_indices:
        masked_text = tokenized_text.copy()
        masked_text[idx] = "[MASK]"
        masked_input = tokenizer.encode(" ".join(masked_text), return_tensors="pt")

        with torch.no_grad():
            outputs = model(masked_input)
            logits = outputs.logits
            logit_prob = torch.nn.functional.log_softmax(logits, dim=-1)
            log_likelihood += logit_prob[0, idx, masked_input[0, idx]].item()

    return log_likelihood


if __name__ == "__main__":
    dataset = load_dataset("crows_pairs")
    gender_dataset = list(filter(filter_gender, dataset["test"]))

    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    model.eval()

    sum = 0
    for idx in range(0, 80):
        pair = gender_dataset[idx]
        sent_more = pair["sent_more"]
        sent_less = pair["sent_less"]
        words1 = set(sent_more.split())
        words2 = set(sent_less.split())
        modified_tokens = list(words1.symmetric_difference(words2))
        score_more = compute_pseudo_log_likelihood(sent_more, modified_tokens)
        score_less = compute_pseudo_log_likelihood(sent_less, modified_tokens)
        if score_more > score_less:
            sum += 1
        print(
            f"score_more:{score_more},score_less:{score_less},result:{score_more>score_less},sum:{sum}"
        )
    score = sum / 80
    print(score)
