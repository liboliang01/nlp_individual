import torch
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
import json

# matched_path = "./test_set/dev_matched_sampled-1.jsonl"
# mismatched_path = "./test_set/dev_mismatched_sampled-1.jsonl"
# matched_data = []
# mismatched_data = []
# with open(matched_path, "r", encoding="utf-8") as file:
#     for line in file:
#         matched_data.append(json.loads(line))
# with open(mismatched_path, "r", encoding="utf-8") as file:
#     for line in file:
#         mismatched_data.append(json.loads(line))

data = [
    {
        "annotator_labels": [
            "contradiction",
            "contradiction",
            "contradiction",
            "contradiction",
            "neutral",
        ],
        "genre": "nineeleven",
        "gold_label": "contradiction",
        "pairID": "39496c",
        "promptID": "39496",
        "sentence1": "Further, there is no universally accepted way to transliterate Arabic words and names into English.",
        "sentence1_binary_parse": "( Further ( , ( there ( ( is ( no ( universally ( accepted ( way ( to ( ( transliterate ( Arabic ( ( words and ) names ) ) ) ( into English ) ) ) ) ) ) ) ) . ) ) ) )",
        "sentence1_parse": "(ROOT (S (RBR Further) (, ,) (NP (EX there)) (VP (VBZ is) (NP (DT no) (JJ universally) (JJ accepted) (NN way) (S (VP (TO to) (VP (VB transliterate) (NP (JJ Arabic) (NNS words) (CC and) (NNS names)) (PP (IN into) (NP (NNP English)))))))) (. .)))",
        "sentence2": "Arabic words and names are easily translated.",
        "sentence2_binary_parse": "( ( Arabic ( ( words and ) names ) ) ( ( ( are easily ) translated ) . ) )",
        "sentence2_parse": "(ROOT (S (NP (JJ Arabic) (NNS words) (CC and) (NNS names)) (VP (VBP are) (ADVP (RB easily)) (VP (VBN translated))) (. .)))",
    }
]

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

t5_tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")


def verbalizer(generated_text):
    entailment_keywords = [
        "yes",
        "true",
        "entailment",
        "hold up",
        "logically follow",
        "is a logical conclusion",
        "does imply",
    ]
    contradiction_keywords = [
        "no",
        "false",
        "contradiction",
        "doesn’t",
        "does not follow",
        "is not a logical conclusion",
        "does not imply",
    ]
    neutral_keywords = [
        "maybe",
        "unclear",
        "neutral",
        "hard to say",
        "indeterminate",
        "neither",
        "nor",
    ]

    for keyword in entailment_keywords:
        if keyword.lower() in generated_text.lower():
            return "entailment"
    for keyword in contradiction_keywords:
        if keyword.lower() in generated_text.lower():
            return "contradiction"
    for keyword in neutral_keywords:
        if keyword.lower() in generated_text.lower():
            return "neutral"

    return "neutral"


def run_nli_with_gpt(
    data, data_type, prompt_template, temperature, top_k, top_p, max_new_tokens
):
    print(prompt_template, temperature, top_k, top_p, max_new_tokens)
    correct_predictions = 0
    idx = 0
    for sample in data:
        prompt = prompt_template.format(
            premise=sample["sentence1"], hypothesis=sample["sentence2"]
        )
        inputs = gpt2_tokenizer.encode(prompt, return_tensors="pt")

        output_sequences = gpt2_model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=1,
        )

        generated_sequence = output_sequences[0].tolist()
        try:
            end_of_prompt_index = (
                max(
                    loc
                    for loc, val in enumerate(generated_sequence)
                    if val == inputs[0][-1]
                )
                + 1
            )
            generated_without_prompt = generated_sequence[end_of_prompt_index:]
        except ValueError:
            generated_without_prompt = generated_sequence
        generated_text = gpt2_tokenizer.decode(
            generated_without_prompt, clean_up_tokenization_spaces=True
        )
        print(f"GPT2: {generated_text}")
        predicted_label = verbalizer(generated_text)

        if str(predicted_label) == str(sample["gold_label"]):
            correct_predictions += 1
        idx += 1
        print(predicted_label, correct_predictions, idx)

    accuracy = correct_predictions / len(data)
    current_result = {
        "model": "GPT2",
        "prompt": prompt_template,
        "data_type": data_type,
        "accuracy": accuracy,
    }
    # with open("./gpt2_vs_t5.json", "r") as f:
    #     content = json.load(f)
    #     content["history"].append(current_result)
    # with open("./gpt2_vs_t5.json", "w") as f:
    #     json.dump(content, f)
    return accuracy


def run_nli_with_t5(data, data_type, prompt_template):
    accuracy = 0
    positive = 0
    t5_model.eval()
    with torch.no_grad():
        for item in data:
            premise = item["sentence1"]
            hypothesis = item["sentence2"]
            input_text = prompt_template.format(premise=premise, hypothesis=hypothesis)
            input_ids = t5_tokenizer(input_text, return_tensors="pt").input_ids
            outputs = t5_model.generate(input_ids)

            decoded_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_label = verbalizer(decoded_output)
            print(f"T5: {decoded_output}")

            if predicted_label == item["gold_label"]:
                positive += 1
            print(
                f"Premise: {item['sentence1']},Hypothesis: {item['sentence2']},Predicted label: {decoded_output},{positive}\n"
            )
    accuracy = positive / len(data)
    current_result = {
        "model": "T5",
        "prompt": prompt_template,
        "data_type": data_type,
        "accuracy": accuracy,
    }
    # with open("./gpt2_vs_t5.json", "r") as f:
    #     content = json.load(f)
    #     content["history"].append(current_result)
    # with open("./gpt2_vs_t5.json", "w") as f:
    #     json.dump(content, f)
    return accuracy


if __name__ == "__main__":
    result = {"history": []}
    with open("./gpt2_vs_t5.json", "w") as f:
        json.dump(result, f)
    run_nli_with_gpt(
        data,
        "matched",
        "'{premise}' Therefore, '{hypothesis}' True, false, or unclear?",
        0.7,
        40,
        0.9,
        20,
    )
    # run_nli_with_gpt(
    #     mismatched_data,
    #     "mismatched",
    #     "'{premise}' Therefore, '{hypothesis}' True, false, or unclear?",
    #     0.7,
    #     40,
    #     0.9,
    #     20,
    # )
    run_nli_with_t5(
        data,
        "matched",
        "premise: {premise} hypothesis: {hypothesis} What is the relationship?</s>",
    )
    # run_nli_with_t5(
    #     mismatched_data,
    #     "mismatched",
    #     "premise: {premise} hypothesis: {hypothesis} What is the relationship?</s>",
    # )
