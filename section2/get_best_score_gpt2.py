import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import json

"""
label:
0: entailment
1: neutral
2: contradiction

    {
        "premise": "The cat sat on the mat.",
        "hypothesis": "The cat is not on the mat.",
        "label": 0,
    },
"""
dataset = load_dataset("nyu-mll/multi_nli", cache_dir="./multi_nli")
train_set = dataset["train"].select(range(300))

tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

prompt_template_list = [
    "Given the premise: '{premise}', is the hypothesis: '{hypothesis}' true or false? Please answer with 'entailment', 'contradiction', or 'neutral'.",
    "Consider this situation: '{premise}'. Based on that, would you say that '{hypothesis}' is a logical conclusion? Please answer with 'entailment', 'contradiction', or 'neutral'.",
    "Can we infer '{hypothesis}' from the fact that '{premise}'? Indicate if it is 'entailment', 'contradiction', or 'neutral'.",
    "'{premise}' Therefore, '{hypothesis}' True, false, or unclear?",
    "Imagine a scenario where '{premise}'. In this scenario, would '{hypothesis}' also be true? Answer with 'yes' for entailment, 'no' for contradiction, or 'maybe' for neutral.",
]


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
            return "0"
    for keyword in contradiction_keywords:
        if keyword.lower() in generated_text.lower():
            return "2"
    for keyword in neutral_keywords:
        if keyword.lower() in generated_text.lower():
            return "1"

    return "1"


def run_nli_with_model(prompt_template, temperature, top_k, top_p, max_new_tokens):
    print(prompt_template, temperature, top_k, top_p, max_new_tokens)
    correct_predictions = 0
    idx = 0
    for sample in train_set:
        prompt = prompt_template.format(
            premise=sample["premise"], hypothesis=sample["hypothesis"]
        )
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        output_sequences = model.generate(
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
        generated_text = tokenizer.decode(
            generated_without_prompt, clean_up_tokenization_spaces=True
        )

        predicted_label = verbalizer(generated_text)

        if str(predicted_label) == str(sample["label"]):
            correct_predictions += 1
        idx += 1
        print(predicted_label, correct_predictions, idx)

    accuracy = correct_predictions / len(train_set)
    return accuracy


def get_best_parameters():
    best_accuracy = 0
    best_settings = {}

    for prompt_template in prompt_template_list:
        for temperature in [0.7, 0.8]:
            for top_k in [40, 50]:
                for top_p in [0.9, 0.95]:
                    for max_new_tokens in [20, 25]:
                        accuracy = run_nli_with_model(
                            prompt_template, temperature, top_k, top_p, max_new_tokens
                        )
                        if accuracy > best_accuracy:
                            best_accuracy = accuracy
                            best_settings = {
                                "prompt_template": prompt_template,
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p,
                                "max_new_tokens": max_new_tokens,
                            }
                        print("accuracy", accuracy)
                        current_result = {
                            "settings":{
                                "prompt_template": prompt_template,
                                "temperature": temperature,
                                "top_k": top_k,
                                "top_p": top_p,
                                "max_new_tokens": max_new_tokens,
                            },
                            "accuracy":accuracy
                        }
                        with open('./gpt2_best_settings.json', 'r') as f:
                            content = json.load(f)
                            content["history"].append(current_result)
                        with open("./gpt2_best_settings.json", "w") as f:
                            json.dump(content, f)   

    print("best setting:", best_settings)
    result = {
        "model_name": "GPT2",
        "best_accuracy": best_accuracy,
        "best_settings": best_settings,
    }
    with open('./gpt2_best_settings.json', 'r') as f:
        content.update(result)

    with open("./gpt2_best_settings.json", "w") as f:
        json.dump(content, f)   


if __name__ == "__main__":
    result = {"history": []}
    with open("./gpt2_best_settings.json", "w") as f:
        json.dump(result, f)
    get_best_parameters()
