from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
import json
import torch


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


# def pre_process(item):
#     label_id = item["label"]
#     if label_id == 0:
#         label = "entailment"
#     elif label_id == 1:
#         label = "neutral"
#     else:
#         label = "contradiction"
#     return {
#         "premise": item["premise"],
#         "hypothesis": item["hypothesis"],
#         "label": label,
#     }


data = train_set

prompt_template_list = [
    "premise: {premise} hypothesis: {hypothesis} What is the relationship?</s>",
    "Given that '{premise}', is it true that '{hypothesis}'? Determine if it is entailment, contradiction, or neutral.</s>",
    "Does '{hypothesis}' logically follow from '{premise}'?</s>",
    "Classify the relationship between these two sentences: '{premise}' and '{hypothesis}'.</s>",
    "Analyze: If '{premise}', then '{hypothesis}'? Categorize as entailment, contradiction, or neutral.</s>",
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
            return "entailment"
    for keyword in contradiction_keywords:
        if keyword.lower() in generated_text.lower():
            return "contradiction"
    for keyword in neutral_keywords:
        if keyword.lower() in generated_text.lower():
            return "neutral"

    return "neutral"

label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}
def prepare_data(data, prompt):
    inputs, labels = [], []
    for item in data:
        premise = item["premise"]
        hypothesis = item["hypothesis"]
        input_text = prompt.format(premise=premise, hypothesis=hypothesis)
        target_text = label_map[item["label"]]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        label_ids = tokenizer(target_text, return_tensors="pt").input_ids

        inputs.append(input_ids)
        labels.append(label_ids)
    return inputs, labels

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")
best_accuracy = 0
best_prompt = ""
result = {"history": []}
with open("./t5_best_settings.json", "w") as f:
    json.dump(result, f)
for prompt in prompt_template_list:
    accuracy = 0
    positive = 0


    inputs, labels = prepare_data(data, prompt)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # model.train()
    # for epoch in range(20):
    #     total_loss = 0
    #     for input_ids, label_ids in zip(inputs, labels):
    #         optimizer.zero_grad()

    #         outputs = model(input_ids=input_ids, labels=label_ids)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()

    #         total_loss += loss.item()
    #     print(f"Epoch {epoch+1}, Loss: {total_loss / len(inputs)}")

    model.eval()
    with torch.no_grad():

        for item in data:

            premise = item["premise"]
            hypothesis = item["hypothesis"]
            input_text = prompt.format(premise=premise, hypothesis=hypothesis)
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            outputs = model.generate(input_ids)

            decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_label = verbalizer(decoded_output)
            
            if predicted_label == label_map[item["label"]]:
                positive += 1
            print(
                f"Premise: {item['premise']},Hypothesis: {item['hypothesis']},Predicted label: {decoded_output},{positive}\n"
            )

    accuracy = positive / len(train_set)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_prompt = prompt
    current_result = {"prompt": prompt, "accuracy": accuracy}
    with open("./t5_best_settings.json", "r") as f:
        content = json.load(f)
        content["history"].append(current_result)
    with open("./t5_best_settings.json", "w") as f:
        json.dump(content, f)
result = {
    "model_name": "T5",
    "best_accuracy": best_accuracy,
    "best_prompt": best_prompt,
}
with open("./t5_best_settings.json", "r") as f:
    content.update(result)

with open("./t5_best_settings.json", "w") as f:
    json.dump(content, f)
