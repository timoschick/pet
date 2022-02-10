import torch
from os.path import split, join
import argparse
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm


PROMPT = (
    "Let’s explain classification decisions.\n"
    "A man is inside a truck looking out with his left arm in front of a door.\n"
    "question: The man is inside the vehicle.\n"
    "true, false, or neither? true\n"
    "why? Inside a truck implies inside the vehicle.\n"
    "###\n"
    "A black man with sunglasses on his head, wearing a brown apron with a knife in his hand looking at chopped up food.\n"
    "question: A man with pink sunglasses looking at chopped up nuts.\n"
    "true, false, or neither? maybe\n"
    "why? Just because a man is looking at chopped up food it doesn't mean he is looking at chopped up nuts.\n"
    "###\n"
    "A bare chested man in shorts is fixing a roof while another man wearing the same thing looks on.\n"
    "question: The men are shivering cold.\n"
    "true, false, or neither? false\n"
    "why? The men wouldn't be wearing shorts if they're shivering cold.\n"
    "###\n"
    "{premise}\n"
    "question: {hypothesis}\n"
    "true, false, or neither? {label}\n"
    "why?"
)


def extract_explanation(gpt_input, gpt_output):
    ex_start = len(gpt_input)
    ex_end = ex_start
    while ex_end < len(gpt_output) and gpt_output[ex_end] != "\n":
        ex_end += 1
    if ex_end == len(gpt_output):
        print("Check correctness of generation: ", gpt_output[ex_start:ex_end].strip())
    return gpt_output[ex_start:ex_end].strip()


def generate_explanations(
    args, model, tokenizer, eos_token_id, data, prompt, temperature
):
    input = prompt.format(**data)
    input_ids = tokenizer(input, return_tensors="pt").input_ids.to(args.device)

    encoded = model.generate(
        input_ids,
        do_sample=(temperature > 0.0),
        temperature=temperature,
        max_length=len(input_ids[0]) + args.expl_max_length,
        eos_token_id=eos_token_id,
        pad_token_id=eos_token_id,
    )
    output = tokenizer.batch_decode(encoded)[0]
    return extract_explanation(input, output)


def process_dataset(file):
    rows = []
    with open(file) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            rows.append({"premise": row[2], "hypothesis": row[3]})
    return rows


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--expl_max_length", type=int, default=80)
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--model_alias", type=str, default="gpt-neo")
    parser.add_argument("--temperature", type=float, default=0.2)

    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    label_set = {0: "false", 1: "true", 2: "maybe"}
    eos_token_id = tokenizer._convert_token_to_id("###")
    prompt = PROMPT

    data = [label_set]

    for row in tqdm(process_dataset(args.dataset_file)):
        all_labels_explanation = {}
        for label_id, label_str in label_set.items():
            all_labels_explanation[label_id] = generate_explanations(
                args,
                model,
                tokenizer,
                eos_token_id,
                dict(**row, label=label_str),
                prompt,
                args.temperature
            )
        data.append(all_labels_explanation)

    df = pd.DataFrame(data)

    dirname, basename = split(args.dataset_file)
    prefix = basename.split(".")[0]
    output_file = join(dirname, f"{prefix}_{args.model_alias}.jsonl")
    df.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
