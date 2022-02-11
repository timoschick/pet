'''
GPT baseline: apply GPT to dev/test set, generate labels. 
'''
import torch
from os.path import split, join
import argparse
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm


PHLE_PROMPT = (
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
    "true, false, or neither?"
)

PHL_PROMPT = (
    "Let’s judge a claim.\n"
    "A man is inside a truck looking out with his left arm in front of a door.\n"
    "question: The man is inside the vehicle.\n"
    "true, false, or neither? true\n"
    "###\n"
    "A black man with sunglasses on his head, wearing a brown apron with a knife in his hand looking at chopped up food.\n"
    "question: A man with pink sunglasses looking at chopped up nuts.\n"
    "true, false, or neither? maybe\n"
    "###\n"
    "A bare chested man in shorts is fixing a roof while another man wearing the same thing looks on.\n"
    "question: The men are shivering cold.\n"
    "true, false, or neither? false\n"
    "###\n"
    "{premise}\n"
    "question: {hypothesis}\n"
    "true, false, or neither?"
)

PHEL_PROMPT = (
    "Let’s analyze the situation and make a judgment.\n"
    "A man is inside a truck looking out with his left arm in front of a door.\n"
    "claim: The man is inside the vehicle.\n"
    "analysis: Inside a truck implies inside the vehicle. true, false, or neither? true\n"
    "###\n"
    "A black man with sunglasses on his head, wearing a brown apron with a knife in his hand looking at chopped up food.\n"
    "claim: A man with pink sunglasses looking at chopped up nuts.\n"
    "analysis: Just because a man is looking at chopped up food it doesn't mean he is looking at chopped up nuts. true, false, or neither? maybe\n"
    "###\n"
    "A bare chested man in shorts is fixing a roof while another man wearing the same thing looks on.\n"
    "claim: The men are shivering cold.\n"
    "analysis: The men wouldn't be wearing shorts if they're shivering cold. true, false, or neither? false\n"
    "###\n"
    "{premise}\n"
    "claim: {hypothesis}\n"
    "analysis:"
)


def process_dataset(file):
    rows = []
    with open(file) as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            rows.append(
                {
                    "label": row[1],
                    "premise": row[2], 
                    "hypothesis": row[3]
                }
            )
    return rows


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches


def generate_label(
    args, model, tokenizer, eos_token_id, data, prompt, temperature,
    after_string=None
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

    if after_string==None:
        sol = len(input)
    else:
        idx_list = list(find_all(output, after_string))
        if len(idx_list) == 0: 
            return ""
        else:
            sol = idx_list[0] + len(after_string)
    while sol < len(output) and output[sol] == " ":
            sol += 1
    eol = sol
    while eol < len(output) and output[eol] not in [" ", "\n"]:
        eol += 1
    return output[sol:eol].strip()


def get_accuracy(args, prompt, model, tokenizer, after_string=None):
    label_map = {"false": "contradiction", "true": "entailment", "maybe": "neutral"}
    eos_token_id = tokenizer._convert_token_to_id("###")

    correct = 0
    total = 0
    pred_dsn = {"contradiction": 0, "entailment": 0, "neutral": 0}
    correct_pred_dsn = {"contradiction": 0, "entailment": 0, "neutral": 0}
    for row in tqdm(process_dataset(args.dataset_file)):
        gpt_label = generate_label(
            args,
            model,
            tokenizer,
            eos_token_id,
            dict(premise=row['premise'], hypothesis=row['hypothesis']),
            prompt,
            args.temperature,
            after_string=after_string,
        )
        if gpt_label in label_map:
            gpt_label = label_map[gpt_label]
            pred_dsn[gpt_label] += 1
        if gpt_label == row['label']:
            correct += 1
            correct_pred_dsn[gpt_label] += 1
        total += 1
    print('accuracy: ', correct/total)
    print('pred distribution: ', pred_dsn)
    print('pred correct_pred_dsn: ', correct_pred_dsn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_file", type=str, default='/data/shared/data/e-SNLI/dataset/esnli_dev_100.csv')
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--expl_max_length", type=int, default=80)
    parser.add_argument("--model_name", type=str, default="EleutherAI/gpt-neo-2.7B")
    parser.add_argument("--model_alias", type=str, default="gpt-neo")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--PHLE", action='store_true', help="Whether to use phle prompt")
    parser.add_argument("--PHL", action='store_true', help="Whether to use phl prompt")
    parser.add_argument("--PHEL", action='store_true', help="Whether to use phel prompt")

    args = parser.parse_args()

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    
    if args.PHLE:
        print('PHLE')
        get_accuracy(
            args=args,
            prompt=PHLE_PROMPT,
            model=model,
            tokenizer=tokenizer,
        )
    if args.PHL:
        print('PHL')
        get_accuracy(
            args=args,
            prompt=PHL_PROMPT,
            model=model,
            tokenizer=tokenizer,
        )
    if args.PHEL:
        print('PHEL')
        get_accuracy(
            args=args,
            prompt=PHL_PROMPT,
            model=model,
            tokenizer=tokenizer,
            after_string="true, false, or neither?"
        )


if __name__=='__main__':
    main()