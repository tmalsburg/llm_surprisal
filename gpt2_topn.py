#!/usr/bin/env python3

import argparse, sys, io

# More packages imported below, but after parsing args to avoid
# unnecessary delays when parameters are mis-specified.

#
# Parameters
#

parser = argparse.ArgumentParser(description='Use GPT2 to generate ranking of the N most likely next tokens.')

parser.add_argument('text', type=str, nargs='?', help='The string of text to be processed.')
parser.add_argument('-n', '--number', type=int, default=10, help='An optional number')
parser.add_argument('-i', '--input', type=argparse.FileType('r', encoding='utf-8'), help='The path to the file from which the input should be read.')
parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='utf-8'), default=io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8'), help='The path to the file to which the results should be written (default is stdout).')
args = parser.parse_args()

#
# Load model:
#

import csv, torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model     = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

#
# Read input text:
#

items = []
if args.input:
  csv_reader = csv.DictReader(args.input)
  for row in csv_reader:
    row['n'] = int(row['n'])
    items.append(row)
else:
  items.append({'item': 1, 'text': args.text, 'n': args.number})

#
# Top-N:
#

def topn(input_text, n):
  input_ids = tokenizer.encode(input_text, return_tensors="pt")
  # Get logits for the last token:
  with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits
  # Get the logits for the last token from the output:
  last_token_logits = predictions[:, -1, :]
  probabilities = F.softmax(last_token_logits, dim=-1)
  topn_probs, topn_tokens = torch.topk(probabilities, n)
  surprisals = (-torch.log2(topn_probs)).tolist()
  topn_tokens_list = [tokenizer.decode([token_id]) for token_id in topn_tokens[0]]
  return zip(topn_tokens_list, surprisals[0])

for item in items:
  item['topn'] = list(topn(item['text'], item['n']))

#
# Write results to file:
#

csvwriter = csv.writer(args.output)
csvwriter.writerow(["item", "s", "w", "rank", "surprisal"])
for item in items:
  for rank,(token, surprisal) in enumerate(item['topn']):
    csvwriter.writerow([item['item'], item['text'], token.strip(), rank+1, surprisal])
