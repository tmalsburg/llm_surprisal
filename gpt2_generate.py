#!/usr/bin/env python3

import argparse, sys, io

# More packages imported below, but after parsing args to avoid
# unnecessary delays when parameters are mis-specified.

#
# Parameters
#

parser = argparse.ArgumentParser(description='Use GPT2 to generate tokens and calculate per-token surprisal.')

# Task:
parser.add_argument('text', type=str, nargs='?', help='The string of text to be processed.')
parser.add_argument('-n', '--number', type=int, default=10, help='An optional number')
parser.add_argument('-n', '--number', type=int, default=0, help='The number of tokes to generate (default is n=0).')
# Reproducibility:
parser.add_argument('-s', '--seed', type=int, default=None, help='Seed for used for sampling (to force reproducible results)')
# Sampling parameters:
parser.add_argument('-t', '--temperature', type=float, default=1.0, help='Temperature when sampling tokens (default is 1).')
parser.add_argument('-k', '--topk', type=int, default=50, help='Only the top k probabilities are considered for sampling the next token (default k=50)')
# Input, output options:
parser.add_argument('-c', '--csv', action='store_true', help='Output in csv format')
parser.add_argument('-i', '--input', type=argparse.FileType('r', encoding='utf-8'), help='The path to the file from which the input should be read.')
default_output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='utf-8'), default=default_output, help='The path to the file to which the results should be written (default is stdout).')
args = parser.parse_args()

#
# Load model:
#

import csv, torch, random, math
import numpy as np
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model     = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  # if torch.cuda.is_available():
  #   torch.cuda.manual_seed_all(seed)

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

# Add input tokens to n for generate:

for item in items:
  tokens = tokenizer(item['text'], return_tensors="pt")
  item['nt'] = item['n'] + len(tokens['input_ids'][0])

#
# Generate:
#

def generate(input_text, nt):
  input_tokens = tokenizer(input_text, return_tensors="pt")
  output_tokens = model.generate(**input_tokens, max_length=nt, temperature=args.temperature, top_k=args.topk, repetition_penalty=1.0, do_sample=True)
  output_text = tokenizer.batch_decode(output_tokens)[0]
  return output_text

if args.seed:
  set_seed(args.seed)

for item in items:
  if item['n'] > 0:
    item['text'] = generate(item['text'], item['nt'])

#
# Surprisal:
#

def surprisal(input_text):
  input_tokens = tokenizer.encode(input_text, return_tensors='pt')
  with torch.no_grad():
    outputs = model(input_tokens, labels=input_tokens)
    logits = outputs.logits

  shifted_logits = logits[..., :-1, :].contiguous()
  shifted_tokens = input_tokens[..., 1:].contiguous()

  # Calculate the log probabilities:
  log_probs = F.log_softmax(shifted_logits, dim=-1)

  # Gather the log probabilities of the target tokens:
  target_log_probs = log_probs.gather(2, shifted_tokens.unsqueeze(-1)).squeeze(-1)

  # Calculate surprisal values: negative log probability
  surprisals = -target_log_probs

  # Convert from log base e to log base 2 (optional, depending on the
  # definition of surprisal you're using):
  surprisals = surprisals / torch.log(torch.tensor(2.0))

  decoded_tokens = [tokenizer.decode([token]) for token in shifted_tokens.squeeze().tolist()]

  return zip([tokenizer.decode([input_tokens[0][0]])] + decoded_tokens,
             [float('nan')] + surprisals.tolist()[0]) 

for item in items:
  item['surprisals'] = list(surprisal(item['text']))

#
# Write results to file:
#

if args.output == default_output and not args.csv:
  #
  # Human readable layout with ASCII art bars for surprisal
  #
  item_max = len("item")
  idx_max = len("idx")
  token_max = len("token")
  surprisal_max = len("surprisal (bits)")
  for item in items:
    for idx,(token, surprisal) in enumerate(item['surprisals']):
      item_max      = max(item_max,      len(str(item['item'])))
      idx_max       = max(idx_max,       len(str(idx+1)))
      token_max     = max(token_max,     len(token.strip()))
      if not math.isnan(surprisal):
        surprisal_max = max(surprisal_max, surprisal)

  args.output.write(
    "%s %s %s: %s\n" % (
      "Item".rjust(item_max),
      "Idx".rjust(idx_max),
      "Token".rjust(token_max),
      "Surprisal (bits)"))
  for item in items:
    for idx,(token, surprisal) in enumerate(item['surprisals']):
      if math.isnan(surprisal):
        sp = ""
      else:
        sp = round(surprisal) * "█"
      args.output.write(
        "%s %s %s: %s %s\n" % (
          str(item['item']).rjust(item_max),
          str(idx+1).rjust(idx_max),
          token.strip().rjust(token_max),
          sp.ljust(round(surprisal_max)),
          ("%.1f" % (surprisal,)).rjust(5)))
else:
  #
  # CSV output:
  #
  class UnixDialect(csv.excel):
    lineterminator = '\n'
  csv.register_dialect("unix_excel", UnixDialect)
  csvwriter = csv.writer(args.output, dialect="unix_excel")
  csvwriter.writerow(["item", "idx", "token", "surprisal"])
  for item in items:
    for idx,(token, surprisal) in enumerate(item['surprisals']):
      csvwriter.writerow([item['item'], idx+1, token.strip(), surprisal])

