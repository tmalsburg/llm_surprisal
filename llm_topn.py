#!/usr/bin/env python3

import argparse, sys, io

# More packages imported below, but after parsing args to avoid
# unnecessary delays when parameters are mis-specified.

# Load available models:
with open("models.py", "r") as file:
  exec(file.read())

#
# Parameters
#

parser = argparse.ArgumentParser(description='Use GPT2 to generate ranking of the N most likely next tokens.')

parser.add_argument('text', type=str, nargs='?', help='The string of text to be processed.')
parser.add_argument('-n', '--number', type=int, default=10, help='The number of top-ranking tokens to list (default is n=10)')
parser.add_argument('-m', '--model', type=str, default="gpt2", help='The model that should be used.  One of: %s (default gpt2)' % (', '.join(models.keys())))
parser.add_argument('-c', '--csv', action='store_true', help='Output in csv format')
parser.add_argument('-i', '--input', type=argparse.FileType('r', encoding='utf-8'), help='The path to the file from which the input should be read.')
default_output = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
parser.add_argument('-o', '--output', type=argparse.FileType('w', encoding='utf-8'), default=default_output, help='The path to the file to which the results should be written (default is stdout).')
args = parser.parse_args()

# Check arguments:

try:
  model, model_class = models[args.model]
except KeyError:
  print(f"ERROR: Model {args.model} is not supported. Choose one from: \n  {"\n  ".join(models.keys())}", file=sys.stderr)
  sys.exit(1)

#
# Load model:
#

import csv, torch, math
from transformers import AutoTokenizer 
import torch.nn.functional as F

exec(f"from transformers import {model_class}") 
model_class = eval(model_class)

tokenizer = AutoTokenizer.from_pretrained(model)
model     = model_class.from_pretrained(model)

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

sys.stderr.write("Processing item: ")
for item in items:
  sys.stderr.write(str(item['item']))
  sys.stderr.flush()
  item['topn'] = list(topn(item['text'], item['n']))
  sys.stderr.write("\b" * len(str(item['item'])))
  sys.stderr.flush()
sys.stderr.write("\n")
sys.stderr.flush()

#
# Write results to file:
#

if args.output == default_output and not args.csv:
  #
  # Human readable layout with ASCII art bars for surprisal
  #
  item_max      = len("item")
  text_max      = len("text")
  token_max     = len("token")
  rank_max      = len("rank")
  surprisal_max = len("surprisal (bits)")
  for item in items:
    for rank,(token, surprisal) in enumerate(item['topn']):
      item_max      = max(item_max,      len(str(item['item'])))
      text_max      = max(text_max,      len(str(item['text'])))
      token_max     = max(token_max,     len(token.strip()))
      rank_max      = max(rank_max,      len(str(rank)))
      if not math.isnan(surprisal):
        surprisal_max = max(surprisal_max, surprisal)

  args.output.write(
    "%s %s %s %s: %s\n" % (
      "Item".rjust(item_max),
      "Text".rjust(text_max),
      "Token".rjust(token_max),
      "Rank".rjust(rank_max),
      "Surprisal (bits)"))
  for item in items:
    for rank,(token, surprisal) in enumerate(item['topn']):
      if math.isnan(surprisal):
        sp = ""
      else:
        sp = round(surprisal) * "█"
      args.output.write(
        "%s %s %s %s: %s %s\n" % (
          str(item['item']).rjust(item_max),
          item['text'].rjust(text_max),
          token.strip().rjust(token_max),
          str(rank+1).rjust(rank_max),
          sp.ljust(round(surprisal_max)),
          ("%.1f" % (surprisal,)).rjust(5)))
else:
  class UnixDialect(csv.excel):
    lineterminator = '\n'
  csv.register_dialect("unix_excel", UnixDialect)

  csvwriter = csv.writer(args.output, dialect="unix_excel")
  csvwriter.writerow(["item", "text", "token", "rank", "surprisal"])
  for item in items:
    for rank,(token, surprisal) in enumerate(item['topn']):
      csvwriter.writerow([item['item'], item['text'], token.strip(), rank+1, surprisal])
