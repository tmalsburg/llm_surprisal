#!/usr/bin/env sh

# The tests below simply make sure that all commands work with all
# models without throwing an error.  There's no verification that the
# results actually make sense.

python3 llm_generate.py "This is a" -n 1 -s 1 -m gpt2
python3 llm_generate.py "This is a" -n 1 -s 1 -m gpt2-large
python3 llm_generate.py "This is a" -n 1 -s 1 -m bloom-560m
python3 llm_generate.py "This is a" -n 1 -s 1 -m bloom-1b7
python3 llm_generate.py "This is a" -n 1 -s 1 -m bloom-3b
python3 llm_generate.py "This is a" -n 1 -s 1 -m xglm-564M
python3 llm_generate.py "This is a" -n 1 -s 1 -m xglm-1.7B
python3 llm_generate.py "This is a" -n 1 -s 1 -m xglm-2.9B

python3 llm_topn.py "This is a" -n 2 -m gpt2
python3 llm_topn.py "This is a" -n 2 -m gpt2-large
python3 llm_topn.py "This is a" -n 2 -m bloom-560m
python3 llm_topn.py "This is a" -n 2 -m bloom-1b7
python3 llm_topn.py "This is a" -n 2 -m bloom-3b
python3 llm_topn.py "This is a" -n 2 -m xglm-564M
python3 llm_topn.py "This is a" -n 2 -m xglm-1.7B
python3 llm_topn.py "This is a" -n 2 -m xglm-2.9B
