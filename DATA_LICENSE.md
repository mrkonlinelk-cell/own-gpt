# Training data attribution

This project trains a small language model on public text plus any files you
add yourself.

## TinyStories (CDLA-Sharing-1.0)

Default story data comes from **TinyStories V2 (GPT-4 valid split)**:

- File downloaded by `prepare_data.py`: `TinyStoriesV2-GPT4-valid.txt`
- Source: https://huggingface.co/datasets/roneneldan/TinyStories
- License: **CDLA-Sharing-1.0**
  (https://cdla.dev/sharing-1-0/)
- Paper: Eldan & Li, "TinyStories: How Small Can Language Models Be and Still
  Speak Coherent English?" (2023), https://arxiv.org/abs/2305.07759

CDLA-Sharing-1.0 allows you to use the data, including computational use such
as training a model. If you publish the data itself, or a modified or enhanced
copy of the data, you must keep this attribution and share it under the same
license.

The raw TinyStories file and the processed corpus are **not** stored in this
repository. `prepare_data.py` (and the training workflow) download them when
needed.

Model weights are typically treated as results of computational use, not as a
copy of the stories, but this is not legal advice.

## Conversations written for this project

`prepare_data.py` also builds simple original `User:` / `Assistant:` dialogues
(greetings, easy facts, short story requests) and rewrites a portion of
TinyStories as "please tell me a story" chats so the small model can learn the
chat format.

## Your own text

Any `.txt` files you put in `data/my_text/` are mixed into the corpus. Only add
text you have the right to use.
