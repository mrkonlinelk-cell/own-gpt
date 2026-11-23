# own-gpt

A tiny language model written and trained **from scratch** — no pretrained weights, no borrowed tokenizer. You can chat with it in the browser, on your PC, or from your phone via GitHub Actions. You can also add your own text and keep training it.

It is a learning project, not a replacement for ChatGPT, DeepSeek, or similar systems. Those models are trained on trillions of words with thousands of GPUs. This one has about **12 million parameters** and a **256-token** context window. After a short CPU training run it can write rough, simple English and try to reply in a `User:` / `Assistant:` format. It still makes plenty of mistakes.

**Live site:** <https://mrkonlinelk-cell.github.io/own-gpt/>

## What’s in this repo

| Path | Role |
|------|------|
| `model.py` | The GPT-style model (embeddings, causal attention, MLP, LayerNorm), written in PyTorch. |
| `tokenizer.py` | Our own BPE tokenizer, trained on our text. |
| `prepare_data.py` | Downloads TinyStories, builds simple dialogues, and includes `data/my_text/`. |
| `train.py` | Trains the model and writes `model/model.pt`. |
| `export_onnx.py` | Exports `model/model.onnx` for the in-browser chat page. |
| `chat.py` / `generate.py` | Terminal chat and prompt completion. |
| `data/my_text/` | Drop your own `.txt` files here (from your phone, PC, or the Train page). |
| `model/` | Published weights (`model.pt`, fp16, ~25MB), `tokenizer.json`, and `model.onnx`. |
| `web/` | The Chat and Train pages, hosted on GitHub Pages. |
| `.github/workflows/train.yml` | Train from your phone or PC. |
| `.github/workflows/chat.yml` | Send a message and read the reply in the job summary. |

The code is MIT-licensed (see `LICENSE`). Training data attribution for TinyStories (CDLA-Sharing-1.0) is in `DATA_LICENSE.md`. The raw TinyStories file is not stored here; the training workflow downloads it.

## Chat in the browser

Open the [Chat page](https://mrkonlinelk-cell.github.io/own-gpt/). The model downloads to your phone or PC and runs locally with ONNX Runtime. Nothing you type is sent to a server.

There are temperature, top-k, and max-token controls in Settings.

## Train from your phone

You have two equally good options.

### Option A — the Train page (easiest)

1. Open [Train](https://mrkonlinelk-cell.github.io/own-gpt/train.html).
2. Create a **fine-grained personal access token** limited to this repository, with **Contents: Read and write** and **Actions: Read and write**:
   [github.com/settings/personal-access-tokens/new](https://github.com/settings/personal-access-tokens/new).
3. Paste the token and save it. It stays in **this browser only** (`localStorage`). It is never written to the repo.
4. Paste text or upload `.txt` files (optional), choose how many minutes to train (default 60, max 330), and whether to **continue** the current model or start **fresh**.
5. Tap **Start training run**. Watch progress under Recent training runs.

When the run finishes, the site redeploys and Chat uses the new model.

### Option B — the GitHub app

1. Open this repository in the GitHub mobile app or on the website.
2. Add a `.txt` file under `data/my_text/` (Add file → Create new file, or upload).
3. Go to the **Actions** tab.
4. Choose **Train**.
5. Tap **Run workflow**.
6. Set minutes (default 60, max 330) and mode (`continue` or `fresh`).
7. Run it. When it finishes, open `samples/latest.md` to see example replies.

To chat from the Actions tab without the website: run the **Chat** workflow, type a message, and read the reply in the run’s job summary.

## Train and chat on your PC

You need Python 3.10+ and a few packages:

```bash
git clone https://github.com/mrkonlinelk-cell/own-gpt.git
cd own-gpt
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
# If you have an NVIDIA GPU, install the official CUDA build of PyTorch instead.
```

Then:

```bash
python prepare_data.py          # downloads TinyStories and mixes in data/my_text
python train.py                 # new model, or:
python train.py --resume        # continue from model/model.pt
python train.py --resume --fresh-tokens --max-wall-seconds 3600
python export_onnx.py           # update the browser model
python chat.py                  # interactive chat
python generate.py --prompt "Once upon a time"
```

If your PC has an NVIDIA GPU, training will be much faster than on GitHub’s free CPU machines. Put your `.txt` files in `data/my_text/` first.

Useful flags:

```bash
python train.py --resume                  # fine-tune the published weights
python train.py --fresh-tokens            # re-encode the corpus (do this after adding text)
python train.py --fresh-tokenizer         # rebuild the vocabulary (cannot resume old weights)
python train.py --skip-full-ckpt          # do not write 142MB optimizer checkpoints
```

`model/model.pt` is a weights-only fp16 file (about 25MB) so it fits under GitHub’s 100MB file limit. `train.py --resume` can fine-tune from it; the optimizer starts fresh unless a local `checkpoints/ckpt_last.pt` is present.

## Data format

You do **not** need structured data. Plain `.txt` files are enough for the model to learn language and style.

Optional, if you want to shape how it answers, write chat examples:

```
User: Who made you?
Assistant: Vihanga built me from scratch.
<|endoftext|>
User: Tell me a short story about a river.
Assistant: Once upon a time, a little river ran down a green hill...
```

Only add text you have the right to use.

## Limits (please read)

- Training on GitHub Actions uses a **CPU**, not a GPU, so it is slow.
- Each Actions job can run at most **6 hours**. The Train workflow caps you at **330 minutes** so there is time left to export the model and save it.
- For a **public** repository, GitHub Actions minutes are **unlimited** (at GitHub’s current public-repo policy).
- The model is small. It will not reason like ChatGPT. It may invent facts (for example, it once said 2 + 2 is 10).
- The first visit to the Chat page downloads the ONNX model (tens of megabytes). After that, the browser can cache it.

## How the model thinks (plain language)

1. **Tokenizer.** Text is split into tokens (characters merged into common chunks like `" the"` or `"once"`). We learn these merges ourselves with byte-pair encoding.
2. **Embeddings.** Each token, and each position in the window, becomes a vector of numbers.
3. **Causal self-attention.** Every token looks at earlier tokens and decides which ones matter. It is not allowed to look ahead.
4. **MLP.** A small neural net at each position gives the model extra capacity after attention.
5. **Stack of blocks.** We stack this pattern six times, with LayerNorm and residual connections.
6. **Next-token prediction.** Training asks: “Given the words so far, what comes next?” At chat time we sample from those scores, one token at a time.

Default size: 6 layers, 6 heads, 384-wide residual stream, context 256, BPE vocabulary of 4,096, about 12 million unique parameters (token embeddings are tied to the output layer).

## Automatic training on new files

The Train workflow is **manual** by default (`workflow_dispatch`) so a save cannot start an accidental loop. If you want it to start whenever you push a file under `data/my_text/`, uncomment the `push:` block at the top of `.github/workflows/train.yml`. Bot commits only update `model/` and `samples/`, so they will not retrigger that path filter.
