"""
All training and model settings in one place.

You can change these values and run train.py again. If you change the
model size (n_layer, n_head, n_embd, block_size) or the tokenizer vocab,
you must start training from scratch — an old checkpoint will not fit.
"""

from pathlib import Path

# ----- paths -----
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MY_TEXT_DIR = DATA_DIR / "my_text"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = ROOT / "checkpoints"
MODEL_DIR = ROOT / "model"
LOG_DIR = ROOT / "logs"
SAMPLES_DIR = ROOT / "samples"
WEB_DIR = ROOT / "web"

CORPUS_PATH = PROCESSED_DIR / "corpus.txt"
TRAIN_BIN = PROCESSED_DIR / "train.bin"
VAL_BIN = PROCESSED_DIR / "val.bin"

# Published, weights-only files (small enough to keep in git / GitHub Pages).
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
MODEL_WEIGHTS = MODEL_DIR / "model.pt"
ONNX_PATH = MODEL_DIR / "model.onnx"
MODEL_META = MODEL_DIR / "meta.json"

# Full training checkpoints (include the optimizer, ~142MB). Not committed.
CKPT_LAST = CHECKPOINT_DIR / "ckpt_last.pt"
CKPT_BEST = CHECKPOINT_DIR / "ckpt_best.pt"
LOSS_LOG = LOG_DIR / "loss.csv"

# Also accept an older tokenizer location from the first local run.
LEGACY_TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"

# ----- tokenizer -----
# Target vocabulary size for our from-scratch BPE tokenizer.
# 4096 is a good fit for short, simple English stories.
VOCAB_SIZE = 4096

# ----- model (decoder-only transformer) -----
# About 12–13 million parameters with the defaults below.
# That is tiny next to ChatGPT, but it is what a CPU can train in under an hour.
N_LAYER = 6       # how many transformer blocks to stack
N_HEAD = 6        # attention heads per block (must divide N_EMBD)
N_EMBD = 384      # width of the residual stream
BLOCK_SIZE = 256  # context length: the model sees this many tokens at once
DROPOUT = 0.0     # 0 helps a short CPU run learn faster; raise to 0.1 for longer training
BIAS = True

# ----- training -----
BATCH_SIZE = 32
MAX_ITERS = 20_000          # upper cap; the wall-clock limit usually stops us first
MAX_WALL_SECONDS = 46 * 60  # default for a local CPU run
EVAL_INTERVAL = 200
EVAL_ITERS = 10
LOG_INTERVAL = 20
CKPT_INTERVAL = 200

LEARNING_RATE = 6e-4
MIN_LR = 6e-5
WARMUP_ITERS = 40
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0

# Hold out this fraction of tokens for eval loss.
VAL_FRACTION = 0.03
SEED = 42

# How generation should behave when we sample after training.
DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_K = 40
DEFAULT_MAX_NEW_TOKENS = 120
