# Add your own training text here

Drop one or more `.txt` files in this folder, then train again.

You can do that:

- **From your phone or the web app:** open the Train page on GitHub Pages, paste
  text or upload a file, and start a training run. Or, in the GitHub app, add a
  file under `data/my_text/`, then run the **Train** workflow from the Actions tab.
- **From your PC:** save `.txt` files here, then run:

```bash
python prepare_data.py
python train.py --resume --fresh-tokens
```

If you also retrained the tokenizer (`--fresh-tokenizer`), start a fresh model
instead of `--resume`, because the old weights will no longer match the vocab.

## Format

Plain text is enough. You do **not** need a special structure.

You can separate documents with the token `<|endoftext|>`.

If you want better chat behaviour, write some examples like this:

```
User: Hello, how are you?
Assistant: I am doing well, thank you! How can I help you?
<|endoftext|>
User: Tell me a short story about a river.
Assistant: Once upon a time, a little river ran down a green hill...
<|endoftext|>
```

Only add text you have the right to use.
