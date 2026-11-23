# own-gpt sample outputs

These samples were generated on this machine after the first from-scratch CPU training run. Nothing here is cherry-picked from a later run: the text under each heading is copied verbatim.

- Checkpoint: `checkpoints/ckpt_best.pt` (step 1096, 12,318,720 parameters)
- Eval loss at this checkpoint: 2.2597 (train 2.1509)
- Sampling: temperature 0.8, top-k 40
- Interpreter: `/workspace/own-image-ai/venv/bin/python`

The writing is rough on purpose to show what a ~12M model can do after about 46 minutes on a CPU. It makes real sentences, but it also repeats itself and gets simple facts wrong.

## Prompt completions (`generate.py`)

Command pattern:

```bash
cd /workspace/own-gpt
/workspace/own-image-ai/venv/bin/python generate.py \
  --prompt "PROMPT" --max-tokens 120 --temperature 0.8 --top-k 40 --seed N
```

### 1. Prompt: `Once upon a time` (seed 1)

Once upon a time, there was a little bug named Tim. Tim could not play with his friends, but he could not play with his friends. One day, Tim and his friends played all day, and they all laughed together.
One day, Tim and his friends decided to play a game. They jumped and jumped, even in the big game. They ran and had so much fun. Tim and his friends laughed and clapped their hands. They all became a big friends, and they played together every day.

### 2. Prompt: `Once upon a time, a little girl` (seed 2)

Once upon a time, a little girl named Lily had a red ball. She loved to play with her ball every day. One day, while playing with her ball, her ball went to play with her.
Lily found a big, red ball. She wanted to play with it, but her mom said, "No, I have my ball." Lily did not want to share. She thought about it was a great idea.
Mom and Lily tried to reach the ball. She asked her mom, "Can I play with you?" Her mom said, "Yes, but be careful. I want to play with it?" Lily

### 3. Prompt: `There was a small dog` (seed 3)

There was a small dog named Spot. He liked to run and run and jump, and sing. Spot was a very big dog with his collar.
One day, Spot and his dog, Spot, were running into the yard. They saw a big dog in the yard. Spot barked, "Spot, come out!" Spot barked, with his mouth. Spot ran to the dog, and Spot ran to get down.
"Spot, come here, Spot!" Spot wagged his tail and barked. Spot barked, and Spot ran to find his mouth. Spot barked, feeling happy to have his tail. Spot ran into

### 4. Prompt: a chat-style story request (seed 4)

Prompt:

```
User: Tell me a short story about a cat.
Assistant:
```

User: Tell me a short story about a cat.
Assistant: Once upon a time there was a little cat named Fluffy. Fluffy lived in a small house with many animals. Fluffy was very happy and loved to play with its friends.
One sunny day, Fluffy and Fluffy went to the park. They wore their favorite clothes. Fluffy saw many people, and other dogs. Fluffy liked to play with Fluffy, and Fluffy played together. Fluffy was very happy.
"Hello, Fluffy!" said Fluffy. Fluffy looked at Fluffy and said, "Hi Fluffy, do you want to play with me?" Fluffy looked at each other and said, "Yes, Fluffy!"

### 5. Prompt: `The little boy found` (seed 5)

The little boy found the shiny rock in the grass. He wanted to share it with his friends.
A little boy named Tim saw Tim and said, "Tim, let's play outside together!" Tim was happy to have his friends. She played with the shiny rock with him. Now, they both had fun fun at the rock together.

## Chat exchanges (`chat.py`)

Multi-turn command used for the three exchanges below:

```bash
cd /workspace/own-gpt
/workspace/own-image-ai/venv/bin/python chat.py \
  --seed 11 --temperature 0.8 --top-k 40 --max-tokens 80 \
  --turns "Hello" "What is 2 plus 2?" "Tell me a short story about a cat"
```

### Exchange 1

User: Hello
Assistant: Hi! I am happy to chat with you.

### Exchange 2

User: What is 2 plus 2?
Assistant: 2 plus 2 is 10.

### Exchange 3

User: Tell me a short story about a cat
Assistant: Once upon a time, in a small house, there lived a kind cat named Mimi. Luna loved to play with her ball. One day, Mimi saw a big, red ball. She was very curious about what was inside.
Anna decided to open the ball, so she jumped around with her paw. She ran and ran as fast as she could. But she could not find to hold the

The greeting is in the right format. The arithmetic is wrong, and the story mixes up names (Mimi / Luna / Anna). That is typical at this size and training budget.

## How to chat

Interactive:

```bash
cd /workspace/own-gpt
/workspace/own-image-ai/venv/bin/python chat.py
```

Type a message and press Enter. Use `/reset` to clear history and `/exit` to quit.

One message, then exit:

```bash
/workspace/own-image-ai/venv/bin/python chat.py --message "Hello"
```

Several turns in one go:

```bash
/workspace/own-image-ai/venv/bin/python chat.py --turns "Hi" "What is a dog?" "Tell me a short story about a cat"
```

Piped input also works (this was verified after training):

```bash
printf 'Hi there\nWhat is a dog?\n/exit\n' | /workspace/own-image-ai/venv/bin/python chat.py
```

Example from that pipe test (seed 13):

```
User: Hi there
Assistant: Hi! I am happy to chat with you.

User: What is a dog?
Assistant: A dog is an animal with a big eyes.
```
