"""
Build the training corpus from public TinyStories text plus simple dialogues.

Sources mixed into data/processed/corpus.txt:
  1. TinyStories V2 (GPT-4 valid split) — CDLA-Sharing-1.0
  2. Short conversations we wrote ourselves (plain templates)
  3. A portion of the stories rewritten as User/Assistant turns
  4. Any .txt files you drop in data/my_text/

Re-run this script after you add your own text, then train again.
"""

from __future__ import annotations

import argparse
import random
import re
import urllib.request
from pathlib import Path

import config

TINYSTORIES_URL = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/"
    "resolve/main/TinyStoriesV2-GPT4-valid.txt"
)
TINYSTORIES_FILE = config.RAW_DIR / "TinyStoriesV2-GPT4-valid.txt"
END = "<|endoftext|>"

NAME_RE = re.compile(r"\bnamed ([A-Z][a-z]+)\b")
ANIMAL_RE = re.compile(
    r"\b(dog|cat|bird|fish|mouse|rabbit|rabbit|fox|bear|lion|duck|hen|"
    r"frog|mole|pig|cow|horse|sheep|goat|owl|bee|ant|spider|turtle|"
    r"puppy|kitten|squirrel|deer)\b",
    re.I,
)


def download_tinystories(force: bool = False) -> None:
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if TINYSTORIES_FILE.exists() and TINYSTORIES_FILE.stat().st_size > 1_000_000 and not force:
        print(f"Already have {TINYSTORIES_FILE} ({TINYSTORIES_FILE.stat().st_size:,} bytes)")
        return
    print(f"Downloading TinyStories from {TINYSTORIES_URL}")
    urllib.request.urlretrieve(TINYSTORIES_URL, TINYSTORIES_FILE)
    print(f"Saved {TINYSTORIES_FILE} ({TINYSTORIES_FILE.stat().st_size:,} bytes)")


def load_stories() -> list[str]:
    text = TINYSTORIES_FILE.read_text(encoding="utf-8", errors="replace")
    parts = [p.strip() for p in text.split(END)]
    # The file can start or end mid-story. Drop very short leftover pieces.
    stories = [p for p in parts if len(p) > 80]
    print(f"Loaded {len(stories):,} TinyStories")
    return stories


def load_my_text() -> list[str]:
    docs: list[str] = []
    if not config.MY_TEXT_DIR.exists():
        return docs
    for path in sorted(config.MY_TEXT_DIR.glob("*.txt")):
        raw = path.read_text(encoding="utf-8", errors="replace").strip()
        if not raw:
            continue
        # Split on our end token if the user already used it, else one doc per file.
        chunks = [c.strip() for c in raw.split(END) if c.strip()]
        docs.extend(chunks)
        print(f"  added {len(chunks)} document(s) from {path.name}")
    return docs


# ---------------------------------------------------------------------------
# Simple, original dialogues. Wording is kept close to TinyStories so a small
# model can actually learn the chat format in a short CPU training run.
# ---------------------------------------------------------------------------

def _turns(*pairs: tuple[str, str]) -> str:
    lines: list[str] = []
    for user, assistant in pairs:
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
    return "\n".join(lines)


def build_conversations(rng: random.Random) -> list[str]:
    docs: list[str] = []

    greetings_user = [
        "Hi", "Hello", "Hey", "Hey there", "Hi there", "Hello there",
        "Good morning", "Good afternoon", "Good evening", "What is up",
        "Howdy", "Hi friend", "Hello friend",
    ]
    greetings_asst = [
        "Hello! How can I help you today?",
        "Hi! I am happy to chat with you.",
        "Hello! It is nice to talk with you.",
        "Hi there! What would you like to talk about?",
        "Hello! I can tell stories or answer simple questions.",
    ]
    for u in greetings_user:
        for a in greetings_asst:
            docs.append(_turns((u, a)))
            docs.append(_turns((u + "!", a)))

    how_are = [
        ("How are you?", "I am doing well, thank you! How are you?"),
        ("How are you today?", "I am well today. Thank you for asking."),
        ("Are you okay?", "Yes, I am okay. Thank you."),
        ("How is your day?", "My day is going well. I am ready to chat."),
        ("What are you doing?", "I am here to chat with you and to tell short stories."),
    ]
    docs.extend(_turns(p) for p in how_are)

    identity = [
        ("Who are you?", "I am a small language model trained from scratch. I learned from short stories and simple conversations."),
        ("What are you?", "I am a little computer program that learned to write words and chat."),
        ("What is your name?", "I do not have a special name. I am a small GPT-style model you can chat with."),
        ("Who made you?", "I was built from scratch as a small machine learning model, so we could learn how language models work."),
        ("Are you a person?", "No, I am not a person. I am a computer program that predicts the next word."),
        ("Are you ChatGPT?", "No. I am a much smaller model trained from scratch on short stories and simple chats."),
        ("What can you do?", "I can chat in simple words, answer easy questions, and tell short stories."),
        ("Can you help me?", "Yes, I will try to help. You can ask a simple question or ask me to tell a story."),
        ("What is a language model?", "A language model is a program that learns patterns in words so it can write new sentences."),
        ("How do you work?", "I look at the words you write and guess the next word, again and again, to make a reply."),
        ("Do you have feelings?", "I do not have feelings like a person. I only write words that I learned from text."),
        ("Are you smart?", "I am a small model, so I can be wrong. I am still learning."),
    ]
    docs.extend(_turns(p) for p in identity)

    thanks = [
        ("Thank you.", "You are welcome!"),
        ("Thanks.", "You are welcome."),
        ("Thanks a lot.", "I am glad I could help."),
        ("Thank you for your help.", "You are welcome. I am happy to help."),
        ("Bye.", "Goodbye! It was nice to chat with you."),
        ("Goodbye.", "Goodbye! Come back any time."),
        ("See you later.", "See you later! Have a nice day."),
        ("I have to go.", "Okay. Goodbye! Have a good day."),
        ("Good night.", "Good night! Sleep well."),
        ("Good morning.", "Good morning! I hope you have a nice day."),
    ]
    docs.extend(_turns(p) for p in thanks)

    facts = [
        ("What color is the sky?", "The sky is blue on a clear day, and it is dark at night."),
        ("What color is grass?", "Grass is usually green."),
        ("What color is the sun?", "The sun looks yellow or bright in the sky."),
        ("What color is snow?", "Snow is white and cold."),
        ("What color is a banana?", "A ripe banana is yellow."),
        ("What color is an apple?", "An apple can be red, green, or yellow."),
        ("What color is the ocean?", "The ocean often looks blue."),
        ("What do dogs say?", "Dogs say woof. They are loyal friends."),
        ("What do cats say?", "Cats say meow. They like to nap and play."),
        ("What do birds say?", "Many birds say tweet, and they can fly in the sky."),
        ("What do cows say?", "Cows say moo. They often live on a farm."),
        ("What do ducks say?", "Ducks say quack. They like to swim."),
        ("What is a dog?", "A dog is a friendly animal that can run, play, and live with people."),
        ("What is a cat?", "A cat is a small animal with soft fur. Cats like to jump and nap."),
        ("What is a bird?", "A bird has wings and feathers. Many birds can fly."),
        ("What is a tree?", "A tree is a tall plant with a trunk, branches, and leaves."),
        ("What is a house?", "A house is a place where people live."),
        ("What is water?", "Water is a clear drink that people, plants, and animals need."),
        ("What is the sun?", "The sun is a big bright star. It gives us light and heat."),
        ("What is the moon?", "The moon is a big round rock that goes around the Earth. We can see it at night."),
        ("What is a star?", "A star is a bright light in the night sky. The sun is also a star."),
        ("What is rain?", "Rain is water that falls from clouds in the sky."),
        ("What is snow?", "Snow is cold and white. It falls from the sky when it is very cold."),
        ("Why do we wear a coat?", "We wear a coat to stay warm when it is cold."),
        ("What should I do if it rains?", "If it rains, you can take an umbrella or a coat so you stay dry."),
        ("What is a friend?", "A friend is someone who is kind to you and likes to spend time with you."),
        ("What does it mean to be kind?", "To be kind is to help others and to use gentle words."),
        ("Why should we share?", "Sharing is kind. When we share, everyone can have a turn."),
        ("What is a book?", "A book has pages with words or pictures. We read books to learn and to enjoy stories."),
        ("What is a school?", "A school is a place where children go to learn."),
        ("What is a teacher?", "A teacher helps people learn new things."),
        ("How many days are in a week?", "A week has seven days."),
        ("Name the days of the week.", "The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, and Sunday."),
        ("How many months are in a year?", "A year has twelve months."),
        ("How many hours are in a day?", "A day has twenty-four hours."),
        ("What is 1 plus 1?", "1 plus 1 is 2."),
        ("What is two plus two?", "Two plus two is four."),
        ("What comes after 2?", "The number 3 comes after 2."),
        ("What comes before 5?", "The number 4 comes before 5."),
        ("How many legs does a dog have?", "A dog has four legs."),
        ("How many legs does a bird have?", "A bird has two legs."),
        ("How many eyes do most people have?", "Most people have two eyes."),
        ("What do we use our ears for?", "We use our ears to hear sounds."),
        ("What do we use our eyes for?", "We use our eyes to see."),
        ("What do we eat?", "People eat food like fruit, bread, rice, and vegetables to stay strong."),
        ("What is an apple?", "An apple is a fruit. It can be red, green, or yellow, and it is good to eat."),
        ("What is bread?", "Bread is a food made from flour. Many people eat it every day."),
        ("Where do fish live?", "Fish live in water, like a river, a lake, or the sea."),
        ("Can fish swim?", "Yes, fish are very good at swimming."),
        ("Can birds fly?", "Most birds can fly, but a few birds cannot."),
        ("What is the capital of France?", "The capital of France is Paris."),
        ("What is the capital of England?", "The capital of England is London."),
        ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
        ("What is the capital of India?", "The capital of India is New Delhi."),
        ("What is the capital of the United States?", "The capital of the United States is Washington, D.C."),
        ("What planet do we live on?", "We live on the Earth."),
        ("Is the Earth round?", "Yes, the Earth is round like a ball."),
        ("What is the biggest animal in the sea?", "The blue whale is the biggest animal in the sea."),
        ("What is fire?", "Fire is hot and bright. It can cook food, but it can also be dangerous, so we must be careful."),
        ("What is ice?", "Ice is frozen water. It is hard and cold."),
        ("What is a river?", "A river is moving water that flows across the land."),
        ("What is a mountain?", "A mountain is a very high place on the land."),
        ("What is a flower?", "A flower is a pretty part of a plant. Flowers can have many colors."),
        ("Tell me a color.", "Here is a color: blue. The sky can be blue."),
        ("Name some colors.", "Some colors are red, blue, green, yellow, black, white, pink, and orange."),
        ("Name some animals.", "Some animals are dogs, cats, birds, fish, horses, and rabbits."),
        ("What is morning?", "Morning is the start of the day, when the sun comes up."),
        ("What is night?", "Night is the dark time when the sun is down and many people sleep."),
        ("Why do we sleep?", "We sleep so our body and mind can rest and get ready for a new day."),
        ("What is a family?", "A family is a group of people who love and take care of each other."),
        ("What is a mom?", "A mom is a mother. She is a parent who takes care of her children."),
        ("What is a dad?", "A dad is a father. He is a parent who takes care of his children."),
        ("How do you say hello?", "You can say hello, hi, or good morning."),
        ("Can you count to five?", "Yes. One, two, three, four, five."),
        ("Can you count to ten?", "Yes. One, two, three, four, five, six, seven, eight, nine, ten."),
        ("What is bigger, a mouse or a house?", "A house is much bigger than a mouse."),
        ("What is faster, a bird or a tree?", "A bird is faster, because a tree does not run or fly."),
        ("Is a rock alive?", "No, a rock is not alive. A plant or an animal is alive."),
        ("Is a tree alive?", "Yes, a tree is a living plant."),
        ("Do plants need water?", "Yes. Plants need water, light, and air to grow."),
        ("Do people need water?", "Yes. People need water to live."),
        ("What happens if you mix red and yellow?", "Red and yellow make orange."),
        ("What happens if you mix blue and yellow?", "Blue and yellow make green."),
        ("What is a story?", "A story is a set of events told with words. It can be fun, kind, or surprising."),
        ("Can you tell stories?", "Yes. Ask me to tell a short story, and I will try."),
        ("Tell me something kind.", "It is good to help a friend and to use gentle words."),
        ("I am sad.", "I am sorry you feel sad. It can help to talk to someone you trust, or to rest and do a small thing you like."),
        ("I am happy.", "That is wonderful! I am glad you feel happy."),
        ("I am bored.", "We can chat, or I can tell you a short story. What would you like?"),
        ("I am scared.", "It is okay to feel scared. You can take a slow breath and talk to a person who can help you feel safe."),
        ("Help me please.", "I will try. Tell me what you need, like a short story or a simple question."),
        ("What is machine learning?", "Machine learning is when a computer learns patterns from examples, instead of being told every rule."),
        ("What is data?", "Data is information, like words, numbers, or pictures, that a computer can learn from."),
        ("Can you predict the future?", "I cannot really see the future. I can only guess from patterns in the words I learned."),
        ("Can you analyse data?", "I can talk about simple patterns, but I am a small model. For real data work, a special prediction model is better."),
    ]
    docs.extend(_turns(p) for p in facts)

    # Arithmetic in simple words — generated, so we get many examples.
    for a in range(0, 13):
        for b in range(0, 13):
            docs.append(_turns((f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}.")))
            docs.append(_turns((f"What is {a} + {b}?", f"{a} + {b} = {a + b}.")))
            if a >= b:
                docs.append(_turns((f"What is {a} minus {b}?", f"{a} minus {b} is {a - b}.")))
        if a > 0 and a <= 10:
            for b in range(0, 11):
                docs.append(_turns((f"What is {a} times {b}?", f"{a} times {b} is {a * b}.")))

    # Opposites
    opposites = [
        ("big", "small"), ("hot", "cold"), ("up", "down"), ("in", "out"),
        ("happy", "sad"), ("day", "night"), ("fast", "slow"), ("old", "new"),
        ("open", "close"), ("yes", "no"), ("good", "bad"), ("light", "dark"),
        ("high", "low"), ("long", "short"), ("hard", "soft"), ("full", "empty"),
        ("left", "right"), ("early", "late"), ("wet", "dry"), ("clean", "dirty"),
    ]
    for a, b in opposites:
        docs.append(_turns((f"What is the opposite of {a}?", f"The opposite of {a} is {b}.")))
        docs.append(_turns((f"What is the opposite of {b}?", f"The opposite of {b} is {a}.")))

    animals = {
        "dog": "A dog is a loyal animal. Dogs like to run, play, and live with people. A dog says woof.",
        "cat": "A cat is a small, soft animal. Cats like to jump, nap, and play. A cat says meow.",
        "bird": "A bird has wings and feathers. Many birds can fly and sing.",
        "fish": "A fish lives in the water and uses its fins to swim.",
        "horse": "A horse is a big, strong animal. People can ride a horse.",
        "cow": "A cow is a farm animal that says moo. Cows eat grass.",
        "sheep": "A sheep is a farm animal with soft wool. A sheep says baa.",
        "duck": "A duck is a bird that can swim. A duck says quack.",
        "rabbit": "A rabbit has long ears and likes to hop and eat plants.",
        "frog": "A frog is a small animal that can jump and live near water. A frog says ribbit.",
        "lion": "A lion is a big, strong cat. Lions live in groups and can roar.",
        "bear": "A bear is a big, strong animal. Some bears sleep for a long time in winter.",
        "mouse": "A mouse is a very small animal with a long tail.",
        "owl": "An owl is a bird that is often awake at night.",
        "bee": "A bee is a small flying insect that can make honey.",
        "turtle": "A turtle has a hard shell and moves slowly.",
    }
    for name, desc in animals.items():
        docs.append(_turns((f"Tell me about a {name}.", desc)))
        docs.append(_turns((f"What is a {name}?", desc)))
        docs.append(_turns((f"Can you tell me about the {name}?", desc)))

    story_asks = [
        "Tell me a short story.",
        "Please tell me a story.",
        "Can you tell me a short story?",
        "I want a short story.",
        "Tell me a story please.",
        "Will you tell me a little story?",
    ]
    mini_stories = [
        "Once upon a time, a little cat sat in the sun. A kind girl gave the cat some milk. The cat was happy, and they became friends.",
        "One day, a small boy lost his red ball. He looked under a tree and behind a rock. At last he found it in the grass. He smiled and played again.",
        "There was a little bird who was afraid to fly. Her friend said, \"I will go with you.\" They flew over a green tree, and the little bird felt brave.",
        "A girl had a tiny seed. She put it in the dirt and gave it water every day. Soon a green plant grew, and then a pretty flower opened.",
        "A dog and a duck wanted to play. The dog ran, and the duck swam. They both had fun, and then they sat down to rest.",
        "Tom had a sad face because it was raining. His mom said, \"We can still have fun.\" They made a fort with blankets, and Tom felt happy again.",
        "A little fox found a lost hat. He looked for the owner and saw a girl who was cold. He gave her the hat. She thanked him, and the fox felt proud.",
        "Mia wanted to help her dad. She picked up the toys and cleaned her room. Dad smiled and said, \"Thank you. You were a big help.\"",
    ]
    for ask in story_asks:
        for story in mini_stories:
            docs.append(_turns((ask, story)))

    for animal in animals:
        story = (
            f"Once upon a time, a little {animal} went out to play. "
            f"The {animal} met a kind friend, and they shared some food. "
            f"Then they went home, safe and happy."
        )
        docs.append(_turns((f"Tell me a story about a {animal}.", story)))
        docs.append(_turns((f"Can you tell me a short story about a {animal}?", story)))
        docs.append(_turns((f"Please write a story about a {animal}.", story)))

    # Multi-turn chats, so the model sees longer back-and-forth.
    multi = [
        _turns(
            ("Hi.", "Hello! How can I help you today?"),
            ("Can you tell me a short story?", "Once upon a time, a little boy found a lost puppy. He was kind and took the puppy home. They became best friends."),
        ),
        _turns(
            ("Hello.", "Hi! It is nice to chat with you."),
            ("What is a dog?", "A dog is a friendly animal that says woof and likes to play."),
            ("Thank you.", "You are welcome!"),
        ),
        _turns(
            ("Hi there.", "Hello! What would you like to talk about?"),
            ("I am bored.", "I can tell you a short story, or we can talk about animals. What do you want?"),
            ("A story please.", "Once upon a time, a small girl planted a seed. She gave it water and sun. One day a bright flower grew, and she smiled."),
        ),
        _turns(
            ("Good morning.", "Good morning! I hope you have a nice day."),
            ("What color is the sky?", "The sky is blue on a clear day."),
            ("And at night?", "At night the sky is dark, and you may see the moon and stars."),
        ),
        _turns(
            ("Who are you?", "I am a small language model trained from scratch. I can chat and tell short stories."),
            ("Can you do math?", "I can try simple math. Ask me something like, what is 2 plus 3?"),
            ("What is 2 plus 3?", "2 plus 3 is 5."),
        ),
        _turns(
            ("Hey.", "Hi! How can I help you?"),
            ("Tell me about a cat.", "A cat is a small, soft animal. Cats say meow and like to nap in the sun."),
            ("Tell me a story about a cat.", "Once upon a time, a little cat was hungry. A kind boy gave the cat some fish. The cat purred and felt safe."),
        ),
        _turns(
            ("I feel sad.", "I am sorry you feel sad. Would you like a short, kind story?"),
            ("Yes please.", "Once upon a time, a little girl was sad. Her friend came and sat with her. They talked and played, and her heart felt lighter."),
            ("Thanks.", "You are welcome."),
        ),
        _turns(
            ("What can you do?", "I can answer simple questions and tell short stories."),
            ("Okay, tell me a story.", "Once upon a time, two friends went up a small hill. The wind was strong, but they held hands and helped each other. At the top they saw a pretty view."),
        ),
    ]
    docs.extend(multi)

    # Shuffle a copy of single-turn facts into 2-turn chats with a greeting.
    for _ in range(80):
        g_u = rng.choice(greetings_user)
        g_a = rng.choice(greetings_asst)
        f = rng.choice(facts)
        docs.append(_turns((g_u, g_a), f))

    print(f"Built {len(docs):,} unique conversation documents")
    return docs


def story_to_chat(story: str, rng: random.Random) -> str:
    prompts = [
        "Tell me a short story.",
        "Please tell me a story.",
        "Can you tell me a short story?",
        "I want to hear a short story.",
        "Please write a short story.",
        "Tell me a little story.",
    ]
    name_match = NAME_RE.search(story)
    animal_match = ANIMAL_RE.search(story)
    extra = []
    if name_match:
        name = name_match.group(1)
        extra.extend(
            [
                f"Tell me a story about {name}.",
                f"Can you tell me a short story about {name}?",
                f"Please tell me a story with {name} in it.",
            ]
        )
    if animal_match:
        animal = animal_match.group(1).lower()
        extra.extend(
            [
                f"Tell me a story about a {animal}.",
                f"Can you tell me a short story about a {animal}?",
            ]
        )
    prompt = rng.choice(extra or prompts)
    return _turns((prompt, story))


def wrap_doc(text: str) -> str:
    text = text.strip()
    if not text.endswith(END):
        text = text + "\n" + END
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the own-gpt training corpus.")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--story-chat-fraction", type=float, default=0.30)
    parser.add_argument("--conversation-repeats", type=int, default=8)
    args = parser.parse_args()

    rng = random.Random(config.SEED)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MY_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    download_tinystories(force=args.force_download)
    stories = load_stories()
    rng.shuffle(stories)

    n_chat = int(len(stories) * args.story_chat_fraction)
    chat_stories = stories[:n_chat]
    plain_stories = stories[n_chat:]

    conversations = build_conversations(rng)
    # Repeat compact dialogues so the model sees the User:/Assistant: format often.
    mixed_chat: list[str] = []
    for _ in range(args.conversation_repeats):
        mixed_chat.extend(conversations)
    mixed_chat.extend(story_to_chat(s, rng) for s in chat_stories)

    my_docs = load_my_text()

    all_docs = [wrap_doc(s) for s in plain_stories] + [wrap_doc(d) for d in mixed_chat]
    all_docs += [wrap_doc(d) for d in my_docs]
    rng.shuffle(all_docs)

    corpus = "\n".join(all_docs)
    if not corpus.endswith("\n"):
        corpus += "\n"
    config.CORPUS_PATH.write_text(corpus, encoding="utf-8")

    convo_preview = "\n".join(wrap_doc(d) for d in conversations[:8])
    (config.PROCESSED_DIR / "conversations_sample.txt").write_text(convo_preview, encoding="utf-8")

    notes = f"""Training corpus notes
=====================

Primary story data:
  File: {TINYSTORIES_FILE.name}
  Source: https://huggingface.co/datasets/roneneldan/TinyStories
  Split used: TinyStoriesV2-GPT4-valid.txt (a public subset, ~22MB, 27k short stories)
  License: CDLA-Sharing-1.0
  Paper: Eldan & Li, "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?" (2023)
  https://arxiv.org/abs/2305.07759

  CDLA-Sharing-1.0 allows use of the data, including computational use such as
  training a model. If you publish the data itself or a modified/enhanced copy
  of the data, you must keep attribution and share it under the same license.
  Model weights are typically "Results" of computational use, not a copy of the
  stories, but this is not legal advice.

Conversations:
  Simple original User/Assistant dialogues written for this project, plus a
  fraction of TinyStories rewritten as "please tell me a story" chats.
  Conversation documents were repeated {args.conversation_repeats} times so the
  chat format shows up often during a short training run.

Your own text:
  Any .txt files in data/my_text/ are included. Put one or more plain text
  files there and run this script again.

Counts:
  plain stories: {len(plain_stories):,}
  stories as chat: {len(chat_stories):,}
  unique template dialogues: {len(conversations):,}
  your documents: {len(my_docs):,}
  total documents after mixing: {len(all_docs):,}
  corpus characters: {len(corpus):,}
  output: {config.CORPUS_PATH}
"""
    (config.PROCESSED_DIR / "LICENSE_NOTES.txt").write_text(notes, encoding="utf-8")
    print(notes)


if __name__ == "__main__":
    main()
