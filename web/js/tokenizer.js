/**
 * A JavaScript port of our from-scratch BPE tokenizer (tokenizer.py).
 * It loads model/tokenizer.json and must stay in lock-step with the Python
 * encode/decode behaviour so the browser and train.py see the same tokens.
 */

const END_OF_TEXT = "<|endoftext|>";
const UNK = "<|unk|>";

// Same pattern as tokenizer.py. The 'u' flag makes \p{L}/\p{N} closer to
// Python 3's Unicode-aware \w than a plain JavaScript \w would be.
const PRETOKEN_RE =
  /<\|endoftext\|>|'(?:[sSdDmM]|ll|LL|re|RE|ve|VE)| ?[\p{L}\p{N}_]+| ?[^\s\p{L}\p{N}_]+|\s+/gu;

function mergePairInWord(word, pairA, pairB, newToken) {
  const out = [];
  let i = 0;
  while (i < word.length) {
    if (i < word.length - 1 && word[i] === pairA && word[i + 1] === pairB) {
      out.push(newToken);
      i += 2;
    } else {
      out.push(word[i]);
      i += 1;
    }
  }
  return out;
}

export class BPETokenizer {
  constructor(data) {
    this.vocab = data.vocab;
    this.stoi = Object.create(null);
    for (let i = 0; i < this.vocab.length; i++) {
      this.stoi[this.vocab[i]] = i;
    }
    this.merges = data.merges.map((p) => [p[0], p[1]]);
    this.ranks = new Map();
    this.merges.forEach((pair, i) => {
      this.ranks.set(`${pair[0]}\0${pair[1]}`, i);
    });
    this.encodeCache = Object.create(null);
    this.encodeCache[END_OF_TEXT] = [this.stoi[END_OF_TEXT]];
    this.unkId = this.stoi[UNK];
    this.endOfTextId = this.stoi[END_OF_TEXT];
    this.vocabSize = this.vocab.length;
  }

  static async load(url, onProgress) {
    const data = await fetchJson(url, onProgress);
    return new BPETokenizer(data);
  }

  splitPretokens(text) {
    return text.match(PRETOKEN_RE) || [];
  }

  encodeOnePretoken(piece) {
    if (Object.prototype.hasOwnProperty.call(this.encodeCache, piece)) {
      return this.encodeCache[piece];
    }
    if (piece === END_OF_TEXT) {
      const ids = [this.endOfTextId];
      this.encodeCache[piece] = ids;
      return ids;
    }

    let symbols = Array.from(piece);
    while (symbols.length >= 2) {
      let candidate = null;
      let bestRank = null;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.ranks.get(`${symbols[i]}\0${symbols[i + 1]}`);
        if (rank !== undefined && (bestRank === null || rank < bestRank)) {
          bestRank = rank;
          candidate = [symbols[i], symbols[i + 1]];
        }
      }
      if (candidate === null) break;
      symbols = mergePairInWord(symbols, candidate[0], candidate[1], candidate[0] + candidate[1]);
    }

    const ids = symbols.map((s) => (Object.prototype.hasOwnProperty.call(this.stoi, s) ? this.stoi[s] : this.unkId));
    if (Object.keys(this.encodeCache).length < 200000) {
      this.encodeCache[piece] = ids;
    }
    return ids;
  }

  encode(text) {
    const ids = [];
    for (const piece of this.splitPretokens(text)) {
      ids.push(...this.encodeOnePretoken(piece));
    }
    return ids;
  }

  decode(ids) {
    let out = "";
    for (const id of ids) {
      const i = Number(id);
      if (i >= 0 && i < this.vocab.length) {
        const tok = this.vocab[i];
        if (tok !== UNK) out += tok;
        else out += UNK;
      } else {
        out += UNK;
      }
    }
    return out;
  }
}

export async function fetchJson(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url} (${res.status})`);
  if (!onProgress || !res.body || !res.headers.get("content-length")) {
    return res.json();
  }
  const total = Number(res.headers.get("content-length"));
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(received, total);
  }
  const blob = new Blob(chunks);
  const text = await blob.text();
  return JSON.parse(text);
}

export { END_OF_TEXT, UNK };
