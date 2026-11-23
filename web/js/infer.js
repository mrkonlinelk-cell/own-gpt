/**
 * In-browser inference with onnxruntime-web.
 * The ONNX model takes token ids (int64, shape [1, seq]) and returns
 * next-token logits (float32, shape [1, vocab]).
 */

import { END_OF_TEXT } from "./tokenizer.js";

const ORT_VERSION = "1.21.0";
const LOCAL_ORT = new URL("./ort/", import.meta.url).href;
const CDN_ORT = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

let ortPromise = null;

/**
 * Prefer a vendored copy of onnxruntime-web (bundled in the Android app so
 * chat works offline). Fall back to the jsDelivr CDN for GitHub Pages.
 */
async function loadOrt() {
  if (ortPromise) return ortPromise;
  ortPromise = (async () => {
    const localEntry = new URL("ort.min.mjs", LOCAL_ORT).href;
    let base = CDN_ORT;
    let mod;
    try {
      mod = await import(localEntry);
      base = LOCAL_ORT;
    } catch {
      mod = await import(`${CDN_ORT}ort.min.mjs`);
      base = CDN_ORT;
    }
    const ort = mod.default ?? mod;
    ort.env.wasm.wasmPaths = base;
    // GitHub Pages cannot send COOP/COEP headers, and many Android WebViews
    // cannot either, so keep WASM single-threaded.
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    return ort;
  })();
  return ortPromise;
}

export async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Could not download the model (${res.status}). Has a training run exported model.onnx yet?`);
  }
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body) {
    const buf = await res.arrayBuffer();
    if (onProgress) onProgress(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (onProgress) onProgress(received, total);
  }
  const out = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

export async function createSession(modelUrl, onProgress) {
  const ort = await loadOrt();
  const buffer = await fetchWithProgress(modelUrl, onProgress);
  const providers = [];
  if (navigator.gpu) providers.push("webgpu");
  providers.push("wasm");
  try {
    return await ort.InferenceSession.create(buffer, { executionProviders: providers });
  } catch (err) {
    if (providers.includes("webgpu")) {
      console.warn("WebGPU failed, falling back to WASM", err);
      return await ort.InferenceSession.create(buffer, { executionProviders: ["wasm"] });
    }
    throw err;
  }
}

function sampleLogits(logits, temperature, topK) {
  const n = logits.length;
  const temp = Math.max(Number(temperature) || 0.8, 1e-6);
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) scores[i] = logits[i] / temp;

  if (topK && topK > 0 && topK < n) {
    const copy = Array.from(scores);
    copy.sort((a, b) => b - a);
    const threshold = copy[topK - 1];
    for (let i = 0; i < n; i++) {
      if (scores[i] < threshold) scores[i] = -Infinity;
    }
  }

  let max = -Infinity;
  for (let i = 0; i < n; i++) if (scores[i] > max) max = scores[i];
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const p = Math.exp(scores[i] - max);
    scores[i] = p;
    sum += p;
  }
  let r = Math.random() * sum;
  for (let i = 0; i < n; i++) {
    r -= scores[i];
    if (r <= 0) return i;
  }
  return n - 1;
}

export async function generateNextTokens(session, tokenizer, promptIds, options) {
  const ort = await loadOrt();
  const {
    maxTokens = 80,
    temperature = 0.8,
    topK = 40,
    blockSize = 256,
    stopAtUser = true,
    signal,
    onToken,
  } = options || {};

  let ids = promptIds.slice();
  if (ids.length === 0) {
    ids = tokenizer.encode(END_OF_TEXT);
  }
  let text = "";
  const stopId = tokenizer.endOfTextId;

  for (let step = 0; step < maxTokens; step++) {
    if (signal && signal.aborted) break;
    const ctx = ids.slice(-blockSize);
    const input = BigInt64Array.from(ctx, (n) => BigInt(n));
    const tensor = new ort.Tensor("int64", input, [1, ctx.length]);
    const result = await session.run({ tokens: tensor });
    const logitsData = result.logits.data;
    // Shape is [1, vocab] (or occasionally [1, 1, vocab] if the exporter wraps it).
    const vocab = tokenizer.vocabSize;
    let logits;
    if (logitsData.length === vocab) {
      logits = logitsData;
    } else if (logitsData.length > vocab) {
      logits = logitsData.subarray(logitsData.length - vocab);
    } else {
      throw new Error(`Unexpected logits length ${logitsData.length}, vocab ${vocab}`);
    }
    const next = sampleLogits(logits, temperature, topK);
    ids.push(next);
    const piece = tokenizer.decode([next]);
    text += piece;
    if (onToken) onToken(piece, text, next);
    if (next === stopId) break;
    if (stopAtUser && /\nUser:/.test(text)) {
      text = text.split(/\nUser:/)[0];
      break;
    }
  }
  return { ids, text };
}

export function buildChatPrompt(history, userMessage) {
  const parts = [];
  for (const turn of history) {
    parts.push(`User: ${turn.user}`);
    parts.push(`Assistant: ${turn.assistant}`);
  }
  parts.push(`User: ${userMessage}`);
  parts.push("Assistant:");
  return parts.join("\n");
}

export function extractReply(text) {
  let reply = text;
  // The prompt already ended with "Assistant:", so `text` is only the continuation.
  if (reply.toLowerCase().startsWith("assistant:")) {
    reply = reply.slice("assistant:".length);
  }
  reply = reply.split(/\nUser:/)[0];
  reply = reply.replace(/<\|endoftext\|>/g, "");
  return reply.trim();
}
