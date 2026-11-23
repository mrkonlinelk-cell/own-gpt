/**
 * In-browser inference with onnxruntime-web (WASM only).
 *
 * The ONNX model takes token ids (int64, shape [1, seq]) and returns
 * next-token logits (float32, shape [1, vocab]).
 *
 * We never load the WebGPU/JSEP build. A failed WebGPU initWasm() would
 * poison the runtime so the WASM backend also reports
 * "previous call to 'initWasm()' failed."
 *
 * GitHub Pages and Android WebView are not cross-origin isolated, so
 * SharedArrayBuffer / multi-threaded WASM is unavailable. We force
 * numThreads = 1 and proxy = false. If the device also lacks WASM SIMD,
 * we fall back to onnxruntime-web 1.18, which still ships a non-SIMD build.
 */

import { END_OF_TEXT } from "./tokenizer.js";

const ORT_PRIMARY = {
  id: "wasm-simd-1.21",
  version: "1.21.0",
  // WASM-only entry. Do not use ort.min.mjs (that bundle also registers WebGPU).
  entry: "ort.wasm.min.mjs",
  mjs: "ort-wasm-simd-threaded.mjs",
  wasm: "ort-wasm-simd-threaded.wasm",
  cdnDist: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/",
  localDir: new URL("./ort/", import.meta.url).href,
  simd: true,
};

const ORT_LEGACY = {
  id: "wasm-nosimd-1.18",
  version: "1.18.0",
  // 1.18's ESM wasm-only build still includes ort-wasm.wasm (no SIMD).
  entry: "ort.wasm.min.js",
  wasm: "ort-wasm.wasm",
  cdnEntry: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/esm/ort.wasm.min.js",
  cdnDist: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/",
  localDir: new URL("./ort/legacy/", import.meta.url).href,
  simd: false,
};

export function wasmSimdSupported() {
  try {
    // Official onnxruntime-web SIMD probe (16 zero bytes after v128.const).
    return WebAssembly.validate(
      new Uint8Array([
        0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1, 28, 0, 65, 0, 253, 15,
        253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 186, 1, 26, 11,
      ])
    );
  } catch {
    return false;
  }
}

export function collectDiagnostics() {
  let sabPostable = false;
  try {
    if (typeof SharedArrayBuffer !== "undefined" && typeof MessageChannel !== "undefined") {
      new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
      sabPostable = true;
    }
  } catch {
    sabPostable = false;
  }
  return {
    crossOriginIsolated: globalThis.crossOriginIsolated === true,
    hasSharedArrayBuffer: typeof SharedArrayBuffer !== "undefined",
    sabPostable,
    wasmSimd: wasmSimdSupported(),
    webgpu: typeof navigator !== "undefined" && Boolean(navigator.gpu),
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
    deviceMemory: typeof navigator !== "undefined" ? navigator.deviceMemory : undefined,
    hardwareConcurrency: typeof navigator !== "undefined" ? navigator.hardwareConcurrency : undefined,
  };
}

function joinUrl(base, name) {
  return new URL(name, base.endsWith("/") ? base : base + "/").href;
}

async function urlExists(url) {
  try {
    const res = await fetch(url, { method: "GET", headers: { Range: "bytes=0-0" } });
    if (res.ok || res.status === 206) return true;
    // Some WebViews ignore Range and still return 200 only for a full GET.
    if (res.status === 405 || res.status === 501) {
      const again = await fetch(url, { method: "GET" });
      return again.ok;
    }
    return false;
  } catch {
    return false;
  }
}

async function resolveConfig(spec) {
  const localEntryName = spec.entry;
  const localEntry = joinUrl(spec.localDir, localEntryName);
  const usedLocal = await urlExists(localEntry);
  if (spec.id === "wasm-simd-1.21") {
    const base = usedLocal ? spec.localDir : spec.cdnDist;
    const entry = joinUrl(base, spec.entry);
    const mjs = joinUrl(base, spec.mjs);
    const wasm = joinUrl(base, spec.wasm);
    return {
      spec,
      source: usedLocal ? "local" : "cdn",
      entry,
      wasmPaths: { mjs, wasm },
      files: [entry, mjs, wasm],
    };
  }
  const entry = usedLocal ? localEntry : spec.cdnEntry;
  const dist = usedLocal ? spec.localDir : spec.cdnDist;
  const wasm = joinUrl(dist, spec.wasm);
  // Map every 1.18 filename ORT might ask for onto the non-SIMD build we ship,
  // in case a default simd=true still slips through.
  const wasmPaths = {
    "ort-wasm.wasm": wasm,
    "ort-wasm-simd.wasm": wasm,
    "ort-wasm-threaded.wasm": wasm,
    "ort-wasm-simd-threaded.wasm": wasm,
  };
  return {
    spec,
    source: usedLocal ? "local" : "cdn",
    entry,
    wasmPaths,
    files: [entry, wasm],
  };
}

function applyWasmFlags(ort, wasmPaths, opts) {
  // Set these before the first InferenceSession.create() so initWasm()
  // never starts a worker or a proxy. ORT 1.18 defaults simd to true and
  // will 404 ort-wasm-simd.wasm unless we force the non-SIMD filename.
  const simd = opts && opts.simd === false ? false : opts && opts.simd === true;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  if (opts && opts.simd === false) {
    ort.env.wasm.simd = false;
  }
  ort.env.wasm.wasmPaths = wasmPaths;
}

const ORT_SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 30, 1, 28, 0, 65, 0, 253, 15,
  253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 186, 1, 26, 11,
]);

function wasmSimdProbeOfficial() {
  try {
    return WebAssembly.validate(ORT_SIMD_PROBE);
  } catch {
    return false;
  }
}

function probeBytesEqual(buf) {
  const u = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u.length !== ORT_SIMD_PROBE.length) return false;
  for (let i = 0; i < u.length; i++) if (u[i] !== ORT_SIMD_PROBE[i]) return false;
  return true;
}

let simdProbePatched = false;

/**
 * Some Chromium/WebView builds can compile ORT's SIMD wasm even though
 * WebAssembly.validate() rejects ORT's tiny feature-probe module. 1.21
 * then throws "WebAssembly SIMD is not supported" and never loads. If
 * the real simd wasm compiles, patch only that exact probe.
 */
async function enableOrtSimdProbeIfWasmWorks(wasmUrl) {
  if (wasmSimdProbeOfficial()) return { ok: true, patched: false };
  try {
    const res = await fetch(wasmUrl);
    if (!res.ok) return { ok: false, patched: false, failedFile: wasmUrl, status: res.status };
    const buf = await res.arrayBuffer();
    await WebAssembly.compile(buf);
  } catch (err) {
    return { ok: false, patched: false, failedFile: wasmUrl, error: err && err.message ? err.message : String(err) };
  }
  if (!simdProbePatched) {
    const orig = WebAssembly.validate.bind(WebAssembly);
    WebAssembly.validate = (bytes) => {
      try {
        if (orig(bytes)) return true;
      } catch {
        /* ignore */
      }
      try {
        if (probeBytesEqual(bytes)) return true;
      } catch {
        /* ignore */
      }
      return false;
    };
    simdProbePatched = true;
  }
  return { ok: true, patched: true };
}

async function importOrt(entryUrl, cacheBust) {
  const url = cacheBust ? `${entryUrl}${entryUrl.includes("?") ? "&" : "?"}bust=${cacheBust}` : entryUrl;
  const mod = await import(url);
  return mod.default ?? mod;
}

function formatOrtError(err, diags, attempts) {
  const detail = err && err.message ? err.message : String(err);
  const attemptLines = attempts
    .map((a) => `  - ${a.id} (${a.source}): ${a.error || a.status}`)
    .join("\n");
  return [
    "The model runtime could not start in this browser.",
    "",
    detail,
    "",
    "Diagnostics:",
    `  crossOriginIsolated: ${diags.crossOriginIsolated}`,
    `  SharedArrayBuffer: ${diags.hasSharedArrayBuffer} (postable: ${diags.sabPostable})`,
    `  WASM SIMD: ${diags.wasmSimd} (compile: ${diags.wasmSimdCompile ?? "?"}, probePatched: ${diags.wasmSimdPatched ?? false})`,
    `  WebGPU: ${diags.webgpu}`,
    `  memory: ${diags.deviceMemory ?? "?"} GB, cores: ${diags.hardwareConcurrency ?? "?"}`,
    "",
    "Attempts:",
    attemptLines,
    "",
    "Pages and the Android app are not cross-origin isolated, so the app uses a single-thread WASM build. If SIMD is missing, a non-SIMD fallback is tried.",
  ].join("\n");
}

export async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Could not download the model (${res.status} from ${url}).`);
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

let loadedOrt = null;

/**
 * Create a WASM inference session. Never uses WebGPU.
 * Retries with the non-SIMD ORT 1.18 build if the primary init fails
 * (or if this device has no WASM SIMD).
 */
export async function createSession(modelUrl, onProgress) {
  const diags = collectDiagnostics();
  const buffer = await fetchWithProgress(modelUrl, onProgress);
  // Always try the 1.21 SIMD build first (after confirming or patching the
  // SIMD probe), then the 1.18 non-SIMD build. Do not skip 1.21 only because
  // the official tiny probe is a false negative on some Chromium builds.
  const specs = [ORT_PRIMARY, ORT_LEGACY];
  const attempts = [];
  let lastErr = null;

  for (const spec of specs) {
    const cfg = await resolveConfig(spec);
    const record = { id: spec.id, source: cfg.source, entry: cfg.entry, error: "", status: "trying" };
    attempts.push(record);
    try {
      const missing = [];
      for (const file of cfg.files) {
        if (!(await urlExists(file))) missing.push(file);
      }
      if (missing.length) {
        throw new Error(`Required runtime file not found: ${missing.join(", ")}`);
      }
      if (spec.simd) {
        const wasmUrl = cfg.wasmPaths.wasm || cfg.files[cfg.files.length - 1];
        const probe = await enableOrtSimdProbeIfWasmWorks(wasmUrl);
        diags.wasmSimdPatched = Boolean(probe.patched);
        diags.wasmSimdCompile = Boolean(probe.ok);
        if (!probe.ok) {
          throw new Error(
            `WASM SIMD runtime file failed to compile or download: ${probe.failedFile || wasmUrl}` +
              (probe.status ? ` (HTTP ${probe.status})` : "") +
              (probe.error ? ` (${probe.error})` : "")
          );
        }
      }
      const ort = await importOrt(cfg.entry, attempts.filter((a) => a.status === "failed").length ? String(Date.now()) : "");
      applyWasmFlags(ort, cfg.wasmPaths, { simd: spec.simd });
      const session = await ort.InferenceSession.create(buffer, {
        executionProviders: ["wasm"],
      });
      record.status = "ok";
      loadedOrt = ort;
      session._ownGptOrt = { id: spec.id, source: cfg.source, diagnostics: diags };
      return session;
    } catch (err) {
      lastErr = err;
      record.status = "failed";
      record.error = err && err.message ? err.message : String(err);
      console.warn(`ORT ${spec.id} failed:`, err);
    }
  }

  throw new Error(formatOrtError(lastErr, diags, attempts));
}

export async function loadOrt() {
  // Kept for generateNextTokens, which only needs the Tensor constructor.
  if (loadedOrt) return loadedOrt;
  const spec = wasmSimdSupported() ? ORT_PRIMARY : ORT_LEGACY;
  const cfg = await resolveConfig(spec);
  const ort = await importOrt(cfg.entry);
  applyWasmFlags(ort, cfg.wasmPaths, { simd: spec.simd });
  loadedOrt = ort;
  return ort;
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
  if (reply.toLowerCase().startsWith("assistant:")) {
    reply = reply.slice("assistant:".length);
  }
  reply = reply.split(/\nUser:/)[0];
  reply = reply.replace(/<\|endoftext\|>/g, "");
  return reply.trim();
}
