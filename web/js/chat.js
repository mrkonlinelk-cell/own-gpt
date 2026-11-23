import {
  DEFAULT_MAX_TOKENS,
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_K,
  DEFAULT_BLOCK_SIZE,
  MODEL_URL,
  TOKENIZER_URL,
  META_URL,
} from "./config.js";
import { BPETokenizer } from "./tokenizer.js";
import { createSession, generateNextTokens, buildChatPrompt, extractReply } from "./infer.js";

const els = {
  status: document.getElementById("status"),
  progress: document.getElementById("progress"),
  log: document.getElementById("log"),
  form: document.getElementById("composer"),
  input: document.getElementById("message"),
  send: document.getElementById("send"),
  stop: document.getElementById("stop"),
  reset: document.getElementById("reset"),
  temperature: document.getElementById("temperature"),
  temperatureVal: document.getElementById("temperature-val"),
  topk: document.getElementById("topk"),
  maxtokens: document.getElementById("maxtokens"),
  settingsToggle: document.getElementById("settings-toggle"),
  settings: document.getElementById("settings"),
};

let session = null;
let tokenizer = null;
let meta = { block_size: DEFAULT_BLOCK_SIZE };
let history = [];
let abortController = null;
let ready = false;

function setStatus(text, kind) {
  els.status.textContent = text;
  els.status.dataset.kind = kind || "";
}

function formatBytes(n) {
  if (!n) return "…";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function addBubble(role, text, streaming) {
  const wrap = document.createElement("div");
  wrap.className = `bubble bubble-${role}`;
  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = role === "user" ? "You" : "Assistant";
  const body = document.createElement("div");
  body.className = "bubble-body";
  body.textContent = text;
  wrap.append(label, body);
  els.log.append(wrap);
  els.log.scrollTop = els.log.scrollHeight;
  return body;
}

function syncControls() {
  els.temperatureVal.textContent = Number(els.temperature.value).toFixed(2);
  els.send.disabled = !ready || !els.input.value.trim() || !!abortController;
  els.stop.disabled = !abortController;
  els.input.disabled = !ready || !!abortController;
}

els.temperature.addEventListener("input", syncControls);
els.input.addEventListener("input", syncControls);
els.settingsToggle.addEventListener("click", () => {
  const open = els.settings.toggleAttribute("hidden") === false || !els.settings.hasAttribute("hidden");
  // toggleAttribute returns false if the attribute is now present (hidden).
  const isHidden = els.settings.hasAttribute("hidden");
  els.settingsToggle.setAttribute("aria-expanded", String(!isHidden));
  els.settingsToggle.textContent = isHidden ? "Settings" : "Hide settings";
});

els.reset.addEventListener("click", () => {
  history = [];
  els.log.replaceChildren();
  addBubble("assistant", "Chat history cleared. Say hello whenever you are ready.");
});

els.stop.addEventListener("click", () => {
  if (abortController) abortController.abort();
});

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = els.input.value.trim();
  if (!message || !ready || abortController) return;
  els.input.value = "";
  addBubble("user", message);
  const body = addBubble("assistant", "…");
  abortController = new AbortController();
  syncControls();
  try {
    const prompt = buildChatPrompt(history, message);
    const ids = tokenizer.encode(prompt);
    const result = await generateNextTokens(session, tokenizer, ids, {
      maxTokens: Number(els.maxtokens.value) || DEFAULT_MAX_TOKENS,
      temperature: Number(els.temperature.value) || DEFAULT_TEMPERATURE,
      topK: Number(els.topk.value) || DEFAULT_TOP_K,
      blockSize: meta.block_size || DEFAULT_BLOCK_SIZE,
      stopAtUser: true,
      signal: abortController.signal,
      onToken(_piece, text) {
        body.textContent = extractReply(text) || "…";
        els.log.scrollTop = els.log.scrollHeight;
      },
    });
    const reply = extractReply(result.text) || "…";
    body.textContent = reply;
    history.push({ user: message, assistant: reply });
    if (history.length > 4) history.shift();
  } catch (err) {
    body.textContent = err.message || String(err);
  } finally {
    abortController = null;
    syncControls();
    els.input.focus();
  }
});

async function loadMeta() {
  try {
    const res = await fetch(META_URL);
    if (res.ok) meta = { ...meta, ...(await res.json()) };
  } catch {
    // meta.json is optional
  }
}

async function boot() {
  setStatus("Loading the model into your browser…", "loading");
  els.progress.hidden = false;
  syncControls();
  try {
    await loadMeta();
    const tokProgress = (received, total) => {
      els.progress.value = total ? received / total : 0;
    };
    tokenizer = await BPETokenizer.load(TOKENIZER_URL, tokProgress);
    session = await createSession(MODEL_URL, (received, total) => {
      const label = total
        ? `Downloading model… ${formatBytes(received)} / ${formatBytes(total)}`
        : `Downloading model… ${formatBytes(received)}`;
      setStatus(label, "loading");
      if (total) {
        els.progress.value = received / total;
      }
    });
    const step = meta.step != null ? `step ${meta.step}` : "ready";
    const params = meta.n_params ? `${Number(meta.n_params).toLocaleString()} parameters` : "";
    const val = meta.val_loss != null ? `val loss ${Number(meta.val_loss).toFixed(3)}` : "";
    const runtime = session._ownGptOrt ? session._ownGptOrt.id : "";
    const bits = [step, params, val, runtime].filter(Boolean).join(" · ");
    setStatus(`Model is running in your browser. ${bits}`, "ready");
    els.progress.hidden = true;
    ready = true;
    syncControls();
    els.input.focus();
  } catch (err) {
    setStatus(err.message || String(err), "error");
    els.progress.hidden = true;
  }
}

boot();
