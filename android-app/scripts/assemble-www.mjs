#!/usr/bin/env node
/**
 * Build the Capacitor www/ folder from the own-gpt web app, the published
 * ONNX model, and a vendored copy of onnxruntime-web (so chat works offline).
 */
import { cpSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repoCandidates = [
  process.env.OWN_GPT_ROOT,
  resolve(root, ".."),                 // android-app/ inside the git repo
  resolve(root, "../own-gpt-repo"),    // sibling folder used on this machine
].filter(Boolean);
const repo = repoCandidates.find(
  (c) => existsSync(join(c, "web/index.html")) && existsSync(join(c, "model/model.onnx"))
);
const www = join(root, "www");
const ortSrc = join(root, "node_modules/onnxruntime-web/dist");

if (!repo) {
  throw new Error(`Cannot find the own-gpt repo (web/ + model/) near ${root}`);
}
if (!existsSync(join(ortSrc, "ort.min.mjs"))) {
  throw new Error("onnxruntime-web is not installed. Run npm install.");
}

rmSync(www, { recursive: true, force: true });
mkdirSync(join(www, "model"), { recursive: true });
mkdirSync(join(www, "js/ort"), { recursive: true });

cpSync(join(repo, "web"), www, { recursive: true });
// Overlay model files used by the chat page (paths in js/config.js are ../model/...)
for (const name of ["model.onnx", "tokenizer.json", "meta.json"]) {
  cpSync(join(repo, "model", name), join(www, "model", name));
}

// Vendored ORT: wasm + ESM loader. Skip the 23MB WebGPU/jsep build.
for (const name of [
  "ort.min.mjs",
  "ort-wasm-simd-threaded.mjs",
  "ort-wasm-simd-threaded.wasm",
]) {
  cpSync(join(ortSrc, name), join(www, "js/ort", name));
}

console.log("Assembled", www);
