#!/usr/bin/env node
/**
 * Copy (or download) the ONNX Runtime Web files the chat page needs:
 *   - 1.21 WASM-only SIMD build (primary)
 *   - 1.18 non-SIMD wasm (fallback for older WebViews)
 *
 * Usage: node scripts/vendor-ort.mjs --dest web/js/ort
 */
import { copyFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const destArg = process.argv.includes("--dest")
  ? process.argv[process.argv.indexOf("--dest") + 1]
  : "web/js/ort";
const dest = resolve(process.cwd(), destArg);

const FILES = [
  {
    dest: "ort.wasm.min.mjs",
    local: join(root, "android-app/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs"),
    altLocal: join(root, "node_modules/onnxruntime-web/dist/ort.wasm.min.mjs"),
    altLocal2: "/workspace/own-gpt-android/node_modules/onnxruntime-web/dist/ort.wasm.min.mjs",
    url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.wasm.min.mjs",
  },
  {
    dest: "ort-wasm-simd-threaded.mjs",
    local: join(root, "android-app/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs"),
    altLocal: join(root, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs"),
    altLocal2: "/workspace/own-gpt-android/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.mjs",
    url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort-wasm-simd-threaded.mjs",
  },
  {
    dest: "ort-wasm-simd-threaded.wasm",
    local: join(root, "android-app/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"),
    altLocal: join(root, "node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm"),
    altLocal2: "/workspace/own-gpt-android/node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm",
    url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort-wasm-simd-threaded.wasm",
  },
  {
    dest: "legacy/ort.wasm.min.js",
    url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/esm/ort.wasm.min.js",
  },
  {
    dest: "legacy/ort-wasm.wasm",
    url: "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/ort-wasm.wasm",
  },
];

async function download(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GET ${url} -> ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

mkdirSync(join(dest, "legacy"), { recursive: true });
for (const file of FILES) {
  const out = join(dest, file.dest);
  mkdirSync(dirname(out), { recursive: true });
  const src = [file.local, file.altLocal, file.altLocal2].find((p) => p && existsSync(p));
  if (src) {
    copyFileSync(src, out);
    console.log(`copied ${file.dest} <- ${src}`);
  } else {
    const buf = await download(file.url);
    writeFileSync(out, buf);
    console.log(`downloaded ${file.dest} (${(buf.length / 1024 / 1024).toFixed(2)} MB) <- ${file.url}`);
  }
}
console.log("Vendored ORT into", dest);
