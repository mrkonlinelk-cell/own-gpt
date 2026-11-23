#!/usr/bin/env node
/**
 * Build the Capacitor www/ folder from the own-gpt web app, the published
 * ONNX model, and vendored onnxruntime-web files (chat works offline).
 */
import { cpSync, mkdirSync, rmSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

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

if (!repo) {
  throw new Error(`Cannot find the own-gpt repo (web/ + model/) near ${root}`);
}

rmSync(www, { recursive: true, force: true });
mkdirSync(join(www, "model"), { recursive: true });

cpSync(join(repo, "web"), www, { recursive: true });
for (const name of ["model.onnx", "tokenizer.json", "meta.json"]) {
  cpSync(join(repo, "model", name), join(www, "model", name));
}

const vendor = spawnSync(process.execPath, [join(repo, "scripts/vendor-ort.mjs"), "--dest", join(www, "js/ort")], {
  stdio: "inherit",
  cwd: repo,
});
if (vendor.status !== 0) {
  throw new Error("vendor-ort.mjs failed");
}

console.log("Assembled", www);
