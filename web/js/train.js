import { REPO_OWNER, REPO_NAME } from "./config.js";

const TOKEN_KEY = "own-gpt.githubToken";
const API = "https://api.github.com";

const els = {
  token: document.getElementById("token"),
  saveToken: document.getElementById("save-token"),
  clearToken: document.getElementById("clear-token"),
  tokenStatus: document.getElementById("token-status"),
  text: document.getElementById("text"),
  file: document.getElementById("file"),
  filename: document.getElementById("filename"),
  minutes: document.getElementById("minutes"),
  mode: document.getElementById("mode"),
  start: document.getElementById("start"),
  formStatus: document.getElementById("form-status"),
  runs: document.getElementById("runs"),
  refresh: document.getElementById("refresh"),
};

function getToken() {
  return (localStorage.getItem(TOKEN_KEY) || "").trim();
}

function setTokenStatus() {
  const has = Boolean(getToken());
  els.tokenStatus.textContent = has
    ? "A token is saved in this browser only. It is not stored in the repo or on the website."
    : "No token saved yet. Training from this page needs one.";
  els.tokenStatus.dataset.kind = has ? "ready" : "";
  if (has && !els.token.value) els.token.placeholder = "Token saved — paste a new one to replace it";
}

els.saveToken.addEventListener("click", () => {
  const value = els.token.value.trim();
  if (!value) {
    els.tokenStatus.textContent = "Paste a token first.";
    els.tokenStatus.dataset.kind = "error";
    return;
  }
  localStorage.setItem(TOKEN_KEY, value);
  els.token.value = "";
  setTokenStatus();
  loadRuns();
});

els.clearToken.addEventListener("click", () => {
  localStorage.removeItem(TOKEN_KEY);
  els.token.value = "";
  els.token.placeholder = "github_pat_…";
  setTokenStatus();
});

async function github(path, options = {}) {
  const token = getToken();
  if (!token) throw new Error("Save a GitHub token first.");
  const res = await fetch(`${API}${path}`, {
    ...options,
    headers: {
      Accept: "application/vnd.github+json",
      Authorization: `Bearer ${token}`,
      "X-GitHub-Api-Version": "2022-11-28",
      ...(options.body ? { "Content-Type": "application/json" } : {}),
      ...(options.headers || {}),
    },
  });
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { raw: text };
    }
  }
  if (!res.ok) {
    const msg = (data && (data.message || data.error)) || res.statusText;
    throw new Error(`${res.status}: ${msg}`);
  }
  return data;
}

function stampFilename(original) {
  const now = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  let base = (original || "notes").replace(/\.txt$/i, "");
  base = base.replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "") || "notes";
  return `${base}-${stamp}.txt`;
}

function toBase64(text) {
  const bytes = new TextEncoder().encode(text);
  let binary = "";
  bytes.forEach((b) => {
    binary += String.fromCharCode(b);
  });
  return btoa(binary);
}

async function readTrainingText() {
  const parts = [];
  if (els.file.files && els.file.files.length) {
    for (const file of els.file.files) {
      const text = (await file.text()).trim();
      if (text) parts.push(text);
    }
  }
  const pasted = els.text.value.trim();
  if (pasted) parts.push(pasted);
  return parts.join("\n\n<|endoftext|>\n\n");
}

async function createTextFile(filename, content) {
  const path = `data/my_text/${filename}`;
  return github(`/repos/${REPO_OWNER}/${REPO_NAME}/contents/${encodeURIComponent(path).replaceAll("%2F", "/")}`, {
    method: "PUT",
    body: JSON.stringify({
      message: `Add training text ${filename} from the web app`,
      content: toBase64(content),
      branch: "main",
    }),
  });
}

async function dispatchTrain(minutes, mode) {
  return github(`/repos/${REPO_OWNER}/${REPO_NAME}/actions/workflows/train.yml/dispatches`, {
    method: "POST",
    body: JSON.stringify({
      ref: "main",
      inputs: {
        minutes: String(minutes),
        mode,
      },
    }),
  });
}

function formatRunTime(iso) {
  if (!iso) return "";
  try {
    return new Date(iso).toLocaleString();
  } catch {
    return iso;
  }
}

function renderRuns(runs) {
  if (!runs || !runs.length) {
    els.runs.innerHTML = "<p class='muted'>No training runs yet.</p>";
    return;
  }
  const html = [
    "<table>",
    "<thead><tr><th>Status</th><th>Mode / time</th><th>Started</th><th>Link</th></tr></thead>",
    "<tbody>",
  ];
  for (const run of runs) {
    const badge = `<span class="badge badge-${run.conclusion || run.status}">${run.conclusion || run.status}</span>`;
    const title = escapeHtml(run.display_title || run.name || "Train");
    html.push(
      `<tr>
        <td>${badge}</td>
        <td>${title}</td>
        <td>${escapeHtml(formatRunTime(run.run_started_at || run.created_at))}</td>
        <td><a href="${run.html_url}" target="_blank" rel="noopener">Open run</a></td>
      </tr>`
    );
  }
  html.push("</tbody></table>");
  els.runs.innerHTML = html.join("");
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadRuns() {
  if (!getToken()) {
    els.runs.innerHTML = "<p class='muted'>Save a token to see recent training runs.</p>";
    return;
  }
  els.runs.innerHTML = "<p class='muted'>Loading runs…</p>";
  try {
    const data = await github(
      `/repos/${REPO_OWNER}/${REPO_NAME}/actions/workflows/train.yml/runs?per_page=10`
    );
    renderRuns(data.workflow_runs || []);
  } catch (err) {
    els.runs.innerHTML = `<p class="error">Could not load runs: ${escapeHtml(err.message)}</p>`;
  }
}

els.refresh.addEventListener("click", () => loadRuns());

document.getElementById("train-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  els.formStatus.textContent = "";
  els.formStatus.dataset.kind = "";
  els.start.disabled = true;
  try {
    const content = await readTrainingText();
    let minutes = Number(els.minutes.value);
    if (!Number.isFinite(minutes)) minutes = 60;
    minutes = Math.max(1, Math.min(330, Math.round(minutes)));
    const mode = els.mode.value === "fresh" ? "fresh" : "continue";

    if (content) {
      const name = stampFilename(els.filename.value || (els.file.files[0] && els.file.files[0].name) || "web-text");
      els.formStatus.textContent = `Uploading ${name} to data/my_text/…`;
      await createTextFile(name, content);
    } else {
      els.formStatus.textContent = "No new text added — starting a run on the current data…";
    }

    els.formStatus.textContent = `Starting a ${minutes}-minute ${mode} training run…`;
    await dispatchTrain(minutes, mode);
    els.formStatus.textContent =
      "Training has been queued on GitHub Actions. Open a run below to watch it. When it finishes, the chat page will use the new model after the site redeploys.";
    els.formStatus.dataset.kind = "ready";
    els.text.value = "";
    els.file.value = "";
    setTimeout(loadRuns, 1500);
  } catch (err) {
    els.formStatus.textContent = err.message || String(err);
    els.formStatus.dataset.kind = "error";
  } finally {
    els.start.disabled = false;
  }
});

setTokenStatus();
loadRuns();
