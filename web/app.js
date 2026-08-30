/* Document Packet Intelligence — UI logic.
   No framework and no build step: the backend serves this file directly, so the whole app runs
   from a `uvicorn` command with nothing else installed. */
"use strict";

const $ = (id) => document.getElementById(id);
const api = (path, options) => fetch(path, options).then(async (response) => {
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `${response.status} ${response.statusText}`);
  return body;
});

/* Backend strings (document types, evidence text) are rendered as HTML, so escape them. */
const esc = (value) => String(value ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (value) => `${(Number(value) * 100).toFixed(1)}%`;
const num = (value, places = 3) => Number(value).toFixed(places);
const titleCase = (value) => String(value ?? "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const kb = (bytes) => bytes >= 1048576 ? `${(bytes / 1048576).toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;

let currentJob = null;
let selectedFile = null;

/* ------------------------------------------------------------------ status */
async function loadStatus() {
  const dot = document.querySelector("#status .dot");
  try {
    const health = await api("/api/health");
    const ready = health.boundary_ready && health.classifier_ready;
    dot.dataset.state = ready ? "ok" : "warn";
    $("status-text").textContent = ready
      ? `models ready · ${health.encoder} encoder`
      : "running on fallbacks";
    $("status").title = `Boundary: ${health.boundary_model}\nClassifier: ${health.classifier_model}\n` +
      `Decision rule: ${health.decision_rule}\nRerank: ${health.rerank}`;
    if (!health.sample_available) $("btn-sample").disabled = true;
  } catch (error) {
    dot.dataset.state = "bad";
    $("status-text").textContent = "backend unreachable";
  }
}

async function loadBenchmarks() {
  try {
    const data = await api("/api/benchmarks");
    $("bench-note").textContent = data.note || "";
    $("benchmarks").innerHTML = ["boundary", "classification", "retrieval"].map((key) => {
      const group = data[key];
      if (!group) return "";
      const rows = group.metrics.map((metric) => `
        <div class="brow ${metric.baseline ? "baseline" : ""}">
          <span class="n">${esc(metric.name)}<span class="d">${esc(metric.dataset)}</span></span>
          <span class="val">${num(metric.value)}</span>
        </div>`).join("");
      return `<div class="bgroup"><h3>${esc(group.label)}</h3>${rows}</div>`;
    }).join("");
  } catch (error) {
    $("benchmarks").innerHTML = `<p class="empty">Benchmarks unavailable: ${esc(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------------ upload */
function chooseFile(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith(".pdf")) return showError("Please choose a PDF file.");
  selectedFile = file;
  const zone = $("dropzone");
  zone.classList.add("has-file");
  zone.querySelector(".dz-main").textContent = file.name;
  zone.querySelector(".dz-sub").textContent = `${kb(file.size)} · ready to process`;
  $("btn-upload").disabled = false;
  hideError();
}

function setBusy(busy, message) {
  $("progress").hidden = !busy;
  $("progress-text").textContent = message || "Processing…";
  $("btn-upload").disabled = busy || !selectedFile;
  $("btn-sample").disabled = busy;
}

const showError = (message) => { $("error").hidden = false; $("error").textContent = message; };
const hideError = () => { $("error").hidden = true; };

async function processUpload() {
  if (!selectedFile) return;
  const form = new FormData();
  form.append("file", selectedFile);
  const truth = $("truth").value.trim();
  if (truth) form.append("ground_truth", truth);
  await runJob(() => api("/api/process", { method: "POST", body: form }),
    `Splitting ${selectedFile.name}…`);
}

async function processSample() {
  await runJob(() => api("/api/process-sample", { method: "POST" }),
    "Processing the bundled sample packet…");
}

async function runJob(request, message) {
  hideError();
  setBusy(true, message);
  try {
    currentJob = await request();
    render(currentJob);
    $("panel-results").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message);
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ rendering */
function render(job) {
  ["panel-results", "panel-documents", "panel-search"].forEach((id) => { $(id).hidden = false; });

  $("dl-all").href = job.archive;
  $("dl-source").href = job.source_pdf;

  $("summary").innerHTML = `
    ${tile("Pages", job.page_count, esc(job.filename))}
    ${tile("Documents found", job.document_count, "split from one PDF")}
    ${tile("Searchable chunks", job.chunk_count, "indexed for retrieval")}
    ${tile("Processing time", `${job.elapsed_seconds}s`, "on CPU")}`;

  renderSeams(job);
  renderEvaluation(job);
  renderDocuments(job);
  renderSearchExamples(job);
  $("search-results").innerHTML = "";
}

const tile = (key, value, sub) =>
  `<div class="tile"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div><div class="s">${sub}</div></div>`;

/* Stage 1 is a pairwise decision: for every seam between two adjacent pages it scores how likely
   a new document starts there. This is the only probability in the app that is about how two
   pages relate — the per-document number is about type, not adjacency. */
function renderSeams(job) {
  const pairs = job.page_pairs || [];
  const target = $("seams");
  if (!pairs.length) { target.innerHTML = ""; return; }

  const cells = pairs.map((pair) => `
    <div class="seam ${pair.split ? "split" : ""}"
         title="Pages ${pair.from} and ${pair.to}: ${pct(pair.probability)} chance a new document starts at page ${pair.to}">
      <div class="seam-bar"><span style="height:${Math.max(4, pair.probability * 100)}%"></span></div>
      <div class="seam-p">${num(pair.probability, 2)}</div>
      <div class="seam-n">${pair.from}<span>|</span>${pair.to}</div>
    </div>`).join("");

  const rule = job.decision_rule === "expected_count"
    ? `Because the probabilities are calibrated, their sum estimates how many boundaries the packet
       contains: <strong>${num(job.expected_boundaries, 2)}</strong> → <strong>${Math.round(job.expected_boundaries)}</strong>
       split${Math.round(job.expected_boundaries) === 1 ? "" : "s"}, taken at the highest-scoring seams.
       No fixed cut-off is applied, so the rule adapts to packets far denser or sparser than the training set.`
    : `A fixed threshold decides each seam independently.`;

  target.innerHTML = `
    <div class="seambox">
      <header>How the split was decided
        <span class="sub">${pairs.length} seam${pairs.length === 1 ? "" : "s"} between ${job.page_count} pages</span>
      </header>
      <p class="small muted seam-help">
        Every pair of neighbouring pages gets one score: <em>how likely is it that a new document
        starts on the second page?</em> Low means the two pages belong together.
      </p>
      <div class="seamstrip">${cells}</div>
      <p class="small seam-rule">${rule}</p>
    </div>`;
}

function renderEvaluation(job) {
  const target = $("evaluation");
  const evaluation = job.evaluation;

  if (!evaluation) {
    target.innerHTML = `<div class="notice">
      <strong>No accuracy or precision shown for this packet.</strong> Those need ground truth —
      the real answer to compare against — and none was supplied. The confidence scores on each
      document below are the model's own certainty, which is a different thing. Add ground truth
      in the upload panel, or try the bundled sample, to see measured accuracy.</div>`;
    return;
  }

  const classification = evaluation.classification || {};
  const boundary = evaluation.boundary || {};
  const rows = (evaluation.documents || []).map((doc) => `
    <tr>
      <td>${doc.actual_pages.join(", ")}</td>
      <td>${esc(titleCase(doc.actual_type))}</td>
      <td>${esc(titleCase(doc.predicted_type))}</td>
      <td>${pct(doc.confidence)}</td>
      <td>${doc.type_correct ? '<span class="tick">correct</span>' : '<span class="cross">wrong</span>'}</td>
      <td>${doc.pages_exact ? '<span class="tick">exact</span>' : '<span class="cross">differs</span>'}</td>
    </tr>`).join("");

  target.innerHTML = `
    <div class="evalbox">
      <header>Measured against your ground truth
        <span class="sub">${evaluation.documents_predicted} predicted vs
        ${evaluation.documents_actual} actual documents across ${evaluation.page_count} pages</span>
      </header>
      <div class="metrics">
        ${metric("Accuracy", pct(classification.accuracy), "document type, per document")}
        ${metric("Precision", num(classification.macro_precision), "macro-averaged")}
        ${metric("Recall", num(classification.macro_recall), "macro-averaged")}
        ${metric("F1", num(classification.macro_f1), "macro-averaged")}
        ${metric("Page grouping", num(evaluation.page_grouping_accuracy), "pairwise agreement")}
        ${metric("Boundary F1", num(boundary.f1), `vs ${num(boundary.trivial_f1)} trivial`)}
      </div>
      <table class="cmp">
        <thead><tr><th>Pages</th><th>Actual type</th><th>Predicted</th>
        <th>Confidence</th><th>Type</th><th>Split</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    ${boundaryNote(boundary, evaluation)}`;
}

const metric = (key, value, sub) =>
  `<div class="metric"><div class="k">${esc(key)}</div><div class="v">${esc(value)}</div><div class="s">${esc(sub)}</div></div>`;

/* The trivial baseline is reported alongside boundary F1 because F1 alone is not comparable
   across packets with different boundary densities — a point the project's own evaluation
   turned on, so the UI should not quietly drop it. */
function boundaryNote(boundary, evaluation) {
  if (boundary.f1 === undefined) return "";
  const lift = Number(boundary.lift_over_trivial);
  const beat = lift > 0;
  return `<div class="notice">
    <strong>${evaluation.documents_exactly_split ?? 0} of ${evaluation.documents_actual}</strong>
    documents were split at exactly the right pages. Boundary F1 is
    <strong>${num(boundary.f1)}</strong> against a trivial "every page is its own document"
    baseline of <strong>${num(boundary.trivial_f1)}</strong> —
    ${beat ? `a lift of <strong>+${num(lift)}</strong>` :
      `<strong>${num(lift)}</strong>, i.e. below that baseline`}
    at a boundary density of ${pct(boundary.base_rate)}.</div>`;
}

/* Two different scorers can produce the confidence on a card, and their numbers are NOT on the
   same scale: the trained model spreads probability over 15 classes (chance = 0.07), while the
   keyword lexicon knows only 4 and its score is mostly "how much evidence did I find". Showing
   both as a bare percentage invites a false comparison, so every card names its scorer. */
const SCORERS = {
  trained: ["trained model",
    "Probability across 15 document classes, so it is shared among all of them — 7% would be pure chance. Not comparable with a keyword score."],
  lexicon_extension: ["keyword evidence",
    "This type has no class in the trained taxonomy, so weighted identifying phrases decide it. The score reflects how much evidence was found among 4 keyword classes — a much easier field than 16."],
  lexicon_backoff: ["keyword fallback",
    "The trained model came in under the confidence floor, so weighted keyword evidence was used instead."],
  lexicon_only: ["keyword only", "No trained model was loaded."],
  abstained_low_confidence: ["abstained", "Nothing cleared the confidence floor, so no type is claimed."],
  abstained_no_trained_model: ["abstained", "No trained model, and keyword evidence was too weak."],
  lexicon_no_evidence: ["no evidence", "No identifying phrases were found at all."],
};

function renderDocuments(job) {
  $("doc-count").textContent = `${job.document_count} document${job.document_count === 1 ? "" : "s"}`;
  $("documents").innerHTML = job.documents.map((doc) => {
    const level = doc.confidence >= 0.6 ? "" : doc.confidence >= 0.35 ? "mid" : "low";
    const [scorer, help] = SCORERS[doc.classification_source] || [doc.classification_source, ""];
    const runner = (doc.alternatives || [])
      .filter((a) => a.label !== doc.doc_type && a.probability > 0)[0];
    const unknown = doc.doc_type === "unknown";
    const elements = Object.entries(doc.elements || {})
      .map(([type, count]) => `${count} ${type}${count === 1 ? "" : "s"}`).join(" · ");
    const headings = (doc.headings || []).length
      ? `<div class="headings">${doc.headings.map((h) => `<span>${esc(h)}</span>`).join("")}</div>` : "";
    const thumb = doc.pages.length
      ? `<div class="thumb" style="background-image:url('/api/jobs/${esc(job.job_id)}/pages/${doc.pages[0]}')"></div>` : "";
    return `
      <article class="card">
        ${thumb}
        <div class="body">
          <div class="row">
            <span class="badge ${unknown ? "unknown" : ""}">${esc(titleCase(doc.doc_type))}</span>
            <span class="pages">Page${doc.pages.length === 1 ? "" : "s"} ${doc.pages.join(", ")}</span>
          </div>
          <div class="conf">
            <span>Confidence</span>
            <span class="track"><span class="lvl ${level}" style="width:${Math.max(3, doc.confidence * 100)}%"></span></span>
            <strong>${pct(doc.confidence)}</strong>
          </div>
          <div class="scale" title="${esc(help)}">
            scored by <b>${esc(scorer)}</b>${runner
              ? ` · next best ${esc(titleCase(runner.label))} ${pct(runner.probability)}` : ""}
          </div>
          <div class="meta">${doc.section_count} section${doc.section_count === 1 ? "" : "s"} ·
            ${doc.chunk_count} chunk${doc.chunk_count === 1 ? "" : "s"}${elements ? " · " + esc(elements) : ""}</div>
          ${headings}
        </div>
        <div class="foot">
          <a class="btn primary small" href="${esc(doc.download)}" download>Download PDF</a>
          <span class="size">${esc(doc.filename || "")} · ${kb(doc.bytes || 0)}</span>
        </div>
      </article>`;
  }).join("");
}

function renderSearchExamples(job) {
  const byType = {
    invoice: "What is the invoice total?",
    bank_statement: "What is the closing balance?",
    resume: "What technical skills are listed?",
    passport: "What is the passport number?",
    contract: "What are the termination terms?",
  };
  const seen = new Set();
  const examples = [];
  job.documents.forEach((doc) => {
    const question = byType[doc.doc_type];
    if (question && !seen.has(question)) { seen.add(question); examples.push(question); }
  });
  if (!examples.length) examples.push("What is the total amount?", "Who is this document addressed to?");
  $("search-examples").innerHTML = examples
    .map((q) => `<button class="chip" type="button">${esc(q)}</button>`).join("");
  $("search-examples").querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => { $("search-input").value = chip.textContent; runSearch(); });
  });
}

async function runSearch() {
  if (!currentJob) return;
  const query = $("search-input").value.trim();
  if (!query) return;
  const target = $("search-results");
  target.innerHTML = `<p class="empty">Searching…</p>`;
  try {
    const data = await api(`/api/jobs/${currentJob.job_id}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_k: 5 }),
    });
    target.innerHTML = data.results.length ? data.results.map((hit, index) => `
      <div class="hit">
        <div class="top">
          <span class="rank">${index + 1}</span>
          <span class="cite">${esc(titleCase(hit.doc_type || "document"))} ·
            page${hit.page_ref.length === 1 ? "" : "s"} ${hit.page_ref.join(", ")}</span>
          <span class="score">confidence ${pct(hit.confidence)}</span>
        </div>
        <div class="text">${esc(hit.evidence)}</div>
      </div>`).join("")
      : `<p class="empty">Nothing matched that query in this packet.</p>`;
  } catch (error) {
    target.innerHTML = `<p class="empty">Search failed: ${esc(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------------ wiring */
function init() {
  const zone = $("dropzone");
  const input = $("file-input");

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
  });
  input.addEventListener("change", () => chooseFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault(); zone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((name) => zone.addEventListener(name, (event) => {
    event.preventDefault(); zone.classList.remove("dragging");
  }));
  zone.addEventListener("drop", (event) => chooseFile(event.dataTransfer.files[0]));

  $("btn-upload").addEventListener("click", processUpload);
  $("btn-sample").addEventListener("click", processSample);
  $("search-form").addEventListener("submit", (event) => { event.preventDefault(); runSearch(); });

  /* Highlight whichever panel is on screen. */
  const links = [...document.querySelectorAll("#nav a")];
  const observer = new IntersectionObserver((entries) => {
    entries.filter((e) => e.isIntersecting).forEach((entry) => {
      links.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`));
    });
  }, { rootMargin: "-76px 0px -60% 0px" });
  ["panel-upload", "panel-documents", "panel-search", "panel-benchmarks"]
    .forEach((id) => observer.observe($(id)));

  loadStatus();
  loadBenchmarks();
}

document.addEventListener("DOMContentLoaded", init);
