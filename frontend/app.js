/* FinAuditAI — веб-интерфейс (без сборки, plain JS) */
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
  return data;
}

function table(sel, rows) {
  const tbody = $(sel + " tbody");
  tbody.innerHTML = rows.join("");
}

/* ---------- Табы ---------- */
$$("nav button").forEach((b) =>
  b.addEventListener("click", () => {
    $$("nav button").forEach((x) => x.classList.remove("active"));
    $$(".tab").forEach((x) => x.classList.remove("active"));
    b.classList.add("active");
    $("#tab-" + b.dataset.tab).classList.add("active");
    if (b.dataset.tab === "history") loadRuns();
    if (b.dataset.tab === "training") { loadDatasets(); loadJobs(); }
    if (b.dataset.tab === "eval") loadEvalHistory();
  })
);

/* ---------- Модели ---------- */
let allModels = [];

async function loadModels() {
  try {
    const data = await api("/api/models");
    allModels = data.models;
    $("#providers").innerHTML = data.providers
      .map(
        (p) => `<div class="card">
          <b>${p.name}</b>
          <span class="badge ${p.health ? "ok" : "bad"}">${p.health ? "доступен" : "недоступен"}</span>
          <div class="hint">${p.kind === "local" ? "локальная" : "API"}</div>
        </div>`
      )
      .join("");
    table("#model-list", data.models.map((m) =>
      `<tr><td>${m.name}</td><td>${m.provider}</td>
       <td>${m.kind === "local" ? "локальная" : "API"}</td>
       <td>${m.can_train ? "✅" : "—"}</td></tr>`));
    $("#model-errors").textContent = data.errors.length ? "⚠ " + data.errors.join(" · ") : "";
    fillModelSelects();
  } catch (e) {
    $("#model-errors").textContent = "Ошибка загрузки: " + e.message;
  }
}

function fillModelSelects() {
  const opts = allModels.map((m) => `<option value="${m.provider}::${m.name}">${m.name}</option>`).join("");
  ["#run-provider", "#eval-provider"].forEach((s) => {
    const sel = $(s);
    if (sel) {
      sel.innerHTML = opts;
      sel.onchange = () => fillModelsFor(sel, s === "#run-provider" ? "#run-model" : "#eval-model");
      fillModelsFor(sel, s === "#run-provider" ? "#run-model" : "#eval-model");
    }
  });
  const train = $("#train-model");
  if (train) train.innerHTML = allModels.filter((m) => m.can_train)
    .map((m) => `<option value="${m.provider}::${m.name}">${m.name}</option>`).join("") || "<option>Нет обучаемых моделей</option>";
}

function fillModelsFor(providerSel, modelSel) {
  const provider = providerSel.value.split("::")[0];
  $(modelSel).innerHTML = allModels.filter((m) => m.provider === provider)
    .map((m) => `<option>${m.name}</option>`).join("");
}

$("#connect-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  try {
    const r = await api("/api/models/connect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(f)),
    });
    $("#connect-result").textContent = `✅ Подключено, найдено моделей: ${r.models_found}`;
    loadModels();
  } catch (err) {
    $("#connect-result").textContent = "❌ " + err.message;
  }
});

/* ---------- Тренировка ---------- */
async function loadDatasets() {
  const d = await api("/api/training/datasets");
  $("#dataset-list").innerHTML = d.datasets.length
    ? "Файлы: " + d.datasets.map((x) => `${x.name} (${x.lines ?? "?"} стр.)`).join(", ")
    : "Датасетов пока нет — загрузи JSONL";
}

$("#dataset-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  try {
    await api("/api/training/datasets", { method: "POST", body: fd });
    loadDatasets();
  } catch (err) {
    $("#dataset-list").textContent = "❌ " + err.message;
  }
});

$("#start-training").addEventListener("click", async () => {
  const v = $("#train-model").value;
  if (!v) return alert("Нет обучаемых моделей");
  const ds = $("#dataset-list").textContent.match(/Файлы: ([^(]+)/);
  const dataset = ds ? ds[1].trim() : prompt("Имя файла датасета в data/datasets:");
  const [provider, model] = v.split("::");
  try {
    await api("/api/training/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model, dataset }),
    });
    loadJobs();
  } catch (e) { alert("❌ " + e.message); }
});

async function loadJobs() {
  const d = await api("/api/training/jobs");
  table("#jobs", d.jobs.map((j) =>
    `<tr data-id="${j.id}" style="cursor:pointer">
      <td>${j.id}</td><td>${j.model}</td><td>${j.dataset}</td>
      <td>${statusBadge(j.status)}</td><td>${Math.round(j.progress * 100)}%</td></tr>`));
  $$("#jobs tbody tr").forEach((tr) =>
    tr.addEventListener("click", async () => {
      const j = await api("/api/training/jobs/" + tr.dataset.id);
      $("#job-logs").textContent = j.logs || "(пусто)";
      $("#job-logs").scrollTop = $("#job-logs").scrollHeight;
    }));
  if (d.jobs.some((j) => j.status === "running" || j.status === "queued"))
    setTimeout(loadJobs, 3000);
}

function statusBadge(s) {
  const cls = s === "done" ? "ok" : s === "failed" ? "bad" : "";
  return `<span class="badge ${cls}">${s}</span>`;
}

/* ---------- Прогон файлов ---------- */
$("#start-run").addEventListener("click", async () => {
  const file = $("#run-file").files[0];
  if (!file) return alert("Выбери файл");
  const [provider] = $("#run-provider").value.split("::");
  const model = $("#run-model").value;
  const fd = new FormData();
  fd.append("file", file);
  $("#run-status").textContent = "⏳ Загрузка и запуск...";
  try {
    const r = await api(`/api/runs?provider=${encodeURIComponent(provider)}&model=${encodeURIComponent(model)}`,
      { method: "POST", body: fd });
    pollRun(r.run_id);
  } catch (e) {
    $("#run-status").textContent = "❌ " + e.message;
  }
});

function pollRun(id) {
  const t = setInterval(async () => {
    const r = await api("/api/runs/" + id);
    if (r.status === "running") { $("#run-status").textContent = "⏳ Модель анализирует файл..."; return; }
    clearInterval(t);
    if (r.status === "failed") {
      $("#run-status").textContent = "❌ " + r.error;
      return;
    }
    $("#run-status").textContent = `✅ Готово: ${r.result.chunks} фрагментов, ${r.result.chars} символов`;
    const f = r.result.findings || [];
    $("#run-result").innerHTML = f.length
      ? `<h3>Найденные ошибки (${f.length})</h3><table class="tbl"><thead>
         <tr><th>Тип</th><th>ID</th><th>Сумма</th><th>Валюта</th><th>Доказательство</th></tr></thead><tbody>` +
        f.map((x) => `<tr><td>${x.type || "?"}</td><td>${(x.affected_ids || []).join(", ")}</td>
          <td>${x.amount ?? ""}</td><td>${x.currency || ""}</td><td>${x.evidence || ""}</td></tr>`).join("") +
        `</tbody></table>`
      : `<p class="hint">Ошибок не найдено.</p>`;
  }, 1500);
}

/* ---------- Eval FinRepGYM ---------- */
$("#setup-checker").addEventListener("click", async () => {
  $("#eval-status").textContent = "⏳ Клонирование FinRepGYM и проверка...";
  try {
    const r = await api("/api/eval/setup", { method: "POST" });
    $("#eval-status").textContent = (r.ok ? "✅ Проверщик готов: " : "⚠ ") + r.path + "\n" + r.stdout.slice(-500);
  } catch (e) { $("#eval-status").textContent = "❌ " + e.message; }
});

$("#start-eval").addEventListener("click", async () => {
  const [provider] = $("#eval-provider").value.split("::");
  const model = $("#eval-model").value;
  const split = $("#eval-split").value;
  $("#eval-status").textContent = "⏳ Прогон через эпизоды FinRepGYM...";
  try {
    const m = await api("/api/eval/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider, model, split, max_episodes: 10 }),
    });
    $("#eval-status").textContent = `✅ Оценено эпизодов: ${m.episodes}`;
    $("#eval-result").innerHTML = `<div class="cards">
      <div class="card"><b>Precision</b><div class="big">${m.precision}</div></div>
      <div class="card"><b>Recall</b><div class="big">${m.recall}</div></div>
      <div class="card"><b>F1</b><div class="big">${m.f1}</div></div>
      <div class="card"><b>Reward</b><div class="big">${m.reward}</div></div>
      <div class="card"><b>Шаги</b><div class="big">${m.steps}</div></div>
    </div>`;
    loadEvalHistory();
  } catch (e) { $("#eval-status").textContent = "❌ " + e.message; }
});

async function loadEvalHistory() {
  try {
    const d = await api("/api/eval/history");
    table("#eval-history", d.results.map((r) =>
      `<tr><td>${r.model}</td><td>${r.split}</td><td>${r.precision}</td>
       <td>${r.recall}</td><td>${r.f1}</td><td>${r.reward}</td></tr>`));
  } catch { /* ignore */ }
}

/* ---------- История ---------- */
async function loadRuns() {
  const d = await api("/api/runs");
  table("#runs", d.runs.map((r) =>
    `<tr data-id="${r.id}" style="cursor:pointer"><td>${r.id}</td><td>${r.file}</td>
     <td>${r.model}</td><td>${r.findings_count}</td><td>${statusBadge(r.status)}</td></tr>`));
  $$("#runs tbody tr").forEach((tr) =>
    tr.addEventListener("click", async () => {
      const r = await api("/api/runs/" + tr.dataset.id);
      $("#run-detail").innerHTML = `<h3>Прогон #${r.id}</h3>
        <p class="hint">${r.file} → ${r.model}</p>
        <pre class="logs">${JSON.stringify(r.result, null, 2).slice(0, 8000)}</pre>`;
    }));
}

loadModels();
