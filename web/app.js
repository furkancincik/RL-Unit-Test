"use strict";

const API_ROOT = "/api/v1";
const ENDPOINTS = Object.freeze({
  inline: "/api/v1/jobs/inline",
  upload: "/api/v1/jobs/upload",
  github: "/api/v1/jobs/github",
});
const TERMINAL_STATUSES = new Set([
  "COMPLETED",
  "PARTIAL",
  "FAILED",
  "TIMED_OUT",
  "CANCELLED",
]);
const MAX_SOURCE_BYTES = 2000000;
const MAX_RETRY_ATTEMPTS = 5;
const POLL_INTERVAL_MS = 1000;
const GITHUB_TARGET_ACKNOWLEDGEMENT_STALE =
  "Hedef seçimi değişti; current target kümesini yeniden onaylayın.";

const state = {
  activeSource: "inline",
  sourcePolicies: {
    inline: "STATIC_DISCOVERY_ONLY",
    upload: "STATIC_DISCOVERY_ONLY",
  },
  githubTrust: null,
  acknowledgedTargetFingerprint: null,
  currentSubmission: null,
  selectedFile: null,
  currentJobId: null,
  currentStatus: null,
  submitInFlight: false,
  activeJobInFlight: false,
  pollTimer: null,
  pollAbortController: null,
  terminalAbortController: null,
  pollGeneration: 0,
  retryAttempt: 0,
  createdAt: null,
  coverageProgressRevision: -1,
};

const byId = (id) => document.getElementById(id);
const sourceTabs = Array.from(document.querySelectorAll("[data-source-tab]"));
const sourcePanels = Array.from(document.querySelectorAll("[data-source-panel]"));
const trustedAcknowledgement = byId("trusted-acknowledgement");

function createNode(tagName, className, text) {
  const node = document.createElement(tagName);
  if (className) {
    node.className = className;
  }
  if (text !== undefined && text !== null) {
    node.textContent = String(text);
  }
  return node;
}

function measured(value, suffix = "") {
  if (value === null || value === undefined || value === "") {
    return "Ölçülmedi";
  }
  return `${value}${suffix}`;
}

function percentage(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > 100) {
    return null;
  }
  return value;
}

function formatPercentage(value) {
  const valid = percentage(value);
  return valid === null ? "Ölçülmedi" : `%${valid.toFixed(2)}`;
}

function formatDuration(value) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    return "Ölçülmedi";
  }
  return `${value.toFixed(2)} sn`;
}

function formatDate(value) {
  if (!value) {
    return "Bekleniyor";
  }
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Ölçülmedi"
    : date.toLocaleString("tr-TR", { dateStyle: "short", timeStyle: "medium" });
}

function sourceKindLabel(value) {
  return {
    INLINE_PYTHON_SOURCE: "Yapıştırılan Python kodu",
    UPLOADED_PYTHON_FILE: "Yüklenen Python dosyası",
    PUBLIC_GITHUB_REPOSITORY: "Public GitHub repository",
  }[value] || measured(value);
}

function policyLabel(value) {
  return value === "STATIC_DISCOVERY_ONLY"
    ? "Güvenli Statik Keşif"
    : value === "TRUSTED_DYNAMIC_ANALYSIS"
      ? "Güvenilir Dinamik Analiz"
      : measured(value);
}

function coverageStageLabel(value) {
  const labels = {
    COVERAGE_OPTIMIZATION: "Coverage optimizasyonu",
  };
  return labels[value] || "Bilinmeyen coverage aşaması";
}

function coverageMetricLabel(value) {
  const labels = {
    LINE: "Satır coverage",
    COMBINED: "Birleşik coverage",
  };
  return labels[value] || "Bilinmeyen coverage metriği";
}

function coverageStopReasonLabel(value) {
  if (value === null || value === undefined || value === "") {
    return "Devam ediyor";
  }
  const labels = {
    TARGET_REACHED: "Seçili güvenli aday kapsamı tamamlandı",
    CANDIDATES_EXHAUSTED: "Güvenli adaylar tamamlandı",
    PROVEN_PLATEAU: "Yeni doğrulanabilir coverage katkısı bulunamadı",
    TEST_LIMIT_REACHED: "Test sınırına ulaşıldı",
    DEADLINE_REACHED: "Zaman sınırına ulaşıldı",
    CANCELLED: "Analiz iptal edildi",
    FAILED: "Coverage ölçümü başarısız oldu",
  };
  return labels[value] || "Bilinmeyen durma nedeni";
}

function setMessage(message) {
  const target = byId("form-message");
  target.textContent = message || "";
  target.hidden = !message;
}

function setSubmitting(submitting) {
  state.submitInFlight = submitting;
  const button = byId("submit-analysis");
  button.disabled = submitting || state.activeJobInFlight;
  button.querySelector(".button-label").hidden = submitting;
  button.querySelector(".loading-label").hidden = !submitting;
}

function resetSourceSpecificControls() {
  invalidateGitHubTrust();
  byId("target-selection-mode").value = "ALL_ELIGIBLE_WITH_LIMIT";
  byId("explicit-target-names").value = "";
  byId("module-target-rows").replaceChildren();
  byId("github-ref").value = "";
}

function activateSource(source, focusPanel = false) {
  if (state.activeSource !== source) {
    rememberSourcePolicy();
    resetSourceSpecificControls();
  }
  state.activeSource = source;
  for (const tab of sourceTabs) {
    const active = tab.dataset.sourceTab === source;
    tab.classList.toggle("is-active", active);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const panel of sourcePanels) {
    const active = panel.dataset.sourcePanel === source;
    panel.hidden = !active;
    if (active && focusPanel) {
      const control = panel.querySelector("textarea, input");
      if (control) {
        control.focus();
      }
    }
  }
  restoreSourcePolicy();
  updateTargetSelectionControls();
  setMessage("");
}

function handleTabKeydown(event) {
  if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
    return;
  }
  event.preventDefault();
  const current = sourceTabs.indexOf(event.currentTarget);
  let next = current;
  if (event.key === "ArrowRight") {
    next = (current + 1) % sourceTabs.length;
  } else if (event.key === "ArrowLeft") {
    next = (current - 1 + sourceTabs.length) % sourceTabs.length;
  } else if (event.key === "Home") {
    next = 0;
  } else if (event.key === "End") {
    next = sourceTabs.length - 1;
  }
  sourceTabs[next].focus();
  activateSource(sourceTabs[next].dataset.sourceTab, false);
}

function selectedPolicy() {
  return document.querySelector('input[name="analysis-mode"]:checked').value;
}

function invalidateGitHubTrust() {
  state.githubTrust = null;
  state.acknowledgedTargetFingerprint = null;
  const trust = byId("github-trust-commit");
  if (trust) {
    trust.checked = false;
  }
  byId("github-trust-panel").hidden = true;
  byId("github-trusted-sha").textContent = "";
  byId("github-discovered-target").replaceChildren(
    new Option("Static discovery hedefi seçin", ""),
  );
  const dynamicPolicy = document.querySelector(
    'input[name="analysis-mode"][value="TRUSTED_DYNAMIC_ANALYSIS"]',
  );
  if (state.activeSource === "github" && dynamicPolicy.checked) {
    document.querySelector(
      'input[name="analysis-mode"][value="STATIC_DISCOVERY_ONLY"]',
    ).checked = true;
  }
  restoreSourcePolicy();
}

function currentGitHubTargetFingerprint() {
  if (state.activeSource !== "github") {
    return null;
  }
  const selection = targetSelectionOptions();
  if (!selection.valid
      || selection.target_selection_mode !== "EXPLICIT_QUALIFIED_TARGETS"
      || selection.explicit_module_targets.length === 0) {
    return null;
  }
  const selectors = selection.explicit_module_targets
    .map(({ module_identity: moduleIdentity, qualified_name: qualifiedName }) => (
      `${moduleIdentity}\u0000${qualifiedName}`
    ))
    .sort();
  return JSON.stringify({
    mode: selection.target_selection_mode,
    selectors,
  });
}

function invalidateGitHubTargetAcknowledgement() {
  state.acknowledgedTargetFingerprint = null;
  byId("github-trust-commit").checked = false;
  restoreSourcePolicy();
}

function updateGitHubTargetAcknowledgement() {
  const acknowledgement = byId("github-trust-commit");
  if (!acknowledgement.checked) {
    state.acknowledgedTargetFingerprint = null;
    restoreSourcePolicy();
    return;
  }
  const fingerprint = currentGitHubTargetFingerprint();
  if (state.githubTrust === null || fingerprint === null) {
    acknowledgement.checked = false;
    state.acknowledgedTargetFingerprint = null;
    restoreSourcePolicy();
    return;
  }
  state.acknowledgedTargetFingerprint = fingerprint;
  restoreSourcePolicy();
}

function applyGitHubDiscoveryTrust(result) {
  const submission = state.currentSubmission;
  const availableTargets = (Array.isArray(result.modules) ? result.modules : [])
    .flatMap((module) => (
      typeof module.module_name === "string"
        ? (Array.isArray(module.discovered_function_names)
            ? module.discovered_function_names
            : []).map((qualifiedName) => ({
              moduleIdentity: module.module_name,
              qualifiedName,
            }))
        : []
    ));
  if (!submission
      || submission.source !== "github"
      || submission.policy !== "STATIC_DISCOVERY_ONLY"
      || result.source_kind !== "PUBLIC_GITHUB_REPOSITORY"
      || result.analysis_policy !== "STATIC_DISCOVERY_ONLY"
      || result.status !== "COMPLETED"
      || result.acquisition_status !== "COMPLETED"
      || byId("github-url").value.trim() !== submission.repositoryUrl
      || byId("github-ref").value.trim() !== submission.repositoryRef
      || availableTargets.length === 0
      || !/^[0-9a-f]{40}$/.test(result.resolved_commit_sha || "")) {
    return;
  }
  state.githubTrust = Object.freeze({
    discoveryJobId: state.currentJobId,
    repositoryUrl: submission.repositoryUrl,
    requestedRef: submission.repositoryRef,
    resolvedCommitSha: result.resolved_commit_sha,
  });
  state.acknowledgedTargetFingerprint = null;
  byId("github-trusted-sha").textContent = result.resolved_commit_sha;
  const targetSelect = byId("github-discovered-target");
  targetSelect.replaceChildren(
    new Option("Static discovery hedefi seçin", ""),
  );
  for (const target of availableTargets) {
    const option = new Option(
      `${target.moduleIdentity} · ${target.qualifiedName}`,
      `${target.moduleIdentity}\u0000${target.qualifiedName}`,
    );
    targetSelect.append(option);
  }
  byId("github-trust-panel").hidden = false;
  byId("github-trust-commit").checked = false;
  restoreSourcePolicy();
}

function rememberSourcePolicy() {
  if (state.activeSource !== "github") {
    state.sourcePolicies[state.activeSource] = selectedPolicy();
  }
}

function restoreSourcePolicy() {
  const githubMode = state.activeSource === "github";
  const staticPolicy = document.querySelector(
    'input[name="analysis-mode"][value="STATIC_DISCOVERY_ONLY"]',
  );
  const dynamicPolicy = document.querySelector(
    'input[name="analysis-mode"][value="TRUSTED_DYNAMIC_ANALYSIS"]',
  );
  const githubDynamicAuthorized = githubMode
    && state.githubTrust !== null
    && byId("github-trust-commit").checked
    && state.acknowledgedTargetFingerprint !== null
    && state.acknowledgedTargetFingerprint === currentGitHubTargetFingerprint();
  dynamicPolicy.disabled = githubMode && !githubDynamicAuthorized;
  if (githubMode) {
    if (!githubDynamicAuthorized) {
      staticPolicy.checked = true;
    }
  } else {
    const restored = state.sourcePolicies[state.activeSource]
      || "STATIC_DISCOVERY_ONLY";
    staticPolicy.checked = restored === "STATIC_DISCOVERY_ONLY";
    dynamicPolicy.checked = restored === "TRUSTED_DYNAMIC_ANALYSIS";
  }
  updateModeControls();
}

function updateModeControls() {
  const dynamic = selectedPolicy() === "TRUSTED_DYNAMIC_ANALYSIS";
  byId("trusted-warning").hidden = !dynamic;
  byId("submit-summary").textContent = dynamic
    ? "Güvenilir dinamik pipeline başlatılacak"
    : "Statik keşif başlatılacak";

  for (const control of document.querySelectorAll(".dynamic-setting input")) {
    control.disabled = !dynamic;
  }
  for (const row of document.querySelectorAll(".dynamic-setting")) {
    row.classList.toggle("is-disabled", !dynamic);
  }
  for (const card of document.querySelectorAll(".choice-card")) {
    card.classList.toggle("is-selected", card.querySelector("input").checked);
  }
  if (!dynamic) {
    trustedAcknowledgement.checked = false;
  }
}

function updateSelectionControls() {
  const mode = byId("selection-mode").value;
  const explicit = mode !== "ALL_ELIGIBLE_WITH_LIMIT";
  byId("explicit-values-field").hidden = !explicit;
  byId("explicit-values-label").textContent =
    mode === "EXPLICIT_RELATIVE_PATHS" ? "Relative Python paths" : "Python module names";
  byId("explicit-values").placeholder =
    mode === "EXPLICIT_RELATIVE_PATHS"
      ? "package/core.py, package/utils.py"
      : "package.core, package.utils";
}

function csvValues(raw) {
  return Array.from(
    new Set(raw.split(",").map((value) => value.trim()).filter(Boolean)),
  );
}

function pythonIdentifier(value) {
  return /^[_\p{ID_Start}][_\p{ID_Continue}]*$/u.test(value);
}

function validateQualifiedTargetName(value) {
  if (!value || value !== value.trim()) {
    return false;
  }
  const parts = value.split(".");
  return (parts.length === 1 || parts.length === 2)
    && parts.every((part) => pythonIdentifier(part));
}

function validateModuleIdentity(value) {
  if (!value || value !== value.trim()) {
    return false;
  }
  return value.split(".").every((part) => pythonIdentifier(part));
}

function createModuleTargetRow() {
  const row = createNode("div", "module-target-row");
  const moduleInput = createNode("input");
  moduleInput.type = "text";
  moduleInput.placeholder = "package.module";
  moduleInput.autocomplete = "off";
  moduleInput.dataset.moduleIdentity = "";
  moduleInput.setAttribute("aria-label", "Module identity");
  const targetInput = createNode("input");
  targetInput.type = "text";
  targetInput.placeholder = "ClassName.method_name";
  targetInput.autocomplete = "off";
  targetInput.dataset.qualifiedTarget = "";
  targetInput.setAttribute("aria-label", "Qualified target name");
  const remove = createNode("button", "text-button", "Satırı kaldır");
  remove.type = "button";
  remove.addEventListener("click", () => {
    row.remove();
    if (!byId("module-target-rows").children.length) {
      byId("module-target-rows").append(createModuleTargetRow());
    }
    invalidateGitHubTargetAcknowledgement();
  });
  row.append(moduleInput, targetInput, remove);
  return row;
}

function updateTargetSelectionControls() {
  const explicit = byId("target-selection-mode").value
    === "EXPLICIT_QUALIFIED_TARGETS";
  const multiModule = state.activeSource === "github";
  byId("explicit-target-names-field").hidden = !explicit || multiModule;
  byId("explicit-module-targets-field").hidden = !explicit || !multiModule;
  if (explicit && multiModule && !byId("module-target-rows").children.length) {
    byId("module-target-rows").append(createModuleTargetRow());
  }
}

function targetSelectionOptions() {
  const mode = byId("target-selection-mode").value;
  if (mode === "ALL_ELIGIBLE_WITH_LIMIT") {
    return {
      valid: true,
      target_selection_mode: mode,
      explicit_target_names: [],
      explicit_module_targets: [],
    };
  }
  if (state.activeSource !== "github") {
    const values = byId("explicit-target-names").value
      .split("\n")
      .filter((value) => value.length > 0);
    if (!values.length || values.some((value) => !validateQualifiedTargetName(value))) {
      return { valid: false };
    }
    return {
      valid: true,
      target_selection_mode: mode,
      explicit_target_names: Array.from(new Set(values)),
      explicit_module_targets: [],
    };
  }
  const selectors = [];
  const seen = new Set();
  for (const row of byId("module-target-rows").children) {
    const moduleIdentity = row.querySelector("[data-module-identity]").value;
    const qualifiedName = row.querySelector("[data-qualified-target]").value;
    if (!validateModuleIdentity(moduleIdentity)
      || !validateQualifiedTargetName(qualifiedName)) {
      return { valid: false };
    }
    const key = `${moduleIdentity}\u0000${qualifiedName}`;
    if (!seen.has(key)) {
      seen.add(key);
      selectors.push({ module_identity: moduleIdentity, qualified_name: qualifiedName });
    }
  }
  return {
    valid: selectors.length > 0,
    target_selection_mode: mode,
    explicit_target_names: [],
    explicit_module_targets: selectors,
  };
}

function numericValue(id, label, { integer = false, nullable = false } = {}) {
  const raw = byId(id).value.trim();
  if (nullable && raw === "") {
    return null;
  }
  const value = Number(raw);
  if (!Number.isFinite(value) || (integer && !Number.isInteger(value))) {
    throw new Error(`${label} geçerli bir sayı olmalıdır.`);
  }
  return value;
}

function buildAnalysisOptions() {
  const policy = selectedPolicy();
  const selectionMode = byId("selection-mode").value;
  const values = csvValues(byId("explicit-values").value);
  if (selectionMode !== "ALL_ELIGIBLE_WITH_LIMIT" && values.length === 0) {
    throw new Error("Explicit modül seçimi en az bir değer gerektirir.");
  }
  const selection = targetSelectionOptions();
  if (!selection.valid) {
    throw new Error("Explicit target seçimi geçerli exact hedefler gerektirir.");
  }

  const options = {
    policy,
    trusted_execution_acknowledged: policy === "TRUSTED_DYNAMIC_ANALYSIS"
      ? trustedAcknowledgement.checked
      : false,
    selection_mode: selectionMode,
    explicit_relative_paths: selectionMode === "EXPLICIT_RELATIVE_PATHS" ? values : [],
    explicit_module_names: selectionMode === "EXPLICIT_MODULE_NAMES" ? values : [],
    target_selection_mode: selection.target_selection_mode,
    explicit_target_names: selection.explicit_target_names,
    explicit_module_targets: selection.explicit_module_targets,
    maximum_module_count: numericValue("maximum-module-count", "Maksimum modül", { integer: true }),
    maximum_function_count: numericValue("maximum-function-count", "Maksimum fonksiyon", { integer: true }),
    project_timeout_seconds: numericValue("project-timeout", "Toplam proje timeout", { nullable: true }),
  };
  if (options.project_timeout_seconds !== null
      && (options.project_timeout_seconds <= 0 || options.project_timeout_seconds > 14400)) {
    throw new Error("Toplam proje timeout 0 ile 14400 saniye arasında olmalıdır.");
  }

  if (policy === "TRUSTED_DYNAMIC_ANALYSIS") {
    if (!trustedAcknowledgement.checked) {
      throw new Error("Dinamik analiz için kaynak güveni onaylanmalıdır.");
    }
    const comparison = byId("strategy-comparison").checked;
    options.episode_count = numericValue("episode-count", "Episode sayısı", { integer: true });
    options.random_seed = numericValue("random-seed", "Random seed", { integer: true, nullable: true });
    options.pytest_coverage_timeout_seconds = numericValue("pytest-timeout", "Pytest timeout");
    options.function_pipeline_timeout_seconds = numericValue("pipeline-timeout", "Pipeline timeout");
    options.greedy_minimization = byId("greedy-minimization").checked || comparison;
    options.strategy_comparison = comparison;
  }
  return options;
}

function validateFile(file) {
  if (!file) {
    throw new Error("Bir Python dosyası seçin.");
  }
  if (!file.name.toLowerCase().endsWith(".py")) {
    throw new Error("Yalnız .py uzantılı dosya yüklenebilir.");
  }
  if (file.size === 0) {
    throw new Error("Python dosyası boş bırakılamaz.");
  }
  if (file.size > MAX_SOURCE_BYTES) {
    throw new Error("Dosya 2.000.000 byte sınırını aşıyor.");
  }
  return file;
}

async function fileIsBlankPythonSource(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let start = 0;
  if (bytes.length >= 3 && bytes[0] === 0xef && bytes[1] === 0xbb && bytes[2] === 0xbf) {
    start = 3;
  }
  const whitespace = new Set([0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x20]);
  for (let index = start; index < bytes.length; index += 1) {
    if (!whitespace.has(bytes[index])) {
      return false;
    }
  }
  return true;
}

function selectFile(file) {
  try {
    state.selectedFile = validateFile(file);
    byId("file-name").textContent = state.selectedFile.name;
    byId("file-size").textContent = `${state.selectedFile.size.toLocaleString("tr-TR")} byte`;
    byId("file-preview").hidden = false;
    setMessage("");
  } catch (error) {
    state.selectedFile = null;
    byId("python-file").value = "";
    byId("file-preview").hidden = true;
    setMessage(error.message);
  }
}

function clearSelectedFile() {
  state.selectedFile = null;
  byId("python-file").value = "";
  byId("file-preview").hidden = true;
  byId("file-name").textContent = "";
  byId("file-size").textContent = "";
}

async function sourceRequest(options) {
  if (state.activeSource === "inline") {
    const sourceCode = byId("inline-source").value;
    if (!sourceCode.replace(/^\uFEFF/, "").trim()) {
      throw new Error("Python kaynak kodu boş bırakılamaz.");
    }
    if (new TextEncoder().encode(sourceCode).length > MAX_SOURCE_BYTES) {
      throw new Error("Python kaynak kodu byte sınırını aşıyor.");
    }
    return {
      url: ENDPOINTS.inline,
      init: {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_code: sourceCode, analysis: options }),
      },
    };
  }

  if (state.activeSource === "upload") {
    const file = validateFile(state.selectedFile);
    if (await fileIsBlankPythonSource(file)) {
      throw new Error("Python dosyası boş bırakılamaz.");
    }
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("analysis", JSON.stringify(options));
    return { url: ENDPOINTS.upload, init: { method: "POST", body: form } };
  }

  const repositoryUrl = byId("github-url").value.trim();
  const repositoryRef = byId("github-ref").value.trim();
  if (!repositoryUrl) {
    throw new Error("Public GitHub URL boş bırakılamaz.");
  }
  let parsed;
  try {
    parsed = new URL(repositoryUrl);
  } catch {
    throw new Error("GitHub URL biçimi geçerli değil.");
  }
  if (parsed.protocol !== "https:"
      || parsed.hostname.toLowerCase() !== "github.com"
      || parsed.username
      || parsed.password
      || parsed.port
      || parsed.search
      || parsed.hash) {
    throw new Error("Beklenen biçim: https://github.com/owner/repository");
  }
  const dynamic = options.policy === "TRUSTED_DYNAMIC_ANALYSIS";
  if (dynamic) {
    const trust = state.githubTrust;
    if (!trust || !byId("github-trust-commit").checked) {
      throw new Error("Önce başarılı static discovery commit'ini açıkça trusted olarak onaylayın.");
    }
    if (repositoryUrl !== trust.repositoryUrl || repositoryRef !== trust.requestedRef) {
      invalidateGitHubTrust();
      throw new Error("Repository veya ref değişti; static discovery yeniden çalıştırılmalıdır.");
    }
    if (options.target_selection_mode !== "EXPLICIT_QUALIFIED_TARGETS"
        || options.explicit_module_targets.length === 0) {
      throw new Error("Pinned GitHub dynamic explicit module ve target seçimi gerektirir.");
    }
    const currentTargetFingerprint = currentGitHubTargetFingerprint();
    if (currentTargetFingerprint === null
        || state.acknowledgedTargetFingerprint !== currentTargetFingerprint) {
      invalidateGitHubTargetAcknowledgement();
      throw new Error(GITHUB_TARGET_ACKNOWLEDGEMENT_STALE);
    }
  }
  return {
    url: ENDPOINTS.github,
    init: {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repository_url: repositoryUrl,
        ref: dynamic ? null : repositoryRef || null,
        discovery_job_id: dynamic ? state.githubTrust.discoveryJobId : null,
        analysis: options,
      }),
    },
    context: {
      source: "github",
      policy: options.policy,
      repositoryUrl,
      repositoryRef,
    },
  };
}

async function safeResponseMessage(response, fallback) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string" && payload.detail.length <= 300) {
      return payload.detail;
    }
  } catch {
    return fallback;
  }
  return fallback;
}

function stopPolling() {
  state.pollGeneration += 1;
  if (state.pollTimer !== null) {
    window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }
  if (state.pollAbortController !== null) {
    state.pollAbortController.abort();
    state.pollAbortController = null;
  }
  if (state.terminalAbortController !== null) {
    state.terminalAbortController.abort();
    state.terminalAbortController = null;
  }
  state.retryAttempt = 0;
}

function resetOutput() {
  byId("result-section").hidden = true;
  byId("error-section").hidden = true;
  byId("module-results").replaceChildren();
  byId("result-summary").replaceChildren();
  byId("artifact-list").replaceChildren(
    createNode("p", "empty-state", "Artifact bilgisi bekleniyor."),
  );
  byId("result-heading").textContent = "Doğrulanmış analiz özeti";
  for (const id of ["result-description", "result-info"]) {
    byId(id).textContent = "";
    byId(id).hidden = true;
  }
}

function resetCoverageProgress() {
  state.coverageProgressRevision = -1;
  const bar = byId("coverage-progress-bar");
  bar.removeAttribute("value");
  byId("coverage-progress-metric").textContent = "Ölçüm bekleniyor";
  for (const id of [
    "coverage-progress-stage",
    "coverage-progress-line",
    "coverage-progress-branch",
    "coverage-progress-candidates",
    "coverage-progress-validated",
    "coverage-progress-effective",
    "coverage-progress-last-gain",
    "coverage-progress-stop-reason",
  ]) {
    byId(id).textContent = "Ölçüm bekleniyor";
  }
}

function progressCount(covered, total) {
  if (!Number.isInteger(covered) || covered < 0
      || !Number.isInteger(total) || total < 0 || covered > total) {
    return "Ölçülmedi";
  }
  return `${covered} / ${total}`;
}

function progressCountWithPercent(covered, total, percent) {
  const count = progressCount(covered, total);
  const validPercent = percentage(percent);
  if (count === "Ölçülmedi" && validPercent === null) {
    return "Ölçülmedi";
  }
  if (count === "Ölçülmedi") {
    return `%${validPercent.toFixed(2)}`;
  }
  return validPercent === null ? count : `${count} · %${validPercent.toFixed(2)}`;
}

function renderCoverageProgress(progress) {
  if (!progress || typeof progress !== "object"
      || !Number.isInteger(progress.revision) || progress.revision < 0
      || progress.revision <= state.coverageProgressRevision) {
    return;
  }
  state.coverageProgressRevision = progress.revision;
  const bar = byId("coverage-progress-bar");
  const overallPercent = percentage(progress.coverage_percent);
  if (overallPercent === null) {
    bar.removeAttribute("value");
  } else {
    bar.value = overallPercent;
  }
  byId("coverage-progress-stage").textContent = coverageStageLabel(progress.stage);
  byId("coverage-progress-metric").textContent =
    `${coverageMetricLabel(progress.metric)}: ${formatPercentage(progress.coverage_percent)}`;
  byId("coverage-progress-line").textContent = progressCountWithPercent(
    progress.covered_lines,
    progress.total_lines,
    progress.line_percent,
  );
  byId("coverage-progress-branch").textContent = progressCountWithPercent(
    progress.covered_branches,
    progress.total_branches,
    progress.branch_percent,
  );
  byId("coverage-progress-candidates").textContent = measured(progress.candidate_count);
  byId("coverage-progress-validated").textContent = measured(progress.validated_count);
  byId("coverage-progress-effective").textContent = measured(progress.effective_test_count);
  const lastGain = formatPercentage(progress.last_gain_percent);
  const newLines = measured(progress.last_new_line_count);
  const newBranches = measured(progress.last_new_branch_count);
  const plateau = measured(progress.plateau_count);
  byId("coverage-progress-last-gain").textContent =
    `${lastGain} · +${newLines} satır · +${newBranches} branch · plateau ${plateau}`;
  byId("coverage-progress-stop-reason").textContent =
    coverageStopReasonLabel(progress.stop_reason);
}

function statusDescription(status) {
  return {
    QUEUED: "İş bounded kuyruğa alındı; worker bekleniyor.",
    RUNNING: "Analiz pipeline’ı çalışıyor.",
    COMPLETED: "Analiz çalışması tamamlandı. Kapsam ve reddedilen yolları sonuç ayrıntılarından kontrol edin.",
    PARTIAL: "Kullanılabilir kısmi sonuç üretildi.",
    FAILED: "Analiz kontrollü bir hata ile durdu.",
    TIMED_OUT: "Pipeline zaman sınırına ulaştı.",
    CANCELLED: "Kuyruktaki iş çalıştırılmadan iptal edildi.",
  }[status] || "Job durumu güncellendi.";
}

function updateJobStatus(snapshot) {
  state.currentStatus = snapshot.status;
  if (TERMINAL_STATUSES.has(snapshot.status)) {
    state.activeJobInFlight = false;
    setSubmitting(state.submitInFlight);
  }
  const badge = byId("job-status-badge");
  badge.textContent = snapshot.status;
  badge.dataset.status = snapshot.status;
  byId("job-live-status").textContent = statusDescription(snapshot.status);
  byId("job-stage").textContent = measured(snapshot.progress_stage);
  byId("job-source-kind").textContent = sourceKindLabel(snapshot.source_kind);
  byId("job-policy").textContent = policyLabel(snapshot.analysis_policy);
  byId("job-started-at").textContent = formatDate(snapshot.started_at);
  byId("job-finished-at").textContent = formatDate(snapshot.finished_at);
  if (snapshot.coverage_progress === null) {
    if (snapshot.status === "FAILED") {
      resetCoverageProgress();
    }
  } else if (snapshot.coverage_progress !== undefined) {
    renderCoverageProgress(snapshot.coverage_progress);
  }

  const beginning = snapshot.started_at || snapshot.created_at || state.createdAt;
  const ending = snapshot.finished_at || new Date().toISOString();
  const elapsed = beginning
    ? Math.max(0, (new Date(ending).getTime() - new Date(beginning).getTime()) / 1000)
    : null;
  byId("job-elapsed").textContent = formatDuration(elapsed);

  const cancel = byId("cancel-job");
  cancel.hidden = TERMINAL_STATUSES.has(snapshot.status);
  cancel.disabled = snapshot.status !== "QUEUED";
  byId("cancel-help").textContent = snapshot.status === "RUNNING"
    ? "Çalışan iş güvenle sonlandırılamaz; mevcut pipeline timeout’u geçerlidir."
    : snapshot.status === "QUEUED"
      ? "Kuyruktaki iş çalıştırılmadan gerçekten iptal edilebilir."
      : "Terminal iş için iptal işlemi kullanılamaz.";
}

function schedulePoll(delay) {
  state.pollTimer = window.setTimeout(pollJob, delay);
}

async function pollJob() {
  if (!state.currentJobId) {
    return;
  }
  state.pollTimer = null;
  const jobId = state.currentJobId;
  const generation = state.pollGeneration;
  const controller = new AbortController();
  state.pollAbortController = controller;
  try {
    const response = await fetch(
      `${API_ROOT}/jobs/${encodeURIComponent(jobId)}`,
      { signal: controller.signal },
    );
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    if (response.status === 404) {
      if (state.pollAbortController === controller) {
        state.pollAbortController = null;
      }
      state.activeJobInFlight = false;
      setSubmitting(state.submitInFlight);
      showError("JOB_NOT_FOUND", measured(state.currentStatus), "Job kaydı retention süresi sonunda kaldırılmış; yeni bir analiz başlatabilirsiniz.");
      return;
    }
    if (!response.ok) {
      throw new Error(await safeResponseMessage(response, "Job durumu alınamadı."));
    }
    const snapshot = await response.json();
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    state.retryAttempt = 0;
    if (state.pollAbortController === controller) {
      state.pollAbortController = null;
    }
    updateJobStatus(snapshot);
    if (TERMINAL_STATUSES.has(snapshot.status)) {
      await loadTerminalResult(snapshot, jobId, generation);
      return;
    }
    schedulePoll(POLL_INTERVAL_MS);
  } catch (error) {
    if (error.name === "AbortError") {
      return;
    }
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    if (state.pollAbortController === controller) {
      state.pollAbortController = null;
    }
    state.retryAttempt += 1;
    if (state.retryAttempt <= MAX_RETRY_ATTEMPTS) {
      const delay = Math.min(POLL_INTERVAL_MS * (2 ** state.retryAttempt), 8000);
      byId("job-live-status").textContent = `Ağ bağlantısı bekleniyor; yeniden deneme ${state.retryAttempt}/${MAX_RETRY_ATTEMPTS}.`;
      schedulePoll(delay);
      return;
    }
    state.activeJobInFlight = false;
    setSubmitting(state.submitInFlight);
    showError("NETWORK_POLLING_FAILED", measured(state.currentStatus), "Job durumu güvenli retry sınırı içinde alınamadı.");
  }
}

function beginJob(snapshot, submission = null) {
  stopPolling();
  resetOutput();
  resetCoverageProgress();
  state.currentJobId = snapshot.job_id;
  state.currentStatus = snapshot.status;
  state.createdAt = snapshot.created_at;
  state.currentSubmission = submission;
  state.activeJobInFlight = true;
  byId("job-section").hidden = false;
  updateJobStatus(snapshot);
  byId("job-section").scrollIntoView({ behavior: "smooth", block: "start" });
  schedulePoll(250);
}

async function submitAnalysis(event) {
  event.preventDefault();
  if (state.submitInFlight) {
    return;
  }
  if (state.activeJobInFlight) {
    setMessage("Mevcut analiz terminal duruma ulaşmadan yeni bir analiz başlatılamaz.");
    return;
  }
  setMessage("");
  try {
    const request = await sourceRequest(buildAnalysisOptions());
    setSubmitting(true);
    if (request.context?.source === "github"
        && request.context.policy === "STATIC_DISCOVERY_ONLY") {
      invalidateGitHubTrust();
    }
    const response = await fetch(request.url, request.init);
    if (!response.ok) {
      if (request.context?.source === "github"
          && request.context.policy === "TRUSTED_DYNAMIC_ANALYSIS"
          && response.status === 422) {
        invalidateGitHubTrust();
      }
      throw new Error(await safeResponseMessage(response, "Analiz isteği kabul edilmedi."));
    }
    const snapshot = await response.json();
    if (state.activeSource === "inline") {
      byId("inline-source").value = "";
      updateSourceByteCount();
    } else if (state.activeSource === "upload") {
      clearSelectedFile();
    }
    beginJob(snapshot, request.context || { source: state.activeSource, policy: selectedPolicy() });
    if (request.context?.source === "github"
        && request.context.policy === "TRUSTED_DYNAMIC_ANALYSIS") {
      invalidateGitHubTrust();
    }
  } catch (error) {
    setMessage(error.message || "Analiz isteği gönderilemedi.");
  } finally {
    setSubmitting(false);
  }
}

function appendMetric(container, label, value) {
  const wrapper = createNode("div");
  wrapper.append(createNode("dt", "", label), createNode("dd", "", value));
  container.append(wrapper);
}

function renderInputRejectionCategories(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return "Yok";
  }
  return values
    .filter((item) => item && typeof item.category === "string" && Number.isInteger(item.count) && item.count >= 0)
    .slice()
    .sort((left, right) => left.category.localeCompare(right.category))
    .map((item) => `${item.category}: ${item.count}`)
    .join(", ") || "Yok";
}

function renderCoverage(container, label, value) {
  const row = createNode("div", "coverage-row");
  row.append(createNode("span", "", label));
  const track = createNode("div", "coverage-track");
  const valid = percentage(value);
  if (valid === null) {
    track.setAttribute("aria-hidden", "true");
  } else {
    const progress = createNode("progress", "coverage-fill");
    progress.max = 100;
    progress.value = valid;
    progress.setAttribute("aria-label", `${label} ${valid.toFixed(2)} yüzde`);
    track.append(progress);
  }
  row.append(track, createNode("strong", "", formatPercentage(value)));
  container.append(row);
}

function renderCoverageStrategy(
  container,
  title,
  lineCoverage,
  branchCoverage,
  coveragePreserved = null,
) {
  if (
    lineCoverage === null
    && branchCoverage === null
    && coveragePreserved === null
  ) {
    return;
  }
  const section = createNode("section", "coverage-strategy");
  section.append(createNode("h5", "", title));
  renderCoverage(section, "Line", lineCoverage);
  renderCoverage(section, "Branch", branchCoverage);
  if (coveragePreserved !== null) {
    section.append(
      createNode(
        "p",
        "coverage-preservation",
        coveragePreserved ? "Exact hedef korundu" : "Exact hedef korunmadı",
      ),
    );
  }
  container.append(section);
}

function renderComparison(functionResult, card) {
  if (!functionResult.comparison_status && !functionResult.strategy_winner) {
    return;
  }
  const comparison = createNode("section", "comparison-card");
  comparison.append(createNode("h5", "", "RL–greedy karşılaştırması"));
  const metrics = createNode("dl", "mini-metrics");
  appendMetric(metrics, "Full scenario pool", measured(functionResult.comparison_scenario_pool_count));
  if (functionResult.greedy_coverage_preserved === true) {
    appendMetric(metrics, "Greedy test", measured(functionResult.greedy_selected_count));
    appendMetric(metrics, "Greedy reduction", formatPercentage(functionResult.greedy_reduction_percentage));
  } else if (functionResult.greedy_coverage_preserved === false) {
    appendMetric(metrics, "Greedy durum", "Exact hedef korunmadı");
  }
  appendMetric(metrics, "RL test", measured(functionResult.rl_selected_count));
  appendMetric(metrics, "RL reduction", formatPercentage(functionResult.rl_reduction_percentage));
  appendMetric(metrics, "Winner", measured(functionResult.strategy_winner));
  appendMetric(
    metrics,
    "Exact coverage",
    functionResult.coverage_equality_verified === null
      ? "Ölçülmedi"
      : functionResult.coverage_equality_verified ? "Korundu" : "Korunmadı",
  );
  appendMetric(metrics, "Durma nedeni", measured(functionResult.rl_done_reason));
  comparison.append(metrics);
  if (functionResult.globally_minimal === false) {
    comparison.append(
      createNode(
        "p",
        "",
        "Seçilen küçültülmüş takım global optimum garantisi taşımaz; sonuç doğrulanmış strateji çıktısıdır.",
      ),
    );
  }
  card.append(comparison);
}

function renderFunction(functionResult) {
  const card = createNode("article", "function-card");
  const title = createNode("div", "function-title");
  title.append(
    createNode("h4", "", functionResult.qualified_name),
    createNode("span", "status-badge", functionResult.status),
  );
  title.lastChild.dataset.status = functionResult.status;
  card.append(title);

  const metrics = createNode("dl", "mini-metrics");
  appendMetric(metrics, "Sınırlandırılmış yol", measured(functionResult.bounded_path_count));
  appendMetric(metrics, "Girdi üretimi kabul", measured(functionResult.input_generation_accepted_count));
  appendMetric(metrics, "Girdi üretimi red", measured(functionResult.input_generation_rejected_count));
  appendMetric(metrics, "Red kategorileri", renderInputRejectionCategories(functionResult.input_rejection_categories));
  appendMetric(metrics, "Final scenario", measured(functionResult.scenario_count));
  appendMetric(metrics, "Concrete kabul", measured(functionResult.concrete_accepted_count));
  appendMetric(metrics, "Concrete red", measured(functionResult.concrete_rejected_count));
  appendMetric(metrics, "RL test", measured(functionResult.rl_test_count));
  appendMetric(metrics, "Q-table state", measured(functionResult.q_table_state_count));
  appendMetric(metrics, "Süre", formatDuration(functionResult.duration_seconds));
  appendMetric(metrics, "Durulan aşama", measured(functionResult.stopped_stage));
  appendMetric(metrics, "Hata kategorisi", measured(functionResult.error_category));
  appendMetric(metrics, "Skip nedeni", measured(functionResult.skip_reason));
  card.append(metrics);
  renderCoverageStrategy(
    card,
    "Hedef / Senaryo Havuzu Coverage",
    functionResult.scenario_pool_line_coverage_percent,
    functionResult.scenario_pool_branch_coverage_percent,
  );
  renderCoverageStrategy(
    card,
    "Greedy Coverage",
    functionResult.greedy_line_coverage_percent,
    functionResult.greedy_branch_coverage_percent,
    functionResult.greedy_coverage_preserved,
  );
  renderCoverageStrategy(
    card,
    "En İyi RL Coverage",
    functionResult.best_rl_line_coverage_percent,
    functionResult.best_rl_branch_coverage_percent,
    functionResult.best_rl_coverage_preserved,
  );
  renderComparison(functionResult, card);
  return card;
}

function renderStaticFunction(qualifiedName) {
  const card = createNode("article", "function-card");
  const title = createNode("div", "function-title");
  title.append(
    createNode("h4", "", qualifiedName),
    createNode("span", "status-badge", "STATIC_ONLY"),
  );
  title.lastChild.dataset.status = "STATIC_ONLY";
  card.append(title);

  const metrics = createNode("dl", "mini-metrics");
  appendMetric(metrics, "Scenario pool", measured(null));
  appendMetric(metrics, "Concrete kabul", measured(null));
  appendMetric(metrics, "Concrete red", measured(null));
  appendMetric(metrics, "RL test", measured(null));
  appendMetric(metrics, "Q-table state", measured(null));
  appendMetric(metrics, "Greedy seçilen test", measured(null));
  appendMetric(metrics, "RL seçilen test", measured(null));
  appendMetric(metrics, "Reduction", measured(null));
  appendMetric(metrics, "Strategy winner", measured(null));
  appendMetric(metrics, "Exact coverage preservation", measured(null));
  card.append(metrics);

  const coverage = createNode("section", "coverage-strategy");
  coverage.append(createNode("h5", "", "Function Coverage"));
  renderCoverage(coverage, "Function line coverage", null);
  renderCoverage(coverage, "Function branch coverage", null);
  card.append(
    coverage,
    createNode("p", "empty-state", "Dinamik olarak çalıştırılmadı"),
  );
  return card;
}

function renderModules(modules) {
  const container = byId("module-results");
  container.replaceChildren();
  if (!Array.isArray(modules) || modules.length === 0) {
    container.append(createNode("p", "empty-state", "Fonksiyon sonucu üretilmedi."));
    return;
  }
  for (const moduleResult of modules) {
    const card = createNode("article", "module-card");
    const header = createNode("header", "module-header");
    const copy = createNode("div");
    copy.append(
      createNode("h3", "", moduleResult.module_name || moduleResult.relative_path),
      createNode("p", "", moduleResult.relative_path),
    );
    const badge = createNode("span", "status-badge", moduleResult.status);
    badge.dataset.status = moduleResult.status;
    header.append(copy, badge);
    card.append(header);
    const functions = createNode("div", "function-list");
    if (Array.isArray(moduleResult.functions) && moduleResult.functions.length > 0) {
      for (const functionResult of moduleResult.functions) {
        functions.append(renderFunction(functionResult));
      }
    } else {
      const inventory = Array.isArray(moduleResult.discovered_function_names)
        ? moduleResult.discovered_function_names
        : [];
      if (inventory.length > 0) {
        functions.append(createNode("h4", "inventory-heading", "Statik fonksiyon envanteri"));
        for (const qualifiedName of inventory) {
          functions.append(renderStaticFunction(qualifiedName));
        }
      } else {
        functions.append(createNode("p", "empty-state", "Bu modül için dinamik fonksiyon metriği yok."));
      }
    }
    card.append(functions);
    container.append(card);
  }
}

function renderProjectCoverage(projectCoverage) {
  const section = byId("project-coverage-section");
  const hasMeasurement = projectCoverage !== null && typeof projectCoverage === "object";
  section.hidden = false;
  if (!hasMeasurement) {
    byId("project-coverage-status").textContent = "Ölçülmedi";
    byId("project-coverage").textContent = "Dinamik analiz yapılmadı";
    const bars = byId("project-coverage-bars");
    bars.replaceChildren();
    renderCoverage(bars, "Project line coverage", null);
    renderCoverage(bars, "Project branch coverage", null);
    byId("project-coverage-metrics").replaceChildren();
    byId("project-coverage-scope").textContent =
      "Whole repository coverage: Ölçülmedi";
    return;
  }
  const badge = byId("project-coverage-status");
  badge.textContent = measured(projectCoverage.run_status);
  badge.dataset.status = projectCoverage.run_status;
  byId("project-coverage").textContent =
    `Line ${formatPercentage(projectCoverage.full_line_coverage_percent)} · Branch ${formatPercentage(projectCoverage.full_branch_coverage_percent)}`;
  const bars = byId("project-coverage-bars");
  bars.replaceChildren();
  renderCoverage(bars, "Full combined line", projectCoverage.full_line_coverage_percent);
  renderCoverage(bars, "Full combined branch", projectCoverage.full_branch_coverage_percent);
  renderCoverage(bars, "Minimized line", projectCoverage.minimized_line_coverage_percent);
  renderCoverage(bars, "Minimized branch", projectCoverage.minimized_branch_coverage_percent);
  const metrics = byId("project-coverage-metrics");
  metrics.replaceChildren();
  appendMetric(metrics, "Full scenario", measured(projectCoverage.full_scenario_count));
  appendMetric(metrics, "Minimized scenario", measured(projectCoverage.final_selected_count));
  appendMetric(metrics, "Kaldırılan", measured(projectCoverage.removal_count));
  appendMetric(metrics, "Azalma", formatPercentage(projectCoverage.reduction_percentage));
  appendMetric(
    metrics,
    "Exact hedef",
    projectCoverage.coverage_preserved === null
      ? "Ölçülmedi"
      : projectCoverage.coverage_preserved ? "Korundu" : "Korunmadı",
  );
  appendMetric(metrics, "Full pytest exit", measured(projectCoverage.full_pytest_exit_code));
  appendMetric(metrics, "Minimized pytest exit", measured(projectCoverage.minimized_pytest_exit_code));
  const scope = projectCoverage.scope || {};
  const scopeLabel = projectCoverage.coverage_scope === "ANALYZED_PROJECT_SCOPE_COVERAGE"
    ? "Analiz Edilen Proje Kapsamı"
    : measured(projectCoverage.coverage_scope);
  appendMetric(metrics, "Analize alınan modül", measured(scope.selected_module_count));
  appendMetric(metrics, "Tamamlanan modül", measured(scope.completed_module_count));
  appendMetric(metrics, "Analiz edilen fonksiyon", measured(scope.analyzed_function_count));
  appendMetric(metrics, "Eksik fonksiyon", measured(scope.incomplete_function_count));
  appendMetric(metrics, "Partial", measured(scope.partial_function_count));
  appendMetric(metrics, "Failed", measured(scope.failed_function_count));
  appendMetric(metrics, "Timed out", measured(scope.timed_out_function_count));
  appendMetric(metrics, "Unsupported", measured(scope.unsupported_function_count));
  appendMetric(metrics, "SKIPPED_LIMIT", measured(scope.skipped_limit_function_count));
  appendMetric(metrics, "SKIPPED_SELECTION", measured(scope.skipped_selection_function_count));
  appendMetric(metrics, "Algorithm", measured(projectCoverage.algorithm));
  appendMetric(
    metrics,
    "Globally minimal",
    projectCoverage.globally_minimal === false ? "Hayır" : measured(projectCoverage.globally_minimal),
  );
  appendMetric(metrics, "Failure category", measured(projectCoverage.failure_category));
  appendMetric(metrics, "Unmeasured reason", measured(projectCoverage.unmeasured_reason));
  byId("project-coverage-scope").textContent =
    `${scopeLabel} · scope_complete=${measured(scope.scope_complete)} · Whole repository coverage: Ölçülmedi${scope.scope_complete === false ? " · Uyarı: analiz kapsamı eksik" : ""}`;
}

function renderProjectCoverageArtifacts(artifacts) {
  const container = byId("project-coverage-artifacts");
  container.replaceChildren();
  const labels = new Map([
    ["test_project_full_pool.py", "Combined pytest indir"],
    ["test_project_greedy_minimized.py", "Minimized pytest indir"],
    ["project_coverage_report.json", "Project JSON indir"],
  ]);
  for (const artifact of Array.isArray(artifacts) ? artifacts : []) {
    const label = labels.get(artifact.filename);
    if (!label) {
      continue;
    }
    const link = createNode("a", "artifact-link", label);
    link.href = `${API_ROOT}/jobs/${encodeURIComponent(state.currentJobId)}/artifacts/${encodeURIComponent(artifact.artifact_id)}`;
    link.setAttribute("download", "");
    container.append(link);
  }
}

function renderResult(result) {
  const section = byId("result-section");
  section.hidden = false;
  const badge = byId("result-status-badge");
  badge.textContent = result.status;
  badge.dataset.status = result.status;
  const noPythonRepository =
    result.source_kind === "PUBLIC_GITHUB_REPOSITORY"
    && result.status === "PARTIAL"
    && Array.isArray(result.issues)
    && result.issues.includes("NO_PYTHON_FILES");
  if (noPythonRepository) {
    byId("result-heading").textContent = "Repository keşfi tamamlandı";
    byId("result-description").textContent =
      "Repository başarıyla alındı ancak desteklenen Python kaynak dosyası bulunamadı.";
    byId("result-description").hidden = false;
    byId("result-info").textContent =
      "Bu araç şu anda yalnız Python projelerini analiz eder.";
    byId("result-info").hidden = false;
    byId("job-live-status").textContent =
      "Repository başarıyla alındı; Python kaynak keşfi kısmi sonuçla tamamlandı.";
  }
  const summary = byId("result-summary");
  summary.replaceChildren();
  appendMetric(summary, "Kaynak", sourceKindLabel(result.source_kind));
  appendMetric(summary, "Resolved commit SHA", measured(result.resolved_commit_sha));
  appendMetric(summary, "Politika", policyLabel(result.analysis_policy));
  appendMetric(summary, "Keşfedilen modül", measured(result.discovered_module_count));
  appendMetric(summary, "Seçilen modül", measured(result.selected_module_count));
  appendMetric(summary, "Keşfedilen fonksiyon", measured(result.discovered_function_count));
  appendMetric(summary, "Çalıştırılan fonksiyon", measured(result.analyzed_function_count));
  appendMetric(summary, "SKIPPED_LIMIT", measured(result.limit_skipped_function_count));
  appendMetric(summary, "SKIPPED_SELECTION", measured(result.selection_skipped_function_count));
  appendMetric(summary, "SKIPPED_DEADLINE", measured(result.deadline_skipped_function_count));
  appendMetric(summary, "Completed target", measured(result.completed_function_count));
  appendMetric(summary, "Partial target", measured(result.partial_function_count));
  appendMetric(summary, "Timed-out target", measured(result.timed_out_function_count));
  appendMetric(summary, "Toplam proje timeout", formatDuration(result.project_timeout_seconds));
  appendMetric(summary, "Deadline'a ulaşıldı", measured(result.project_deadline_exceeded));
  appendMetric(summary, "Son tamamlanan aşama", measured(result.last_completed_stage));
  appendMetric(summary, "Deadline aşaması", measured(result.deadline_stage));
  appendMetric(summary, "Toplam süre", formatDuration(result.duration_seconds));
  appendMetric(summary, "Cleanup", measured(result.cleanup_status));
  appendMetric(summary, "Kategori", measured(result.issues?.[0]));
  renderProjectCoverage(result.project_coverage);
  renderModules(result.modules);
}

function artifactKind(artifact) {
  if (artifact.content_type === "application/json") {
    return "JSON raporu";
  }
  if (artifact.content_type === "text/x-python") {
    return "Generated pytest";
  }
  return "Artifact";
}

function renderArtifacts(artifacts) {
  const container = byId("artifact-list");
  container.replaceChildren();
  if (!Array.isArray(artifacts) || artifacts.length === 0) {
    container.append(createNode("p", "empty-state", "İndirilebilir artifact üretilmedi."));
    renderProjectCoverageArtifacts([]);
    return;
  }
  for (const artifact of artifacts) {
    const item = createNode("article", "artifact-item");
    const copy = createNode("div");
    copy.append(
      createNode("strong", "", artifact.filename),
      createNode("small", "", `${artifactKind(artifact)} · ${measured(artifact.size_bytes, " byte")}`),
    );
    const link = createNode("a", "artifact-link", "İndir");
    link.href = `${API_ROOT}/jobs/${encodeURIComponent(state.currentJobId)}/artifacts/${encodeURIComponent(artifact.artifact_id)}`;
    link.setAttribute("download", "");
    link.setAttribute("aria-label", `${artifact.filename} artifact dosyasını indir`);
    item.append(copy, link);
    container.append(item);
  }
  renderProjectCoverageArtifacts(artifacts);
}

function showError(category, stage, message) {
  const section = byId("error-section");
  section.hidden = false;
  byId("error-category").textContent = measured(category);
  byId("error-stage").textContent = measured(stage);
  byId("error-message").textContent = message || "Analiz güvenli bir hata ile durdu.";
}

async function loadArtifactsForJob(jobId, generation) {
  const controller = new AbortController();
  state.terminalAbortController = controller;
  try {
    const response = await fetch(
      `${API_ROOT}/jobs/${encodeURIComponent(jobId)}/artifacts`,
      { signal: controller.signal },
    );
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    if (!response.ok) {
      renderArtifacts([]);
      return;
    }
    const payload = await response.json();
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    renderArtifacts(payload.artifacts);
  } catch (error) {
    if (error.name !== "AbortError"
        && state.currentJobId === jobId
        && state.pollGeneration === generation) {
      renderArtifacts([]);
    }
  } finally {
    if (state.terminalAbortController === controller) {
      state.terminalAbortController = null;
    }
  }
}

async function loadTerminalResult(snapshot, jobId, generation) {
  state.pollTimer = null;
  if (state.currentJobId !== jobId
      || state.pollGeneration !== generation) {
    return;
  }
  if (snapshot.status === "CANCELLED") {
    showError("CANCELLED", snapshot.progress_stage, "Kuyruktaki iş çalıştırılmadan iptal edildi.");
    return;
  }
  const controller = new AbortController();
  state.terminalAbortController = controller;
  try {
    const resultResponse = await fetch(
      `${API_ROOT}/jobs/${encodeURIComponent(jobId)}/result`,
      { signal: controller.signal },
    );
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    if (!resultResponse.ok) {
      if (resultResponse.status === 409) {
        showError(
          snapshot.safe_error_category || "RESULT_UNAVAILABLE",
          snapshot.progress_stage,
          "Terminal durum için güvenli sonuç özeti henüz mevcut değil.",
        );
        return;
      }
      throw new Error(await safeResponseMessage(resultResponse, "Analiz sonucu alınamadı."));
    }
    const result = await resultResponse.json();
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    renderResult(result);
    applyGitHubDiscoveryTrust(result);
    if (state.terminalAbortController === controller) {
      state.terminalAbortController = null;
    }
    await loadArtifactsForJob(jobId, generation);
    if (state.currentJobId !== jobId
        || state.pollGeneration !== generation) {
      return;
    }
    if (["FAILED", "TIMED_OUT"].includes(snapshot.status)) {
      showError(
        snapshot.safe_error_category || result.issues?.[0] || snapshot.status,
        snapshot.progress_stage,
        "Pipeline kontrollü bir terminal durumla tamamlandı; mevcut kısmi sonuçlar korunmuştur.",
      );
    }
    byId("result-section").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    if (error.name !== "AbortError"
        && state.currentJobId === jobId
        && state.pollGeneration === generation) {
      showError("RESULT_FETCH_FAILED", snapshot.progress_stage, error.message);
    }
  } finally {
    if (state.terminalAbortController === controller) {
      state.terminalAbortController = null;
    }
  }
}

async function cancelCurrentJob() {
  if (!state.currentJobId || state.currentStatus !== "QUEUED") {
    byId("job-live-status").textContent =
      "Çalışan veya terminal iş iptal edilmiş gibi gösterilemez; pipeline timeout’u geçerlidir.";
    return;
  }
  const button = byId("cancel-job");
  button.disabled = true;
  try {
    const response = await fetch(
      `${API_ROOT}/jobs/${encodeURIComponent(state.currentJobId)}/cancel`,
      { method: "POST" },
    );
    if (response.status === 409) {
      byId("job-live-status").textContent =
        "İş çalışmaya başladı; cancellation başarı sayılmadı. Pipeline timeout’u geçerlidir.";
      schedulePoll(250);
      return;
    }
    if (!response.ok) {
      throw new Error(await safeResponseMessage(response, "İptal isteği kabul edilmedi."));
    }
    const snapshot = await response.json();
    updateJobStatus(snapshot);
    if (snapshot.status === "CANCELLED") {
      stopPolling();
      const jobId = state.currentJobId;
      const generation = state.pollGeneration;
      await loadTerminalResult(snapshot, jobId, generation);
    }
  } catch (error) {
    byId("job-live-status").textContent = error.message;
    button.disabled = false;
  }
}

function updateSourceByteCount() {
  const bytes = new TextEncoder().encode(byId("inline-source").value).length;
  byId("source-byte-count").textContent = `${bytes.toLocaleString("tr-TR")} / 2.000.000 byte`;
  byId("source-byte-count").classList.toggle("is-over-limit", bytes > MAX_SOURCE_BYTES);
}

async function checkHealth() {
  try {
    const response = await fetch(`${API_ROOT}/health`);
    if (!response.ok) {
      throw new Error("health unavailable");
    }
    const health = await response.json();
    byId("health-dot").className = "status-dot status-dot-ok";
    byId("health-status").textContent = "API hazır";
    byId("capacity-status").textContent = `${health.running_jobs} çalışan · ${health.queued_jobs} kuyrukta · ${health.maximum_active_jobs} kapasite`;
  } catch {
    byId("health-dot").className = "status-dot status-dot-error";
    byId("health-status").textContent = "API erişilemiyor";
    byId("capacity-status").textContent = "Sunucu bağlantısını kontrol edin";
  }
}

for (const tab of sourceTabs) {
  tab.addEventListener("click", () => activateSource(tab.dataset.sourceTab, true));
  tab.addEventListener("keydown", handleTabKeydown);
}

for (const radio of document.querySelectorAll('input[name="analysis-mode"]')) {
  radio.addEventListener("change", () => {
    rememberSourcePolicy();
    updateModeControls();
  });
}

byId("selection-mode").addEventListener("change", updateSelectionControls);
byId("target-selection-mode").addEventListener("change", () => {
  updateTargetSelectionControls();
  invalidateGitHubTargetAcknowledgement();
});
byId("add-module-target").addEventListener("click", () => {
  byId("module-target-rows").append(createModuleTargetRow());
  invalidateGitHubTargetAcknowledgement();
});
byId("module-target-rows").addEventListener(
  "input",
  invalidateGitHubTargetAcknowledgement,
);
byId("module-target-rows").addEventListener(
  "change",
  invalidateGitHubTargetAcknowledgement,
);
byId("strategy-comparison").addEventListener("change", (event) => {
  if (event.currentTarget.checked) {
    byId("greedy-minimization").checked = true;
  }
});
byId("inline-source").addEventListener("input", updateSourceByteCount);
byId("python-file").addEventListener("change", (event) => selectFile(event.target.files[0]));
byId("remove-file").addEventListener("click", clearSelectedFile);
byId("github-url").addEventListener("input", invalidateGitHubTrust);
byId("github-ref").addEventListener("input", invalidateGitHubTrust);
byId("github-trust-commit").addEventListener(
  "change",
  updateGitHubTargetAcknowledgement,
);
byId("github-discovered-target").addEventListener("change", (event) => {
  invalidateGitHubTargetAcknowledgement();
  const [moduleIdentity, qualifiedName] = event.currentTarget.value.split("\u0000");
  if (!moduleIdentity || !qualifiedName) {
    return;
  }
  byId("target-selection-mode").value = "EXPLICIT_QUALIFIED_TARGETS";
  byId("module-target-rows").replaceChildren();
  const row = createModuleTargetRow();
  row.querySelector("[data-module-identity]").value = moduleIdentity;
  row.querySelector("[data-qualified-target]").value = qualifiedName;
  byId("module-target-rows").append(row);
  updateTargetSelectionControls();
});
byId("analysis-form").addEventListener("submit", submitAnalysis);
byId("cancel-job").addEventListener("click", cancelCurrentJob);
byId("retry-analysis").addEventListener("click", () => {
  byId("error-section").hidden = true;
  byId("analysis-form").scrollIntoView({ behavior: "smooth", block: "start" });
});

const dropZone = byId("drop-zone");
for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  });
}
dropZone.addEventListener("drop", (event) => {
  if (event.dataTransfer.files.length !== 1) {
    setMessage("Yalnız tek bir .py dosyası bırakın.");
    return;
  }
  selectFile(event.dataTransfer.files[0]);
});

window.addEventListener("beforeunload", stopPolling);
updateModeControls();
updateSelectionControls();
updateTargetSelectionControls();
updateSourceByteCount();
checkHealth();
