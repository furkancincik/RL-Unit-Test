from __future__ import annotations

from pathlib import Path


SCRIPT = Path("web/app.js")


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_source_kinds_use_distinct_api_endpoints() -> None:
    script = _script()

    assert '"/api/v1/jobs/inline"' in script
    assert '"/api/v1/jobs/upload"' in script
    assert '"/api/v1/jobs/github"' in script
    assert "local_path" not in script


def test_github_ref_and_resolved_sha_have_distinct_web_contracts() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert 'id="github-ref"' in markup
    assert 'for="github-ref"' in markup
    assert "ref: dynamic ? null : repositoryRef" in script
    assert "resolved_commit_sha" in script
    assert '"Resolved commit SHA"' in script


def test_github_mode_requires_static_snapshot_before_trusted_dynamic() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert 'id="github-static-policy"' in markup
    assert "Önce güvenli statik keşif" in markup
    assert "sourcePolicies" in script
    assert 'state.activeSource === "github"' in script
    assert "githubTrust" in script
    assert "discovery_job_id" in script
    assert 'id="github-trust-commit"' in markup
    assert "invalidateGitHubTrust" in script
    assert "applyGitHubDiscoveryTrust" in script
    assert "rememberSourcePolicy" in script
    assert "restoreSourcePolicy" in script


def test_github_dynamic_enablement_uses_one_authoritative_invariant() -> None:
    script = _script()

    assert "function canEnableGitHubDynamicAnalysis()" in script
    eligibility = script.split(
        "function canEnableGitHubDynamicAnalysis()", 1
    )[1].split("function restoreSourcePolicy", 1)[0]
    for prerequisite in (
        'state.activeSource !== "github"',
        "state.githubTrust === null",
        '!byId("github-trust-commit").checked',
        "currentTargetFingerprint === null",
        "state.acknowledgedTargetFingerprint === null",
        "state.acknowledgedTargetFingerprint === currentTargetFingerprint",
    ):
        assert prerequisite in eligibility

    restore = script.split("function restoreSourcePolicy", 1)[1].split(
        "function updateModeControls", 1
    )[0]
    assert "const githubDynamicAuthorized = canEnableGitHubDynamicAnalysis();" in restore


def test_discovered_github_target_rechecks_eligibility_after_state_update() -> None:
    script = _script()

    handler = script.split(
        'byId("github-discovered-target").addEventListener("change", (event) => {',
        1,
    )[1].split("});", 1)[0]
    assert handler.index("updateTargetSelectionControls();") < handler.rindex(
        "invalidateGitHubTargetAcknowledgement();"
    )


def test_submission_guards_and_polling_lifecycle_are_explicit() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert "submitInFlight" in script
    assert "if (state.submitInFlight)" in script
    assert "trustedAcknowledgement" in script
    assert "if (!trustedAcknowledgement.checked)" in script
    assert 'if (!sourceCode.replace(/^\\uFEFF/, "").trim())' in script
    assert 'id="inline-source"' in markup
    assert "required" in markup.split('id="inline-source"', 1)[1].split(">", 1)[0]
    assert '.endsWith(".py")' in script
    assert "file.size === 0" in script
    assert "fileIsBlankPythonSource" in script
    assert "sourceCode.replace(/^\\uFEFF/, \"\").trim()" in script
    assert "if (!repositoryUrl)" in script
    assert "stopPolling" in script
    assert "function beginJob(snapshot, submission = null)" in script
    assert "AbortController" in script
    assert "TERMINAL_STATUSES" in script
    assert "if (TERMINAL_STATUSES.has(snapshot.status))" in script
    assert "retryAttempt" in script
    assert "state.retryAttempt <= MAX_RETRY_ATTEMPTS" in script
    assert "setTimeout" in script


def test_polling_and_terminal_fetch_are_scoped_to_the_current_job_generation() -> None:
    script = _script()

    assert "pollGeneration" in script
    assert "terminalAbortController" in script
    assert "const jobId = state.currentJobId" in script
    assert "const generation = state.pollGeneration" in script
    assert "state.currentJobId !== jobId" in script
    assert "state.pollGeneration !== generation" in script
    assert "loadTerminalResult(snapshot, jobId, generation)" in script
    assert "encodeURIComponent(jobId)" in script
    assert "signal: controller.signal" in script


def test_terminal_result_survives_artifact_network_failure() -> None:
    script = _script()
    terminal_loader = script.split(
        "async function loadTerminalResult", 1
    )[1].split("async function cancelCurrentJob", 1)[0]

    assert "async function loadArtifactsForJob" in script
    assert "renderResult(result)" in terminal_loader
    assert "await loadArtifactsForJob(jobId, generation)" in terminal_loader
    assert "Promise.all([" not in terminal_loader


def test_active_job_blocks_duplicate_submit_until_terminal_state() -> None:
    script = _script()

    assert "activeJobInFlight" in script
    assert "if (state.activeJobInFlight)" in script
    assert "state.activeJobInFlight = true" in script
    assert "state.activeJobInFlight = false" in script


def test_expired_polling_and_rejected_pinned_trust_are_recoverable() -> None:
    script = _script()

    assert "response.status === 404" in script
    assert 'showError("JOB_NOT_FOUND"' in script
    assert "state.activeJobInFlight = false" in script
    assert "request.context.policy === \"TRUSTED_DYNAMIC_ANALYSIS\"" in script
    assert "response.status === 422" in script
    assert "invalidateGitHubTrust()" in script


def test_source_mode_switch_resets_target_and_ref_state() -> None:
    script = _script()

    assert "function resetSourceSpecificControls" in script
    assert 'byId("target-selection-mode").value = "ALL_ELIGIBLE_WITH_LIMIT"' in script
    assert 'byId("explicit-target-names").value = ""' in script
    assert 'byId("module-target-rows").replaceChildren()' in script
    assert 'byId("github-ref").value = ""' in script
    assert "resetSourceSpecificControls()" in script


def test_source_switch_and_invalid_submission_clear_stale_results() -> None:
    script = _script()

    activate_source = script.split("function activateSource", 1)[1].split(
        "function handleTabKeydown", 1
    )[0]
    assert "resetOutput();" in activate_source

    submit = script.split("async function submitAnalysis", 1)[1].split(
        "function appendMetric", 1
    )[0]
    assert submit.index("resetOutput();") < submit.index(
        "await sourceRequest(buildAnalysisOptions())"
    )


def test_project_deadline_has_distinct_input_and_backend_authoritative_metrics() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert 'id="project-timeout"' in markup
    assert "Toplam proje timeout" in markup
    assert "project_timeout_seconds" in script
    assert "project_deadline_exceeded" in script
    assert "deadline_skipped_function_count" in script
    assert "last_completed_stage" in script
    assert "deadline_stage" in script
    assert "function_pipeline_timeout_seconds" in script
    assert "calculateDeadline" not in script


def test_frontend_does_not_persist_or_unsafely_render_payloads() -> None:
    script = _script()

    assert ".innerHTML" not in script
    assert "localStorage" not in script
    assert "sessionStorage" not in script
    assert "console.log" not in script
    assert ".textContent" in script


def test_result_rendering_preserves_backend_authority() -> None:
    script = _script()

    assert '"Ölçülmedi"' in script
    assert "strategy_winner" in script
    assert "coverage_equality_verified" in script
    assert "globally_minimal" in script
    assert "calculateWinner" not in script
    assert "artifact.artifact_id" in script
    assert "/artifacts/" in script
    assert "response.status === 409" in script


def test_dynamic_coverage_strategies_are_rendered_separately() -> None:
    script = _script()

    assert '"Hedef / Senaryo Havuzu Coverage"' in script
    assert '"Greedy Coverage"' in script
    assert '"En İyi RL Coverage"' in script
    assert "scenario_pool_line_coverage_percent" in script
    assert "greedy_line_coverage_percent" in script
    assert "best_rl_line_coverage_percent" in script
    assert "greedy_coverage_preserved" in script
    assert "best_rl_coverage_preserved" in script
    assert '"Exact hedef korunmadı"' in script
    assert "calculateWinner" not in script


def test_no_python_repository_has_non_conflicting_partial_copy() -> None:
    script = _script()

    assert 'result.source_kind === "PUBLIC_GITHUB_REPOSITORY"' in script
    assert 'result.status === "PARTIAL"' in script
    assert 'result.issues.includes("NO_PYTHON_FILES")' in script
    assert '"Repository keşfi tamamlandı"' in script
    assert (
        '"Repository başarıyla alındı ancak desteklenen Python kaynak dosyası bulunamadı."'
        in script
    )
    assert '"Bu araç şu anda yalnız Python projelerini analiz eder."' in script


def test_static_inventory_is_rendered_without_fake_execution_results() -> None:
    script = _script()

    assert "moduleResult.discovered_function_names" in script
    assert '"Statik fonksiyon envanteri"' in script
    assert '"Dinamik olarak çalıştırılmadı"' in script
    assert "function renderStaticFunction" in script
    for label in (
        "Scenario pool",
        "Concrete kabul",
        "Concrete red",
        "RL test",
        "Q-table state",
        "Function line coverage",
        "Function branch coverage",
        "Greedy seçilen test",
        "RL seçilen test",
        "Reduction",
        "Strategy winner",
        "Exact coverage preservation",
        "Project line coverage",
        "Project branch coverage",
    ):
        assert f'"{label}"' in script
    assert "renderStaticFunction(qualifiedName)" in script


def test_dynamic_result_rendering_keeps_backend_numeric_metrics() -> None:
    script = _script()

    assert 'measured(functionResult.scenario_count)' in script
    assert 'measured(functionResult.concrete_accepted_count)' in script
    assert 'measured(functionResult.concrete_rejected_count)' in script
    assert 'measured(functionResult.rl_test_count)' in script
    assert 'measured(functionResult.q_table_state_count)' in script
    assert 'measured(functionResult.greedy_selected_count)' in script
    assert 'measured(functionResult.rl_selected_count)' in script
    assert 'measured(functionResult.strategy_winner)' in script


def test_dynamic_result_renders_input_rejections_without_overstating_completion() -> None:
    script = _script()

    assert 'measured(functionResult.bounded_path_count)' in script
    assert 'measured(functionResult.input_generation_accepted_count)' in script
    assert 'measured(functionResult.input_generation_rejected_count)' in script
    assert "function renderInputRejectionCategories" in script
    assert "functionResult.input_rejection_categories" in script
    assert '"Sınırlandırılmış yol"' in script
    assert '"Girdi üretimi kabul"' in script
    assert '"Girdi üretimi red"' in script
    assert '"Red kategorileri"' in script
    assert '"Final scenario"' in script
    assert '"Analiz çalışması tamamlandı. Kapsam ve reddedilen yolları sonuç ayrıntılarından kontrol edin."' in script
    assert 'COMPLETED: "Analiz eksiksiz tamamlandı."' not in script
    assert ".innerHTML" not in script


def test_project_coverage_has_a_distinct_exact_scope_section() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert 'id="project-coverage-section"' in markup
    assert "function renderProjectCoverage" in script
    assert "result.project_coverage" in script
    assert "coverage_scope" in script
    assert "scope_complete" in script
    assert "full_scenario_count" in script
    assert "final_selected_count" in script
    assert "reduction_percentage" in script
    assert "coverage_preserved" in script
    assert '"Ölçülmedi"' in script
    assert '"Dinamik analiz yapılmadı"' in script
    assert '"Analize alınan modül"' in script
    assert '"Eksik fonksiyon"' in script
    assert '"Unsupported"' in script
    assert '"SKIPPED_LIMIT"' in script
    assert '"SKIPPED_SELECTION"' in script
    assert '"Globally minimal"' in script
    assert "analiz kapsamı eksik" in script
    assert '"Combined pytest indir"' in script
    assert '"Minimized pytest indir"' in script
    assert '"Project JSON indir"' in script


def test_explicit_qualified_target_controls_validate_without_ambiguous_csv() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert 'id="target-selection-mode"' in markup
    assert 'value="ALL_ELIGIBLE_WITH_LIMIT"' in markup
    assert 'value="EXPLICIT_QUALIFIED_TARGETS"' in markup
    assert 'id="explicit-target-names"' in markup
    assert 'id="module-target-rows"' in markup
    assert 'id="add-module-target"' in markup
    assert "validateQualifiedTargetName" in script
    assert "validateModuleIdentity" in script
    assert "explicit_target_names" in script
    assert "explicit_module_targets" in script
    assert "target_selection_mode" in script
    assert 'split("\\n")' in script
    target_selection_implementation = script.split(
        "function targetSelectionOptions()", 1
    )[1].split("function numericValue", 1)[0]
    assert 'split(",")' not in target_selection_implementation
    assert "if (!selection.valid)" in script


def test_artifact_section_explains_generated_pytest_import_root_contract() -> None:
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert "Generated pytest hedef kaynak modülünü içermez" in markup
    assert "python -m pytest" in markup
    assert "import root" in markup


def test_github_target_acknowledgement_is_bound_to_current_target_fingerprint() -> None:
    script = _script()

    assert "acknowledgedTargetFingerprint" in script
    assert "currentGitHubTargetFingerprint" in script
    assert "invalidateGitHubTargetAcknowledgement" in script
    assert "const currentTargetFingerprint = currentGitHubTargetFingerprint();" in script
    assert (
        "state.acknowledgedTargetFingerprint === currentTargetFingerprint"
        in script
    )

    target_invalidation = script.split(
        "function invalidateGitHubTargetAcknowledgement()", 1
    )[1].split("function updateGitHubTargetAcknowledgement", 1)[0]
    assert "state.githubTrust = null" not in target_invalidation
    assert 'byId("github-trust-panel").hidden = true' not in target_invalidation
    assert 'byId("github-trusted-sha").textContent = ""' not in target_invalidation


def test_all_github_target_set_mutations_invalidate_dynamic_acknowledgement() -> None:
    script = _script()

    assert 'byId("target-selection-mode").addEventListener("change", () => {' in script
    assert 'byId("add-module-target").addEventListener("click", () => {' in script
    assert 'byId("github-discovered-target").addEventListener("change", (event) => {' in script
    assert '"input",\n  invalidateGitHubTargetAcknowledgement,' in script
    assert '"change",\n  invalidateGitHubTargetAcknowledgement,' in script
    row_factory = script.split("function createModuleTargetRow()", 1)[1].split(
        "function updateTargetSelectionControls", 1
    )[0]
    assert "invalidateGitHubTargetAcknowledgement();" in row_factory
    assert script.count("invalidateGitHubTargetAcknowledgement();") >= 5


def test_github_submit_rejects_stale_target_acknowledgement_before_request() -> None:
    script = _script()

    assert "GITHUB_TARGET_ACKNOWLEDGEMENT_STALE" in script
    source_request = script.split("async function sourceRequest", 1)[1].split(
        "async function safeResponseMessage", 1
    )[0]
    assert (
        "state.acknowledgedTargetFingerprint !== currentTargetFingerprint"
        in source_request
    )
    assert "invalidateGitHubTargetAcknowledgement();" in source_request
    assert "throw new Error(GITHUB_TARGET_ACKNOWLEDGEMENT_STALE);" in source_request


def test_job_card_exposes_authoritative_coverage_progress_contract() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    for element_id in (
        "coverage-progress-card",
        "coverage-progress-stage",
        "coverage-progress-metric",
        "coverage-progress-line",
        "coverage-progress-branch",
        "coverage-progress-candidates",
        "coverage-progress-validated",
        "coverage-progress-effective",
        "coverage-progress-last-gain",
        "coverage-progress-stop-reason",
    ):
        assert f'id="{element_id}"' in markup
    assert 'id="coverage-progress-bar"' in markup
    assert 'aria-label="Canlı coverage ilerlemesi"' in markup
    assert "function renderCoverageProgress" in script
    assert "snapshot.coverage_progress" in script
    assert '"Ölçüm bekleniyor"' in script


def test_coverage_progress_is_backend_authoritative_and_never_fakes_percentages() -> None:
    script = _script()
    renderer = script.split("function renderCoverageProgress", 1)[1].split(
        "function statusDescription", 1
    )[0]

    for field in (
        "revision",
        "stage",
        "metric",
        "coverage_percent",
        "line_percent",
        "branch_percent",
        "covered_lines",
        "total_lines",
        "covered_branches",
        "total_branches",
        "candidate_count",
        "validated_count",
        "effective_test_count",
        "last_gain_percent",
        "last_new_line_count",
        "last_new_branch_count",
        "plateau_count",
        "stop_reason",
    ):
        assert f"progress.{field}" in renderer
    assert "Math.min" not in renderer
    assert "Math.max" not in renderer
    assert "TERMINAL_STATUSES" not in renderer
    assert "innerHTML" not in renderer
    assert ".textContent" in renderer


def test_coverage_progress_reset_revision_and_stale_job_guards_are_explicit() -> None:
    script = _script()

    assert "coverageProgressRevision" in script
    assert "function resetCoverageProgress" in script
    assert "resetCoverageProgress();" in script.split(
        "function beginJob", 1
    )[1].split("async function submitAnalysis", 1)[0]
    assert "progress.revision <= state.coverageProgressRevision" in script
    poller = script.split("async function pollJob", 1)[1].split(
        "function beginJob", 1
    )[0]
    assert poller.index("state.currentJobId !== jobId") < poller.index(
        "updateJobStatus(snapshot)"
    )
    assert poller.index("state.pollGeneration !== generation") < poller.index(
        "updateJobStatus(snapshot)"
    )


def test_terminal_progress_persists_but_failure_without_measurement_clears_it() -> None:
    script = _script()
    updater = script.split("function updateJobStatus", 1)[1].split(
        "function schedulePoll", 1
    )[0]

    assert "renderCoverageProgress(snapshot.coverage_progress)" in updater
    assert "snapshot.coverage_progress === null" in updater
    assert 'snapshot.status === "FAILED"' in updater
    assert "resetCoverageProgress();" in updater
    assert 'snapshot.status === "COMPLETED"' not in updater
    assert 'snapshot.status === "PARTIAL"' not in updater


def test_coverage_stage_metric_and_stop_reason_use_safe_allowlist_labels() -> None:
    script = _script()
    renderer = script.split("function renderCoverageProgress", 1)[1].split(
        "function statusDescription", 1
    )[0]

    assert "function coverageStageLabel" in script
    assert 'COVERAGE_OPTIMIZATION: "Coverage optimizasyonu"' in script
    assert 'return labels[value] || "Bilinmeyen coverage aşaması";' in script
    assert "function coverageMetricLabel" in script
    assert 'LINE: "Satır coverage"' in script
    assert 'COMBINED: "Birleşik coverage"' in script
    assert 'return labels[value] || "Bilinmeyen coverage metriği";' in script
    assert "function coverageStopReasonLabel" in script
    for reason in (
        "TARGET_REACHED",
        "CANDIDATES_EXHAUSTED",
        "PROVEN_PLATEAU",
        "TEST_LIMIT_REACHED",
        "DEADLINE_REACHED",
        "CANCELLED",
        "FAILED",
    ):
        assert f'{reason}: "' in script
    assert 'return labels[value] || "Bilinmeyen durma nedeni";' in script
    assert "coverageStageLabel(progress.stage)" in renderer
    assert "coverageMetricLabel(progress.metric)" in renderer
    assert "coverageStopReasonLabel(progress.stop_reason)" in renderer
    assert "measured(progress.stage)" not in renderer
    assert "measured(progress.stop_reason)" not in renderer
    assert "`${progress.metric}" not in renderer


def test_progress_lifecycle_contract_covers_terminal_reset_and_stale_revision() -> None:
    script = _script()
    updater = script.split("function updateJobStatus", 1)[1].split(
        "function schedulePoll", 1
    )[0]
    begin_job = script.split("function beginJob", 1)[1].split(
        "async function submitAnalysis", 1
    )[0]
    renderer = script.split("function renderCoverageProgress", 1)[1].split(
        "function statusDescription", 1
    )[0]

    assert "renderCoverageProgress(snapshot.coverage_progress)" in updater
    assert 'snapshot.status === "COMPLETED"' not in updater
    assert 'snapshot.status === "PARTIAL"' not in updater
    assert "resetCoverageProgress();" in begin_job
    assert "state.coverageProgressRevision = -1" in script
    assert "progress.revision <= state.coverageProgressRevision" in renderer
    assert "state.coverageProgressRevision = progress.revision" in renderer


def test_target_reached_describes_exact_safe_candidate_scope_not_a_numeric_maximum() -> None:
    script = _script()
    stop_labels = script.split("function coverageStopReasonLabel", 1)[1].split(
        "function setMessage", 1
    )[0]
    renderer = script.split("function renderCoverageProgress", 1)[1].split(
        "function statusDescription", 1
    )[0]
    representative_backend_snapshot = {
        "coverage_percent": 57.4627,
        "stop_reason": "TARGET_REACHED",
    }

    assert representative_backend_snapshot["coverage_percent"] < 100.0
    assert (
        'TARGET_REACHED: "Seçili güvenli aday kapsamı tamamlandı"'
        in stop_labels
    )
    assert 'TARGET_REACHED: "Coverage hedefine ulaşıldı"' not in stop_labels
    assert "maksimum" not in stop_labels.casefold()
    assert "progress.coverage_percent === 100" not in renderer
    assert "progress.coverage_percent >= 100" not in renderer


def test_coverage_stop_labels_keep_distinct_controlled_terminal_meanings() -> None:
    script = _script()
    stop_labels = script.split("function coverageStopReasonLabel", 1)[1].split(
        "function setMessage", 1
    )[0]

    expected = {
        "CANDIDATES_EXHAUSTED": "Güvenli adaylar tamamlandı",
        "PROVEN_PLATEAU": "Yeni doğrulanabilir coverage katkısı bulunamadı",
        "TEST_LIMIT_REACHED": "Test sınırına ulaşıldı",
        "DEADLINE_REACHED": "Zaman sınırına ulaşıldı",
    }
    for reason, label in expected.items():
        assert f'{reason}: "{label}"' in stop_labels
    assert 'return labels[value] || "Bilinmeyen durma nedeni";' in stop_labels
