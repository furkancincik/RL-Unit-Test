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


def test_submission_guards_and_polling_lifecycle_are_explicit() -> None:
    script = _script()
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert "submitInFlight" in script
    assert "if (state.submitInFlight)" in script
    assert "trustedAcknowledgement" in script
    assert "if (!trustedAcknowledgement.checked)" in script
    assert "if (!sourceCode.trim())" in script
    assert 'id="inline-source"' in markup
    assert "required" in markup.split('id="inline-source"', 1)[1].split(">", 1)[0]
    assert '.endsWith(".py")' in script
    assert "if (!repositoryUrl)" in script
    assert "stopPolling" in script
    assert "function beginJob(snapshot)" in script
    assert "AbortController" in script
    assert "TERMINAL_STATUSES" in script
    assert "if (TERMINAL_STATUSES.has(snapshot.status))" in script
    assert "retryAttempt" in script
    assert "state.retryAttempt <= MAX_RETRY_ATTEMPTS" in script
    assert "setTimeout" in script


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
    assert 'split(",")' not in script.split("explicit-target-names", 1)[-1]
    assert "if (!selection.valid)" in script


def test_artifact_section_explains_generated_pytest_import_root_contract() -> None:
    markup = Path("web/index.html").read_text(encoding="utf-8")

    assert "Generated pytest hedef kaynak modülünü içermez" in markup
    assert "python -m pytest" in markup
    assert "import root" in markup
