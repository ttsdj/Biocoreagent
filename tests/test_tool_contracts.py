import json
import inspect

from scicore.agent.tools import ALL_TOOLS

from scicore.bio.domain import EXCLUDED_LEGACY_TOOLS, build_domain_adapters


def test_all_retained_scicore_tool_schemas_match_execute_signatures():
    for tool in ALL_TOOLS:
        properties = set(tool.parameters.get("properties", {}))
        required = set(tool.parameters.get("required", []))
        signature = inspect.signature(tool.execute)
        accepts_kwargs = any(
            parameter.kind == parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

        assert required <= properties, tool.name
        assert accepts_kwargs or properties <= set(signature.parameters), tool.name


def test_domain_registry_excludes_duplicate_control_plane_tools():
    adapters = build_domain_adapters()

    assert not (set(adapters) & EXCLUDED_LEGACY_TOOLS)
    assert "bio_rnaseq_compare" in adapters
    assert "pubmed_search" in adapters
    assert "pubmed_fetch_details" in adapters
    assert "pubmed_literature_review" in adapters
    assert "literature_red_blue_review" in adapters
    assert "literature_export_xlsx" in adapters
    assert "code_literature_link_save" in adapters
    assert "code_literature_link_search" in adapters
    assert "skill_save" in adapters
    assert "wiki_save" in adapters
    assert "ssh_bash" in adapters


def test_literature_red_blue_review_flags_missing_citation_fields():
    adapters = build_domain_adapters()
    tool = adapters["literature_red_blue_review"].tool

    result = tool.execute(
        items_json='[{"title":"Example protein folding paper","year":"2024","core_conclusion":"This proves folding always works."}]',
        claim="protein folding review",
    )

    assert "citation_quality" in result
    assert "MODIFY" in result
    assert "Missing DOI" in result


def test_literature_red_blue_review_repairs_and_reports_rounds():
    adapters = build_domain_adapters()
    tool = adapters["literature_red_blue_review"].tool

    result = tool.execute(
        items_json="""Here is JSON:
```json
[{"Title":"Protein folding example","Year":"2024","Core Conclusion":"This proves folding always works."}]
```
""",
        claim="protein folding",
        iterations=3,
        convergence_threshold=90,
    )
    payload = json.loads(result)

    assert payload["rounds"]
    assert payload["final_items"][0]["doi"] == "VERIFY_DOI_OR_PMID"
    assert payload["final_items"][0]["evidence_strength"] == "VERIFY_AND_CLASSIFY"
    assert "supports" in payload["final_items"][0]["core_conclusion"]
    assert any(item["layer"] == "fenced_json" and item["status"] == "success" for item in payload["parser_fallbacks"])
    assert "oscillation_detected" in payload
    assert "convergence_trace" in payload
    assert payload["stop_reason"] in {"score_converged", "oscillation_detected", "max_iterations"}


def test_literature_red_blue_review_converges_for_complete_rows():
    adapters = build_domain_adapters()
    tool = adapters["literature_red_blue_review"].tool

    result = tool.execute(
        items_json=json.dumps(
            [
                {
                    "title": "A complete protein folding paper",
                    "year": "2024",
                    "doi": "10.1000/example",
                    "abstract": "Protein folding is studied with controlled experiments.",
                    "core_conclusion": "The data support a folding mechanism.",
                    "evidence_strength": "medium",
                    "limitations": "Single domain model system.",
                }
            ]
        ),
        claim="protein folding mechanism",
        iterations=2,
        convergence_threshold=90,
    )
    payload = json.loads(result)

    assert payload["converged"] is True
    assert payload["score"] >= 90
    assert payload["rounds"][0]["red_agent_attacks"] == []


def test_literature_export_xlsx_writes_workbook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    adapters = build_domain_adapters()
    tool = adapters["literature_export_xlsx"].tool
    redblue = {
        "score": 95,
        "converged": True,
        "oscillation_detected": False,
        "stop_reason": "score_converged",
        "convergence_trace": {"score_history": [95]},
        "rounds": [],
    }

    result = tool.execute(
        items_json=json.dumps([{"title": "Protein folding", "year": "2024", "doi": "10.1000/example"}]),
        red_blue_json=json.dumps(redblue),
        output_path="reports/literature.xlsx",
    )
    payload = json.loads(result)

    assert payload["item_count"] == 1
    assert (tmp_path / "reports" / "literature.xlsx").exists()


def test_code_literature_link_save_and_search(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = tmp_path / "analysis.py"
    code.write_text("def fold_model():\n    return 'ok'\n", encoding="utf-8")
    adapters = build_domain_adapters()
    save_tool = adapters["code_literature_link_save"].tool
    search_tool = adapters["code_literature_link_search"].tool

    save_result = save_tool.execute(
        code_path=str(code),
        code_symbol="fold_model",
        purpose="protein folding demo",
        evidence_summary="Based on a protein folding paper.",
        literature_json=json.dumps([{"title": "Protein folding paper", "doi": "10.1000/fold"}]),
    )
    search_result = search_tool.execute(query="10.1000/fold")

    assert json.loads(save_result)["saved"] is True
    found = json.loads(search_result)
    assert found["count"] == 1
    assert found["links"][0]["code_symbol"] == "fold_model"
