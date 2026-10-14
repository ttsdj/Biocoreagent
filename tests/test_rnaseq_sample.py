import json
from pathlib import Path

from scicore.kernel.providers.clients import FakeModelClient
from scicore.kernel.run_store import RunStore
from scicore.kernel.session_store import SessionStore
from scicore.kernel.workspace import WorkspaceContext

from scicore.bio.runtime import BioRuntime


def test_small_rnaseq_comparison_reaches_expected_direction(tmp_path: Path):
    (tmp_path / "counts.tsv").write_text(
        "gene\tctrl1\tctrl2\tctrl3\tcase1\tcase2\tcase3\n"
        "UP\t10\t12\t11\t100\t110\t105\n"
        "DOWN\t100\t95\t105\t10\t12\t11\n"
        "STABLE\t50\t51\t49\t50\t52\t48\n",
        encoding="utf-8",
    )
    (tmp_path / "metadata.tsv").write_text(
        "sample\tgroup\n"
        "ctrl1\tcontrol\nctrl2\tcontrol\nctrl3\tcontrol\n"
        "case1\tcase\ncase2\tcase\ncase3\tcase\n",
        encoding="utf-8",
    )
    client = FakeModelClient(
        [
            '<tool>{"name":"bio_rnaseq_compare","args":{'
            '"count_matrix_path":"counts.tsv","metadata_path":"metadata.tsv",'
            '"case_group":"case","control_group":"control","method":"welch","top_n":3}}</tool>',
            "<final>Comparison completed and inspected.</final>",
        ]
    )
    agent = BioRuntime(
        model_client=client,
        workspace=WorkspaceContext.build(tmp_path),
        session_store=SessionStore(tmp_path / ".scicore" / "sessions"),
        run_store=RunStore(tmp_path / ".scicore" / "runs"),
        approval_policy="auto",
        role="bio_worker",
        allow_orchestration=False,
    )

    answer = agent.ask("Compare case versus control.")
    tool_event = next(item for item in agent.session["history"] if item["role"] == "tool")
    result = json.loads(tool_event["content"])
    genes = {row["gene"]: row for row in result["top_results"]}

    assert answer == "Comparison completed and inspected."
    assert result["can_compare"] is True
    assert result["case_samples"] == ["case1", "case2", "case3"]
    assert result["control_samples"] == ["ctrl1", "ctrl2", "ctrl3"]
    assert genes["UP"]["log2_fold_change"] > 0
    assert genes["DOWN"]["log2_fold_change"] < 0
