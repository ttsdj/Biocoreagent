"""Quality evaluation for temporal, evidence-backed research graph projection."""

from __future__ import annotations

import re
import statistics


def _normalize(value):
    return re.sub(r"\s+", " ", str(value).strip().lower().replace("\\", "/"))


def _f1(expected, predicted):
    expected, predicted = set(expected), set(predicted)
    true_positive = len(expected & predicted)
    precision = true_positive / len(predicted) if predicted else float(not expected)
    recall = true_positive / len(expected) if expected else float(not predicted)
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "expected": len(expected),
        "predicted": len(predicted),
        "precision": precision,
        "recall": recall,
        "f1": score,
    }


def evaluate_research_graph(cases, project, *, min_score=80.0):
    """Evaluate entity, event, effective-time, relation and evidence quality.

    ``project`` receives the case's attributed memory and returns the canonical
    graph projection. Cases must be independently labelled and frozen before
    the resulting metrics can be used as evidence.
    """

    cases = list(cases)
    if not cases:
        raise ValueError("research graph evaluation requires at least one case")
    expected_entities, predicted_entities = [], []
    expected_relations, predicted_relations = [], []
    event_type_matches, temporal_matches, evidence_matches = [], [], []
    details = []
    for case in cases:
        graph = project(dict(case["memory"]))
        case_id = str(case["case_id"])
        expected = dict(case["expected"])
        expected_case_entities = {
            (_normalize(item["entity_type"]), _normalize(item["canonical_name"]))
            for item in expected.get("entities", [])
        }
        predicted_case_entities = {
            (_normalize(item["entity_type"]), _normalize(item["canonical_name"]))
            for item in graph.get("entities", [])
        }
        expected_entities.extend((case_id, *item) for item in expected_case_entities)
        predicted_entities.extend((case_id, *item) for item in predicted_case_entities)

        expected_case_relations = {
            (
                _normalize(item["predicate"]),
                _normalize(item["object_event_id"]),
            )
            for item in expected.get("event_relations", [])
        }
        predicted_case_relations = {
            (
                _normalize(item["predicate"]),
                _normalize(item["object_event_id"]),
            )
            for item in graph.get("event_relations", [])
        }
        expected_relations.extend((case_id, *item) for item in expected_case_relations)
        predicted_relations.extend((case_id, *item) for item in predicted_case_relations)

        event_match = _normalize(graph["event"]["event_type"]) == _normalize(
            expected["event_type"]
        )
        event_type_matches.append(float(event_match))
        expected_times = dict(expected.get("effective_time", {}))
        time_fields = [
            field
            for field in ("valid_from", "valid_to")
            if field in expected_times
        ]
        time_checks = [
            _normalize(graph["event"].get(field)) == _normalize(expected_times[field])
            for field in time_fields
        ]
        temporal_match = statistics.mean(float(item) for item in time_checks) if time_checks else 1.0
        temporal_matches.append(temporal_match)

        expected_evidence = list(expected.get("evidence", []))
        projected_evidence = list(graph.get("evidence_links", []))
        evidence_checks = [
            any(
                _normalize(actual.get("source_uri")) == _normalize(label["source_uri"])
                and (
                    not label.get("source_sha256")
                    or _normalize(actual.get("source_sha256"))
                    == _normalize(label["source_sha256"])
                )
                and (
                    not label.get("evidence_role")
                    or _normalize(actual.get("evidence_role"))
                    == _normalize(label["evidence_role"])
                )
                for actual in projected_evidence
            )
            for label in expected_evidence
        ]
        evidence_match = (
            statistics.mean(float(item) for item in evidence_checks)
            if evidence_checks
            else 1.0
        )
        evidence_matches.append(evidence_match)
        details.append(
            {
                "case_id": case_id,
                "event_type_match": event_match,
                "effective_time_accuracy": temporal_match,
                "evidence_coverage": evidence_match,
                "entity": _f1(expected_case_entities, predicted_case_entities),
                "event_relation": _f1(
                    expected_case_relations, predicted_case_relations
                ),
            }
        )

    entity = _f1(expected_entities, predicted_entities)
    relation = _f1(expected_relations, predicted_relations)
    event_type_accuracy = statistics.mean(event_type_matches)
    effective_time_accuracy = statistics.mean(temporal_matches)
    evidence_coverage = statistics.mean(evidence_matches)
    component_scores = {
        "entity_normalization": 30.0 * entity["f1"],
        "event_type": 20.0 * event_type_accuracy,
        "effective_time": 15.0 * effective_time_accuracy,
        "event_relation": 20.0 * relation["f1"],
        "evidence_coverage": 15.0 * evidence_coverage,
    }
    total_score = sum(component_scores.values())
    hard_gates = {
        "all_expected_evidence_linked": evidence_coverage == 1.0,
        "no_event_type_mismatch": event_type_accuracy == 1.0,
    }
    return {
        "case_count": len(cases),
        "entity": entity,
        "event_relation": relation,
        "event_type_accuracy": event_type_accuracy,
        "effective_time_accuracy": effective_time_accuracy,
        "evidence_coverage": evidence_coverage,
        "component_scores": component_scores,
        "total_score": total_score,
        "minimum_score": float(min_score),
        "hard_gates": hard_gates,
        "hard_gates_passed": all(hard_gates.values()),
        "passed": all(hard_gates.values()) and total_score >= float(min_score),
        "details": details,
        "interpretation": (
            "Scores are valid only for the supplied independently labelled, "
            "frozen research scenarios."
        ),
    }
