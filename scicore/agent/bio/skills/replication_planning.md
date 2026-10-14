---
slug: replication-planning
title: Replication Planning from Protocol Evidence
summary: Generate a structured replication plan from evidence-store ProtocolCards, TaskCards, and ResourceCards. Separates automatable steps from those requiring manual confirmation.
tags: replication, planning, evidence, rag
---

# Replication Planning

## When To Use

Use this skill when:
1. The user asks "can you help me reproduce this study?"
2. The user requests a replication plan after a protocol has been ingested and extracted.
3. You need to assess whether sufficient evidence exists to automate a bioinformatics workflow.
4. The user wants a structured breakdown of what can and cannot be automated.

## Procedure

1. **Gather evidence** — use `bio_query_evidence` to retrieve all ProtocolCards, TaskCards, and ResourceCards relevant to the target study.
2. **Review completeness** — check each TaskCard for:
   - Are input files specified with real paths?
   - Are tool versions known?
   - Are database/reference genome versions known?
   - Are key parameters documented?
   - Is the computational environment (conda, Docker, modules) described?
3. **Build the ReplicationPlan** — use `bio_replication_plan` with:
   - `protocol_ids`: the protocols you are planning against.
   - `plan_json`: the structured ReplicationPlan (constructed by you, the LLM, from evidence).
4. **Separate concerns**:
   - `automatable_steps`: steps that can be scripted or run by Snakemake/Nextflow.
   - `manual_confirmation_required`: steps that need human review (e.g. checking QC reports, verifying sample metadata).
   - `cannot_proceed_reasons`: blockers that prevent any automation (e.g. missing input file paths).
5. **Present the plan** — follow the output format below.

## Output Format

When the user asks for a replication plan, present it structured as:

```
# Research Replication Plan

## 1. Can this be replicated?
[Yes / Partially / No — with reasoning]

## 2. Reasons automation is blocked (if any)
- ...

## 3. Required inputs
- ...

## 4. Required tools
| Tool | Version | Purpose | Evidence |
|---|---|---|---|

## 5. Required databases
| Database | Version | Purpose | Evidence |
|---|---|---|---|

## 6. Task breakdown
| Step | Task | Tool | Input | Output | Automatable? |
|---|---|---|---|---|---|

## 7. Automatable steps
- ...

## 8. Steps requiring manual confirmation
- ...

## 9. Code/workflow that CAN be generated now
- ...

## 10. Code/workflow that CANNOT be generated yet
- ... (with reasons)

## 11. Next action recommendation
- ...
```

## Hard Rules

1. Do NOT claim a study has been replicated unless actual commands were executed and output was observed.
2. Do NOT claim a pipeline will succeed unless it has actually been run.
3. If critical experimental details are missing, do NOT generate an executable plan — generate a `cannot_proceed` explanation.
4. If only reference information is missing, generate a comparison workflow (not an execution workflow).
5. If input files are missing, generate a parameterised script (not an execution workflow).
6. All conclusions must distinguish `observed evidence` from `inferred planning`.

## Pitfalls

- It is tempting to fill in "standard" parameter values.  Resist — list them as missing.
- Different protocols may use different reference genome assemblies for the same species.  Flag this as a compatibility issue.
- Tool versions can change results dramatically (e.g. STAR 2.7.9a vs 2.7.11b).  Unknown versions are a real blocker for exact reproduction.

## Interview Notes

Q: What confidence level is needed to generate a workflow?
A: All tasks should have confidence "high" for the workflow to be executable.  "medium" or "low" confidence tasks should be flagged for manual review.

Q: What if two protocols conflict?
A: Note the conflict explicitly and ask the user which version they want to follow.

## Source Task

Replication planning from protocol evidence — Phase 1 of SciCoreAgent RAG extension.
