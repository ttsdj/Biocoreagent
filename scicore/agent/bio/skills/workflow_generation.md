---
slug: workflow-generation
title: Workflow Generation from Replication Plans
summary: Generate Snakemake, Nextflow, or Python workflow drafts from a completed ReplicationPlan when evidence is sufficient. Phase 1 generates drafts only — no automatic execution.
tags: workflow, snakemake, nextflow, pipeline, generation
---

# Workflow Generation

## When To Use

Use this skill when:
1. A ReplicationPlan exists with sufficient evidence (all tasks confidence "high").
2. The user explicitly requests workflow/pipeline code generation.
3. The user asks for a Snakemake, Nextflow, or Python script to automate a documented analysis.

## Pre-flight Checklist

Before generating any workflow code, verify ALL of the following:

1. ☐ Are input file paths known (real paths, not placeholders)?
2. ☐ Are sample names/metadata available?
3. ☐ Is the reference genome specified with version?
4. ☐ Are reference/annotation file paths or download URLs known?
5. ☐ Are tool/software choices documented with versions?
6. ☐ Are parameter values documented?
7. ☐ Is the output directory structure defined?
8. ☐ Is the sequencing read type known (single-end vs paired-end)?
9. ☐ Are key parameter thresholds documented (e.g. q-value cutoff, fold-change threshold)?
10. ☐ Has the user explicitly confirmed they want a workflow generated?

## Procedure

1. **Verify the pre-flight checklist** — go through every item above using evidence from the store.
2. **If information is insufficient:**
   - Do NOT generate a workflow.
   - Output exactly:
     ```
     Cannot generate executable workflow.
     Missing information:
     - [list each missing item with what is needed]
     ```
3. **If information is sufficient:**
   - Generate a workflow DRAFT.
   - Annotate every step with its source EvidenceSpan.
   - Include validation checks (e.g. `if` guards for expected output files).
   - Default to `--dry-run` or `echo` placeholders for destructive operations.

## Output Format

For Snakemake:

```python
# Workflow: [study name]
# Generated from evidence: [protocol_ids]
# Status: DRAFT — review before execution
#
# Evidence reference:
#   Step 1: [source_id] [chunk_id] [location]
#   Step 2: ...

rule all:
    input:
        ...

rule step_1:
    input:
        ...
    output:
        ...
    params:
        ...
    shell:
        """
        [command with explicit parameters]
        """
```

For Nextflow:

```nextflow
// Workflow: [study name]
// Generated from evidence: [protocol_ids]
// Status: DRAFT — review before execution
```

## Hard Rules

1. Do NOT generate workflow code if the pre-flight checklist is not fully satisfied.
2. Every rule/process must cite its evidence source as a comment.
3. Do NOT execute the generated workflow unless the user explicitly requests it.
4. Do NOT hard-code paths specific to your machine.
5. Use `{input}`, `{output}`, and `{params}` placeholders (Snakemake) or channel variables (Nextflow) — never literal paths.
6. Include `--dry-run` flags or `echo` protection around commands that modify data.
7. Never embed API keys, passwords, or access tokens in generated workflows.

## Pitfalls

- Generated workflows may reference tools not installed on the user's system.  Do not assume availability — mention requirements in a comment.
- Resource requirements (memory, CPUs, wall time) are rarely documented in protocols.  Use conservative defaults and flag them as estimates.
- Paired-end vs single-end assumptions can break an entire pipeline.  Verify from the protocol evidence.

## Interview Notes

Q: Should I install Snakemake/Nextflow for the user?
A: No — Phase 1 only generates workflow drafts.  The user decides whether and how to execute them.

Q: What workflow language should I default to?
A: Ask the user.  If they don't specify, default to Snakemake (most common in bioinformatics) and explain the choice.

Q: Can I generate a Python script instead?
A: Yes, if the workflow is linear and simple.  For multi-step pipelines with branching, prefer Snakemake or Nextflow.

## Source Task

Workflow generation from replication plans — Phase 1 of SciCoreAgent RAG extension.
