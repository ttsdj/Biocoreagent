---
slug: protocol-extraction
title: Protocol Extraction from Documents
summary: Extract structured ProtocolCard, TaskCard, and ResourceCard data from bioinformatics protocol documents, papers, READMEs, and workflow documentation.
tags: protocol, extraction, rag, evidence
---

# Protocol Extraction

## When To Use

Use this skill when:
1. The user uploads or points to a bioinformatics protocol document.
2. The user asks you to summarise a protocol or paper methods section.
3. The user asks what tools, databases, or resources a study requires.
4. The user wants to know what is needed to reproduce a specific analysis.

## Procedure

1. **Ingest the document** — use `bio_ingest_protocol` to read and chunk the document into the evidence store.
2. **Review the chunks** — use `bio_query_evidence` with `search_type="chunks"` to examine the document content.
3. **Extract structured cards** — use `bio_extract_protocol` to create:
   - A `ProtocolCard` summarising research goal, organism, assay type, inputs, and outputs.
   - One `TaskCard` per computational step (QC, alignment, quantification, etc.).
   - One `ResourceCard` per tool, database, reference genome, or software package.
4. **Cite every field** — each extracted value must reference an `EvidenceSpan` pointing to the source chunk and line range.
5. **List gaps explicitly** — any information not found in the source goes into `missing_information`.

## Hard Rules

1. **Never invent** biological data, organism names, tool versions, database names, or parameter values.
2. **Never guess** sample names, reference genomes, or sequencing platforms.
3. **Never fabricate** file paths or URLs that are not in the source.
4. If the source says "standard settings" without specifics, list those parameters in `missing_information`.
5. If the source mentions a tool without a version, leave `version` empty — do not look up the latest version.
6. Every important field must carry at least one `EvidenceSpan`.

## Pitfalls

- Protocols often omit version numbers for tools and databases.  Do not fill them in from memory.
- Reference genome assemblies (hg19, GRCh38, mm10, etc.) are frequently mentioned without full URLs — list the URL as missing if not provided.
- Adapter sequences and primer sequences may be implied by kit names; note the kit but do not invent sequences.
- Some protocols describe optional steps ambiguously — flag these in `notes` on the TaskCard.

## Interview Notes

Q: What if the protocol is very vague (e.g. "analyse with standard pipeline")?
A: Extract what is explicit; list all gaps in `missing_information`.  The replication planning step will determine whether automation is possible.

Q: Should I extract citations and references?
A: Yes, as `ResourceCard` entries with `resource_type="reference"`.  Include the DOI or PMID if present.

Q: What about figures and tables?
A: Reference them by number and describe their content if it adds evidence for task parameters.  Do not attempt to parse bitmap images.

## Source Task

Protocol RAG document ingestion pipeline — Phase 1 of SciCoreAgent RAG extension.
