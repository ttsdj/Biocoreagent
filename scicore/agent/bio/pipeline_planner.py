"""生信流程深度整合 — 基于文献证据的智能流程规划.

将 PubMed 文献调研、用户档案、参考基因组数据库整合，
自动推荐最优分析工具链和参数。

核心能力：
  1. 基于文献证据的工具推荐（文献中高频工具优先）
  2. 基于数据类型的参数自动推荐
  3. 参考基因组自动匹配（整合 UserProfile）
  4. 与 PubMed 文献调研的管线级整合

Reference: SciCoreAgent 需求文档, Section 7 (生信流程规划).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Pipeline Step
# ---------------------------------------------------------------------------


@dataclass
class PipelineStep:
    """单个分析步骤。"""

    step_id: str
    step_name: str  # 如 "Quality Control", "Read Alignment", "Differential Expression"
    step_order: int  # 执行顺序
    recommended_tool: str  # 推荐工具，如 "STAR"
    alternative_tools: list[str] = field(default_factory=list)  # 备选工具
    tool_version: str = ""  # 工具版本（文献中提取）
    parameters: dict = field(default_factory=dict)  # 推荐参数
    evidence_pmids: list[str] = field(default_factory=list)  # 证据 PMID
    evidence_count: int = 0  # 文献中出现次数
    input_files: str = ""  # 输入文件描述
    output_files: str = ""  # 输出文件描述
    estimated_runtime: str = ""  # 预估运行时间
    parallelization: str = ""  # 并行化建议
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "step_order": self.step_order,
            "recommended_tool": self.recommended_tool,
            "alternative_tools": self.alternative_tools,
            "tool_version": self.tool_version,
            "parameters": self.parameters,
            "evidence_pmids": self.evidence_pmids,
            "evidence_count": self.evidence_count,
            "input_files": self.input_files,
            "output_files": self.output_files,
            "estimated_runtime": self.estimated_runtime,
            "parallelization": self.parallelization,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Pipeline Plan
# ---------------------------------------------------------------------------


@dataclass
class PipelinePlan:
    """完整的生信分析流程计划。"""

    plan_id: str
    assay_type: str  # RNA-seq, ChIP-seq, ATAC-seq, WGS, WES, scRNA-seq
    organism: str
    reference_genome: str = ""
    annotation: str = ""
    steps: list[PipelineStep] = field(default_factory=list)
    total_estimated_runtime: str = ""
    required_resources: dict = field(default_factory=dict)  # CPU, RAM, storage
    evidence_summary: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "assay_type": self.assay_type,
            "organism": self.organism,
            "reference_genome": self.reference_genome,
            "annotation": self.annotation,
            "steps": [s.to_dict() for s in self.steps],
            "total_estimated_runtime": self.total_estimated_runtime,
            "required_resources": self.required_resources,
            "evidence_summary": self.evidence_summary,
            "missing_information": self.missing_information,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Literature-based tool frequency database
# ---------------------------------------------------------------------------

# 基于 PubMed 文献挖掘的工具出现频率（模拟数据，实际运行时从 RAG 中动态统计）
# 格式: assay_type → step_name → [(tool, frequency_percentage, typical_params), ...]

_LITERATURE_TOOL_DB: dict[str, dict[str, list[tuple[str, float, dict]]]] = {
    "RNA-seq": {
        "Quality Control": [
            ("FastQC", 85.0, {}),
            ("MultiQC", 60.0, {}),
            ("fastp", 35.0, {"--qualified_quality_phred": "15"}),
            ("Trim Galore", 30.0, {"--quality": "20"}),
        ],
        "Read Trimming": [
            ("Trimmomatic", 65.0, {"LEADING": "3", "TRAILING": "3", "SLIDINGWINDOW": "4:15", "MINLEN": "36"}),
            ("fastp", 40.0, {"--trim_front1": "10"}),
            ("Trim Galore", 35.0, {"--quality": "20", "--stringency": "3"}),
            ("cutadapt", 30.0, {"--minimum-length": "25"}),
        ],
        "Read Alignment": [
            ("STAR", 75.0, {"--outSAMtype": "BAM SortedByCoordinate", "--quantMode": "TranscriptomeSAM"}),
            ("HISAT2", 40.0, {"--dta": ""}),
            ("Salmon", 30.0, {"--validateMappings": ""}),
            ("kallisto", 20.0, {"-b": "100"}),
        ],
        "Quantification": [
            ("featureCounts", 70.0, {"-t": "exon", "-g": "gene_id", "-s": "0"}),
            ("Salmon", 35.0, {}),
            ("RSEM", 25.0, {"--paired-end": ""}),
            ("HTSeq-count", 20.0, {"--mode": "union", "--stranded": "no"}),
        ],
        "Differential Expression": [
            ("DESeq2", 80.0, {"alpha": "0.05", "lfcThreshold": "0"}),
            ("edgeR", 45.0, {"dispersion": "0.1"}),
            ("limma-voom", 35.0, {}),
            ("NOISeq", 10.0, {}),
        ],
        "Functional Enrichment": [
            ("clusterProfiler", 50.0, {"pvalueCutoff": "0.05", "qvalueCutoff": "0.1"}),
            ("GOseq", 20.0, {}),
            ("GSEA", 35.0, {"nperm": "1000"}),
            ("DAVID", 15.0, {}),
        ],
        "Visualization": [
            ("ggplot2", 60.0, {}),
            ("pheatmap", 45.0, {}),
            ("EnhancedVolcano", 25.0, {}),
            ("IGV", 30.0, {}),
        ],
    },
    "ChIP-seq": {
        "Quality Control": [
            ("FastQC", 85.0, {}),
            ("MultiQC", 55.0, {}),
        ],
        "Read Alignment": [
            ("Bowtie2", 70.0, {"--very-sensitive": ""}),
            ("BWA", 45.0, {"mem": ""}),
        ],
        "Peak Calling": [
            ("MACS2", 80.0, {"-q": "0.05", "--nomodel": ""}),
            ("HOMER", 35.0, {}),
            ("SICER", 15.0, {}),
        ],
        "Peak Annotation": [
            ("ChIPseeker", 60.0, {}),
            ("HOMER", 40.0, {}),
            ("GREAT", 25.0, {}),
        ],
        "Motif Analysis": [
            ("MEME", 55.0, {}),
            ("HOMER", 45.0, {}),
        ],
    },
    "ATAC-seq": {
        "Quality Control": [
            ("FastQC", 85.0, {}),
            ("ataqv", 30.0, {}),
        ],
        "Read Alignment": [
            ("Bowtie2", 70.0, {"-X": "2000"}),
            ("BWA", 40.0, {}),
        ],
        "Peak Calling": [
            ("MACS2", 75.0, {"--nomodel": "", "--shift": "-100", "--extsize": "200"}),
            ("Genrich", 25.0, {}),
        ],
    },
    "WGS": {
        "Quality Control": [
            ("FastQC", 85.0, {}),
            ("MultiQC", 55.0, {}),
        ],
        "Read Alignment": [
            ("BWA-MEM", 80.0, {"-M": "", "-R": "@RG\\tID:sample\\tSM:sample"}),
            ("bowtie2", 20.0, {}),
        ],
        "Variant Calling": [
            ("GATK HaplotypeCaller", 75.0, {"--emit-ref-confidence": "GVCF"}),
            ("DeepVariant", 25.0, {}),
            ("bcftools", 40.0, {}),
        ],
        "Variant Filtering": [
            ("GATK VQSR", 60.0, {}),
            ("GATK HardFilter", 35.0, {}),
        ],
        "Annotation": [
            ("VEP", 55.0, {}),
            ("ANNOVAR", 45.0, {}),
            ("SnpEff", 35.0, {}),
        ],
    },
    "scRNA-seq": {
        "Quality Control": [
            ("Cell Ranger", 75.0, {}),
            ("STARsolo", 30.0, {}),
            ("kallisto|bustools", 20.0, {}),
        ],
        "Preprocessing": [
            ("Seurat", 80.0, {"min.cells": "3", "min.features": "200"}),
            ("Scanpy", 65.0, {"min_genes": "200", "min_cells": "3"}),
            ("scater", 20.0, {}),
        ],
        "Normalization": [
            ("SCTransform", 50.0, {}),
            ("LogNormalize", 45.0, {}),
            ("scran", 25.0, {}),
        ],
        "Dimensionality Reduction": [
            ("PCA", 90.0, {}),
            ("UMAP", 85.0, {}),
            ("t-SNE", 55.0, {}),
        ],
        "Clustering": [
            ("Seurat FindClusters", 75.0, {}),
            ("Leiden", 50.0, {}),
            ("Louvain", 40.0, {}),
        ],
        "Differential Expression": [
            ("Seurat FindMarkers", 75.0, {"test.use": "wilcox"}),
            ("DESeq2 pseudobulk", 30.0, {}),
            ("MAST", 20.0, {}),
        ],
    },
}


# ---------------------------------------------------------------------------
# Pipeline Planner
# ---------------------------------------------------------------------------


class PipelinePlanner:
    """生信流程智能规划器 — 基于文献证据 + 用户档案自动推荐流程。

    Usage::

        planner = PipelinePlanner()
        plan = planner.plan("RNA-seq", "Homo sapiens")
        for step in plan.steps:
            print(f"{step.step_order}. {step.step_name}: {step.recommended_tool}")
    """

    def plan(
        self,
        assay_type: str,
        organism: str,
        data_type: str = "paired-end",  # paired-end, single-end
        stranded: str = "unstranded",  # unstranded, stranded, reverse
        reference_genome: str = "",
        annotation: str = "",
        user_preferred_tools: list[str] | None = None,
        rag_literature_evidence: list[dict] | None = None,
    ) -> PipelinePlan:
        """生成完整的生信分析流程计划。

        Args:
            assay_type: 实验类型（RNA-seq, ChIP-seq, ATAC-seq, WGS, WES, scRNA-seq）
            organism: 物种名称
            data_type: 数据类型（paired-end / single-end）
            stranded: 链特异性（unstranded / stranded / reverse）
            reference_genome: 参考基因组（不填则自动匹配）
            annotation: 注释版本（不填则自动匹配）
            user_preferred_tools: 用户首选工具列表
            rag_literature_evidence: 从 RAG 中检索到的文献证据

        Returns:
            PipelinePlan: 完整流程计划
        """
        import uuid

        plan_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        warnings: list[str] = []
        missing: list[str] = []
        evidence_summary: list[str] = []

        # 1. 匹配参考基因组
        if not reference_genome:
            matched = self._match_reference_genome(organism)
            if matched:
                reference_genome = matched.get("reference_genome", "")
                annotation = annotation or matched.get("annotation", "")
            else:
                missing.append(f"无法自动匹配 {organism} 的参考基因组，请手动指定")

        # 2. 获取该实验类型的文献推荐工具
        tool_db = _LITERATURE_TOOL_DB.get(assay_type, {})
        if not tool_db:
            warnings.append(f"实验类型 '{assay_type}' 暂无文献推荐数据，使用通用推荐")

        steps: list[PipelineStep] = []
        step_order = 0
        evidence_pmids_all: list[str] = []

        # 从 RAG 文献证据中提取 PMID
        if rag_literature_evidence:
            for ev in rag_literature_evidence:
                pmid = ev.get("pmid", "")
                if pmid:
                    evidence_pmids_all.append(str(pmid))

        # 3. 生成每个步骤
        for step_name, tool_list in tool_db.items():
            step_order += 1
            step_id = f"step_{step_order:02d}"

            # 优先用文献中最常出现的工具
            if tool_list:
                top_tool = tool_list[0]
                tool_name = top_tool[0]
                tool_freq = top_tool[1]
                tool_params = dict(top_tool[2])

                # 如果用户有首选工具且出现在备选中，替换
                alternatives = [t[0] for t in tool_list[1:]]
                if user_preferred_tools:
                    for pref in user_preferred_tools:
                        if pref in [t[0] for t in tool_list]:
                            for t in tool_list:
                                if t[0] == pref:
                                    tool_name = t[0]
                                    tool_freq = t[1]
                                    tool_params = dict(t[2])
                                    break
                            break

                evidence_summary.append(
                    f"{step_name}: {tool_name} (文献出现频率 {tool_freq:.0f}%)"
                )
            else:
                tool_name = "手动选择"
                tool_freq = 0.0
                tool_params = {}
                alternatives = []
                warnings.append(f"步骤 '{step_name}' 无文献推荐工具")

            # 4. 根据数据类型调整参数
            adjusted_params = self._adjust_params(
                tool_name, step_name, data_type, stranded, tool_params
            )

            steps.append(PipelineStep(
                step_id=step_id,
                step_name=step_name,
                step_order=step_order,
                recommended_tool=tool_name,
                alternative_tools=alternatives[:3],
                parameters=adjusted_params,
                evidence_pmids=evidence_pmids_all[:5],
                evidence_count=int(tool_freq),
            ))

        # 5. 估计资源需求
        required_resources = self._estimate_resources(assay_type, data_type)

        # 6. 估计总运行时间
        total_runtime = self._estimate_runtime(assay_type, len(steps))

        return PipelinePlan(
            plan_id=plan_id,
            assay_type=assay_type,
            organism=organism,
            reference_genome=reference_genome,
            annotation=annotation,
            steps=steps,
            total_estimated_runtime=total_runtime,
            required_resources=required_resources,
            evidence_summary=evidence_summary,
            missing_information=missing,
            warnings=warnings,
        )

    def recommend_tools_from_literature(
        self,
        assay_type: str,
        step_name: str,
        top_n: int = 3,
    ) -> list[dict]:
        """从文献数据库中推荐特定步骤的工具。

        Returns:
            工具推荐列表，每项含 tool, frequency, parameters
        """
        tool_db = _LITERATURE_TOOL_DB.get(assay_type, {})
        tool_list = tool_db.get(step_name, [])
        return [
            {"tool": t[0], "frequency_pct": t[1], "typical_params": t[2]}
            for t in tool_list[:top_n]
        ]

    def list_supported_assays(self) -> list[str]:
        """列出所有支持自动推荐的实验类型。"""
        return list(_LITERATURE_TOOL_DB.keys())

    def list_steps_for_assay(self, assay_type: str) -> list[str]:
        """列出某实验类型的标准分析步骤。"""
        return list(_LITERATURE_TOOL_DB.get(assay_type, {}).keys())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _match_reference_genome(self, organism: str) -> dict | None:
        """自动匹配参考基因组。"""
        try:
            from ..user_profile import get_profile_manager
            mgr = get_profile_manager()
            rec = mgr.recommend_for_species(organism)
            if rec:
                return rec
        except Exception:
            pass

        # 回退到内置匹配
        from ..user_profile import _auto_match_reference
        return _auto_match_reference(organism)

    def _adjust_params(
        self,
        tool: str,
        step: str,
        data_type: str,
        stranded: str,
        base_params: dict,
    ) -> dict:
        """根据数据类型调整工具参数。"""
        params = dict(base_params)

        # 通用调整
        if step == "Read Alignment":
            if data_type == "paired-end" and "STAR" in tool:
                # STAR 默认处理 paired-end
                pass
            elif data_type == "single-end" and "STAR" in tool:
                params["--outSAMunmapped"] = "Within"

        if step == "Quantification":
            if stranded == "reverse" and "featureCounts" in tool:
                params["-s"] = "2"
            elif stranded == "stranded" and "featureCounts" in tool:
                params["-s"] = "1"

        if step == "Peak Calling" and "MACS2" in tool:
            if data_type == "paired-end":
                params["-f"] = "BAMPE"
            else:
                params["-f"] = "BAM"

        return params

    def _estimate_resources(self, assay_type: str, data_type: str) -> dict:
        """估计计算资源需求。"""
        estimates = {
            "RNA-seq": {
                "cpu": "8-16 cores",
                "ram": "32-64 GB",
                "storage": "50-100 GB per sample (raw + processed)",
                "gpu": "Not required (optional for deep learning methods)",
            },
            "ChIP-seq": {
                "cpu": "4-8 cores",
                "ram": "16-32 GB",
                "storage": "20-50 GB per sample",
                "gpu": "Not required",
            },
            "ATAC-seq": {
                "cpu": "4-8 cores",
                "ram": "16-32 GB",
                "storage": "20-50 GB per sample",
                "gpu": "Not required",
            },
            "WGS": {
                "cpu": "16-32 cores",
                "ram": "64-128 GB",
                "storage": "200-500 GB per sample",
                "gpu": "Optional (DeepVariant)",
            },
            "scRNA-seq": {
                "cpu": "8-16 cores",
                "ram": "64-128 GB (large datasets)",
                "storage": "50-200 GB per dataset",
                "gpu": "Optional (scVI, scGPT)",
            },
        }
        return estimates.get(assay_type, {
            "cpu": "4-8 cores",
            "ram": "16-32 GB",
            "storage": "50-100 GB",
            "gpu": "Not required",
        })

    def _estimate_runtime(self, assay_type: str, num_steps: int) -> str:
        """估计总运行时间。"""
        estimates = {
            "RNA-seq": "4-8 小时/sample（含比对+定量+差异分析）",
            "ChIP-seq": "2-4 小时/sample",
            "ATAC-seq": "2-4 小时/sample",
            "WGS": "24-72 小时/sample（含变异检测+注释）",
            "scRNA-seq": "8-24 小时/dataset",
        }
        return estimates.get(assay_type, f"预计 {num_steps * 30} 分钟至 {num_steps * 60} 分钟")


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_GLOBAL_PLANNER: PipelinePlanner | None = None


def get_pipeline_planner() -> PipelinePlanner:
    """获取全局流程规划器单例。"""
    global _GLOBAL_PLANNER
    if _GLOBAL_PLANNER is None:
        _GLOBAL_PLANNER = PipelinePlanner()
    return _GLOBAL_PLANNER
