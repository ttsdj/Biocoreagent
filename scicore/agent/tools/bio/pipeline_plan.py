"""生信流程智能规划工具 — 基于文献证据的流程推荐和实验计划生成。"""

from __future__ import annotations

import json

from ..base import Tool
from ...bio.pipeline_planner import get_pipeline_planner


class BioPipelinePlanTool(Tool):
    """基于文献证据生成生信分析流程计划。"""

    name = "bio_pipeline_plan"
    description = (
        "基于文献证据和用户档案，自动推荐生信分析流程。"
        "根据实验类型（RNA-seq/ChIP-seq/ATAC-seq/WGS/scRNA-seq），"
        "自动匹配最优工具链、参数、参考基因组和计算资源需求。"
        "所有推荐都标注了文献中出现频率作为证据支撑。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "assay_type": {
                "type": "string",
                "description": "实验类型: RNA-seq, ChIP-seq, ATAC-seq, WGS, WES, scRNA-seq",
                "enum": ["RNA-seq", "ChIP-seq", "ATAC-seq", "WGS", "WES", "scRNA-seq"],
            },
            "organism": {
                "type": "string",
                "description": "物种名称，如 'Homo sapiens', 'Mus musculus', 'Arabidopsis thaliana'",
            },
            "data_type": {
                "type": "string",
                "description": "测序数据类型: paired-end 或 single-end",
                "enum": ["paired-end", "single-end"],
                "default": "paired-end",
            },
            "stranded": {
                "type": "string",
                "description": "链特异性: unstranded, stranded, reverse",
                "enum": ["unstranded", "stranded", "reverse"],
                "default": "unstranded",
            },
            "reference_genome": {
                "type": "string",
                "description": "参考基因组版本（不填则自动匹配），如 'GRCh38/hg38'",
            },
            "preferred_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "用户首选工具列表，如 ['STAR', 'DESeq2']",
            },
        },
        "required": ["assay_type", "organism"],
    }

    def execute(
        self,
        assay_type: str,
        organism: str,
        data_type: str = "paired-end",
        stranded: str = "unstranded",
        reference_genome: str = "",
        preferred_tools: list[str] | None = None,
    ) -> str:
        planner = get_pipeline_planner()

        plan = planner.plan(
            assay_type=assay_type,
            organism=organism,
            data_type=data_type,
            stranded=stranded,
            reference_genome=reference_genome,
            user_preferred_tools=preferred_tools,
        )

        return json.dumps({
            "success": True,
            "plan": plan.to_dict(),
        }, ensure_ascii=False, indent=2)


class BioPipelineSupportedAssaysTool(Tool):
    """列出所有支持自动推荐的实验类型和分析步骤。"""

    name = "bio_pipeline_supported"
    description = (
        "列出 PipelinePlanner 支持自动推荐的实验类型及每类实验的标准分析步骤。"
        "用于了解系统能为哪些实验提供自动流程规划。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "assay_type": {
                "type": "string",
                "description": "查看特定实验类型的步骤（可选，不填则列出所有类型）",
            },
        },
    }

    def execute(self, assay_type: str = "") -> str:
        planner = get_pipeline_planner()

        if assay_type:
            steps = planner.list_steps_for_assay(assay_type)
            return json.dumps({
                "assay_type": assay_type,
                "steps": steps,
                "step_count": len(steps),
            }, ensure_ascii=False, indent=2)

        assays = planner.list_supported_assays()
        result = {}
        for assay in assays:
            steps = planner.list_steps_for_assay(assay)
            result[assay] = steps

        return json.dumps({
            "supported_assays": result,
            "total_assays": len(assays),
        }, ensure_ascii=False, indent=2)


class BioExperimentPlanTool(Tool):
    """从研究问题生成完整实验方案。"""

    name = "bio_experiment_plan"
    description = (
        "从研究问题生成完整的实验方案，包括：研究假设、实验设计（对照/处理/重复）、"
        "材料清单、实验步骤时间线、测序策略、生信分析流程、预期结果和可能的风险。"
        "整合文献证据、用户档案和 Protocol RAG 中的知识。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "research_question": {
                "type": "string",
                "description": "研究问题，如 '肝癌细胞系中药物X处理前后的转录组变化'",
            },
            "organism": {
                "type": "string",
                "description": "研究物种/细胞系，如 'Homo sapiens / HepG2'",
            },
            "assay_type": {
                "type": "string",
                "description": "主要实验技术: RNA-seq, ChIP-seq, ATAC-seq, WGS, scRNA-seq",
            },
            "conditions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "实验条件/分组，如 ['对照组 (DMSO)', '处理组 (药物X 10μM, 24h)']",
            },
            "replicates": {
                "type": "integer",
                "description": "每组生物学重复数（默认从用户档案读取）",
            },
            "additional_requirements": {
                "type": "string",
                "description": "额外需求（可选），如 '需要 mRNA 富集', '链特异性文库'",
            },
        },
        "required": ["research_question", "organism", "assay_type", "conditions"],
    }

    def execute(
        self,
        research_question: str,
        organism: str,
        assay_type: str,
        conditions: list[str],
        replicates: int = 0,
        additional_requirements: str = "",
    ) -> str:
        import uuid
        from ...user_profile import get_profile_manager

        plan_id = f"exp_{uuid.uuid4().hex[:8]}"
        warnings: list[str] = []
        missing: list[str] = []

        # 1. 获取用户档案
        mgr = get_profile_manager()
        if replicates <= 0:
            replicates = mgr.profile.default_replicates if mgr.is_configured() else 3

        # 2. 推荐测序策略
        seq_rec = {}
        try:
            seq_rec = mgr.recommend_for_assay(assay_type)
        except Exception:
            warnings.append("无法获取测序推荐（用户档案未配置）")

        # 3. 匹配参考基因组
        ref_genome = {}
        try:
            ref_genome = mgr.recommend_for_species(organism) or {}
        except Exception:
            pass

        # 4. 获取工作流推荐
        pipeline_plan = None
        try:
            planner = get_pipeline_planner()
            pipeline_plan = planner.plan(
                assay_type=assay_type,
                organism=organism,
            )
        except Exception as e:
            warnings.append(f"流程推荐失败: {e}")

        # 5. 构建实验方案
        if not ref_genome:
            missing.append(f"参考基因组（{organism}）：请手动指定")

        timeline = self._build_timeline(assay_type, len(conditions), replicates)

        experiment_plan = {
            "plan_id": plan_id,
            "research_question": research_question,
            "organism": organism,
            "assay_type": assay_type,
            "design": {
                "conditions": conditions,
                "replicates_per_condition": replicates,
                "total_samples": len(conditions) * replicates,
                "type": "controlled experiment" if len(conditions) >= 2 else "single condition profiling",
            },
            "reference_genome": ref_genome,
            "sequencing_strategy": seq_rec,
            "materials": self._suggest_materials(assay_type, organism, conditions),
            "timeline": timeline,
            "pipeline": pipeline_plan.to_dict() if pipeline_plan else None,
            "quality_controls": self._suggest_qc(assay_type),
            "expected_results": self._suggest_expected_results(assay_type, len(conditions)),
            "potential_risks": self._suggest_risks(assay_type),
            "missing_information": missing,
            "warnings": warnings,
        }

        return json.dumps({
            "success": True,
            "experiment_plan": experiment_plan,
        }, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_timeline(self, assay_type: str, num_conditions: int, replicates: int) -> list[dict]:
        """构建实验时间线。"""
        total_samples = num_conditions * replicates
        timeline = [
            {"day": 1, "task": "样品准备与 RNA 提取", "duration": "1-2 天", "depends_on": []},
            {"day": 3, "task": "RNA 质量检测 (Bioanalyzer/NanoDrop)", "duration": "0.5 天", "depends_on": [1]},
            {"day": 4, "task": f"文库构建 ({total_samples} 个样品)", "duration": f"{max(1, total_samples // 8)} 天", "depends_on": [3]},
            {"day": 5, "task": "文库质检 (Qubit/qPCR)", "duration": "0.5 天", "depends_on": [4]},
            {"day": 6, "task": "上机测序", "duration": "1-3 天（视通量）", "depends_on": [5]},
            {"day": 9, "task": "数据下机 + FastQC 质控", "duration": "1 天", "depends_on": [6]},
            {"day": 10, "task": f"生物信息学分析（比对→定量→差异分析）", "duration": "3-5 天", "depends_on": [9]},
            {"day": 15, "task": "结果整理与可视化", "duration": "1-2 天", "depends_on": [10]},
        ]

        if assay_type == "scRNA-seq":
            timeline.insert(3, {"day": 3, "task": "单细胞悬液制备", "duration": "0.5 天", "depends_on": [1]})
            timeline.insert(4, {"day": 3, "task": "10x Genomics 油滴生成 + 文库构建", "duration": "1 天", "depends_on": [3]})

        return timeline

    def _suggest_materials(self, assay_type: str, organism: str, conditions: list[str]) -> list[str]:
        """推荐实验材料。"""
        materials = [
            f"细胞/组织样品: {organism}",
            f"实验条件: {', '.join(conditions)}",
        ]

        if "RNA-seq" in assay_type:
            materials.extend([
                "TRIzol 或等效 RNA 提取试剂",
                "DNase I（去除基因组 DNA）",
                "RNA 文库构建试剂盒（如 NEBNext Ultra II）",
                "Qubit RNA HS Assay Kit",
                "Bioanalyzer RNA 6000 Nano Kit",
            ])
        elif "ChIP-seq" in assay_type:
            materials.extend([
                "ChIP 级抗体",
                "Protein A/G 磁珠",
                "甲醛交联试剂",
                "ChIP DNA 纯化试剂盒",
                "文库构建试剂盒",
            ])
        elif "ATAC-seq" in assay_type:
            materials.extend([
                "Tn5 转座酶",
                "ATAC-seq 文库构建试剂盒",
                "MinElute PCR 纯化试剂盒",
            ])

        return materials

    def _suggest_qc(self, assay_type: str) -> list[str]:
        """推荐质量控制检查点。"""
        qc = [
            "RNA/DNA 完整性 (RIN/DIN ≥ 7.0)",
            "文库浓度 (Qubit)",
            "文库片段分布 (Bioanalyzer/TapeStation)",
        ]
        if assay_type in ("RNA-seq", "scRNA-seq"):
            qc.extend([
                "Raw reads: FastQC → ≥70% Q30",
                "比对率: ≥80% uniquely mapped",
                "基因检出数: ≥15000 (RNA-seq)",
                "生物学重复相关性: Pearson r ≥ 0.9",
            ])
        elif assay_type in ("ChIP-seq", "ATAC-seq"):
            qc.extend([
                "FRiP score: ≥1% (ChIP-seq)",
                "TSS enrichment: ≥6 (ATAC-seq)",
                "Library complexity (NRF ≥ 0.8)",
            ])
        return qc

    def _suggest_expected_results(self, assay_type: str, num_conditions: int) -> list[str]:
        """描述预期结果。"""
        results = []
        if "RNA-seq" in assay_type:
            results = [
                "获得各组基因表达矩阵（raw counts + normalized counts）",
                "PCA 分析显示组间分离情况",
            ]
            if num_conditions >= 2:
                results.extend([
                    "差异表达基因列表（log2FC + adjusted p-value）",
                    "火山图和 MA 图展示差异基因分布",
                    "GO/KEGG 富集分析揭示受影响的生物学通路",
                ])
        elif "ChIP-seq" in assay_type:
            results = [
                "Peak calling 结果（narrowPeak/broadPeak 文件）",
                "Peak 在基因组功能区域的分布（启动子/增强子/基因体）",
                "差异结合位点分析（如有多个条件）",
                "Motif 富集分析",
            ]
        return results

    def _suggest_risks(self, assay_type: str) -> list[str]:
        """列出可能的风险。"""
        risks = [
            "样品质量不合格（RIN < 7）→ 需要重新提取 RNA",
            "文库浓度不足 → 需要增加 PCR 循环数",
            "测序数据量不足 → 可能需要补测",
        ]
        if assay_type == "ChIP-seq":
            risks.append("抗体特异性差 → 需要更换抗体或优化 IP 条件")
        if assay_type == "scRNA-seq":
            risks.append("细胞活性 < 80% → 需要优化解离条件")
        return risks
