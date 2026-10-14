"""用户档案工具 — 将 UserProfileManager 暴露为 Agent 可调用的工具。"""

from __future__ import annotations

import json

from .base import Tool
from ..user_profile import get_profile_manager


class UserProfileSetTool(Tool):
    """设置用户档案信息的工具。"""

    name = "user_profile_set"
    description = (
        "设置用户单位信息和个人偏好。配置后可获得个性化的实验条件推荐、"
        "参考基因组匹配和日程推算。可设置机构、仪器、研究物种、工作制等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "institution_name": {
                "type": "string",
                "description": "机构名称，如 '清华大学生命科学学院'",
            },
            "department": {
                "type": "string",
                "description": "部门/系，如 '生物信息学与系统生物学实验室'",
            },
            "lab": {
                "type": "string",
                "description": "实验室名称",
            },
            "location": {
                "type": "string",
                "description": "所在地，如 '北京'、'上海'",
            },
            "instruments_json": {
                "type": "string",
                "description": (
                    "可用仪器列表的 JSON 字符串。每项包含: name (仪器名称), type (类型: sequencer/microscope/qPCR等), "
                    "model (型号), specs (规格说明)。例如: [{\"name\": \"Illumina NovaSeq 6000\", \"type\": \"sequencer\"}]"
                ),
            },
            "organisms_json": {
                "type": "string",
                "description": (
                    "常用研究物种列表的 JSON 字符串。每项包含: name (物种名), strain (品系/细胞系), "
                    "reference_genome (参考基因组), annotation (注释版本)。不填参考基因组则自动匹配。"
                ),
            },
            "preferred_tools_json": {
                "type": "string",
                "description": "首选分析工具的 JSON 数组，如 '[\"STAR\", \"DESeq2\", \"IGV\"]'",
            },
            "preferred_sequencing_platform": {
                "type": "string",
                "description": "首选测序平台，如 'Illumina NovaSeq 6000'",
            },
            "default_replicates": {
                "type": "integer",
                "description": "默认生物学重复数",
            },
            "working_days_json": {
                "type": "string",
                "description": "工作日的 JSON 数组，如 '[\"Mon\", \"Tue\", \"Wed\", \"Thu\", \"Fri\"]'",
            },
            "working_hours": {
                "type": "string",
                "description": "工作时间段，如 '09:00-18:00'",
            },
        },
    }

    def execute(
        self,
        institution_name: str = "",
        department: str = "",
        lab: str = "",
        location: str = "",
        instruments_json: str = "",
        organisms_json: str = "",
        preferred_tools_json: str = "",
        preferred_sequencing_platform: str = "",
        default_replicates: int = 0,
        working_days_json: str = "",
        working_hours: str = "",
    ) -> str:
        mgr = get_profile_manager()

        if institution_name or department or lab or location:
            mgr.set_institution(
                name=institution_name,
                department=department,
                lab=lab,
                location=location,
            )

        if instruments_json:
            try:
                instruments = json.loads(instruments_json)
                for inst in instruments:
                    mgr.add_instrument(
                        name=inst.get("name", ""),
                        instrument_type=inst.get("type", ""),
                        model=inst.get("model", ""),
                        specs=inst.get("specs", ""),
                    )
            except json.JSONDecodeError as e:
                return json.dumps({
                    "success": False,
                    "error": f"instruments_json 不是有效的 JSON: {e}",
                }, ensure_ascii=False, indent=2)

        if organisms_json:
            try:
                organisms = json.loads(organisms_json)
                for org in organisms:
                    mgr.add_organism(
                        name=org.get("name", ""),
                        strain=org.get("strain", ""),
                        reference_genome=org.get("reference_genome", ""),
                        annotation=org.get("annotation", ""),
                        source=org.get("source", ""),
                    )
            except json.JSONDecodeError as e:
                return json.dumps({
                    "success": False,
                    "error": f"organisms_json 不是有效的 JSON: {e}",
                }, ensure_ascii=False, indent=2)

        if preferred_tools_json:
            try:
                mgr.profile.preferred_analysis_tools = json.loads(preferred_tools_json)
            except json.JSONDecodeError:
                pass

        if preferred_sequencing_platform:
            mgr.profile.preferred_sequencing_platform = preferred_sequencing_platform

        if default_replicates > 0:
            mgr.profile.default_replicates = default_replicates

        if working_days_json:
            try:
                mgr.set_schedule(working_days=json.loads(working_days_json))
            except json.JSONDecodeError:
                pass

        if working_hours:
            mgr.set_schedule(working_hours=working_hours)

        mgr.save()

        return json.dumps({
            "success": True,
            "message": "用户档案已更新",
            "profile_summary": {
                "institution": mgr.profile.institution.name or "未设置",
                "instruments_count": len(mgr.profile.instruments),
                "organisms_count": len(mgr.profile.organisms),
                "preferred_platform": mgr.profile.preferred_sequencing_platform or "未设置",
                "default_replicates": mgr.profile.default_replicates,
            },
        }, ensure_ascii=False, indent=2)


class UserProfileGetTool(Tool):
    """查看用户档案的工具。"""

    name = "user_profile_get"
    description = "查看当前用户档案的完整信息，包括机构、仪器、物种、偏好等。"
    parameters = {
        "type": "object",
        "properties": {},
    }

    def execute(self) -> str:
        mgr = get_profile_manager()
        if not mgr.is_configured():
            return json.dumps({
                "configured": False,
                "message": "用户档案尚未配置。使用 user_profile_set 工具配置。",
            }, ensure_ascii=False, indent=2)

        return json.dumps({
            "configured": True,
            "profile": mgr.profile.to_dict(),
        }, ensure_ascii=False, indent=2)


class UserProfileRecommendTool(Tool):
    """基于用户档案推荐实验条件的工具。"""

    name = "user_profile_recommend"
    description = (
        "基于用户档案（机构、仪器、物种）推荐个性化实验条件。"
        "包括测序策略、参考基因组选择、实验日程等。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "assay_type": {
                "type": "string",
                "description": "实验类型，如 'RNA-seq', 'ChIP-seq', 'WGS', 'WES', 'ATAC-seq', 'scRNA-seq'（可选）",
            },
            "organism_name": {
                "type": "string",
                "description": "物种名称，如 'Homo sapiens', 'Mus musculus'（可选）",
            },
        },
    }

    def execute(self, assay_type: str = "", organism_name: str = "") -> str:
        mgr = get_profile_manager()

        if not mgr.is_configured():
            return json.dumps({
                "success": False,
                "message": "用户档案尚未配置。使用 user_profile_set 配置后可使用此功能。",
            }, ensure_ascii=False, indent=2)

        result: dict = {}

        if assay_type:
            result["assay_recommendation"] = mgr.recommend_for_assay(assay_type)

        if organism_name:
            species_rec = mgr.recommend_for_species(organism_name)
            if species_rec:
                result["species_recommendation"] = species_rec
            else:
                result["species_recommendation"] = {
                    "organism": organism_name,
                    "warning": "未找到该物种的参考基因组匹配。请手动配置或使用 user_profile_set 添加。",
                }

        result["general_recommendations"] = mgr.recommend_experiment_conditions()

        return json.dumps(result, ensure_ascii=False, indent=2)
