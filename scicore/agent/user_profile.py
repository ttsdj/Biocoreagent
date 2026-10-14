"""用户单位信息与个性化配置 — 基于机构的实验条件推荐.

允许用户提供单位信息（机构、实验室、可用仪器、常用物种/参考基因组等），
系统据此个性化推荐实验条件、日程安排和工作流参数。

核心功能：
  1. 用户 Profile 管理（CRUD）
  2. 基于仪器的实验条件推荐
  3. 基于物种的参考基因组自动匹配
  4. 实验日程推算（考虑机构工作制）
  5. 个性化系统提示生成

Reference: SciCoreAgent 需求文档, Section 8 (用户单位信息).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class InstitutionInfo:
    """机构信息。"""

    name: str = ""  # 机构名称，如 "清华大学生命科学学院"
    department: str = ""  # 部门，如 "生物信息学与系统生物学实验室"
    lab: str = ""  # 实验室名称
    location: str = ""  # 所在地，如 "北京"
    timezone: str = "Asia/Shanghai"  # 时区


@dataclass
class InstrumentInfo:
    """仪器信息。"""

    name: str  # 仪器名称，如 "Illumina NovaSeq 6000"
    instrument_type: str  # 类型：sequencer, microscope, qPCR, centrifuge 等
    model: str = ""  # 型号
    specs: str = ""  # 规格说明（如 "PE150, 800M reads/flowcell"）
    status: str = "available"  # available, maintenance, shared


@dataclass
class OrganismInfo:
    """常用研究物种/细胞系。"""

    name: str  # 物种名称，如 "Homo sapiens"
    strain: str = ""  # 品系/细胞系，如 "HEK293T", "C57BL/6"
    reference_genome: str = ""  # 参考基因组，如 "GRCh38/hg38"
    annotation: str = ""  # 注释版本，如 "GENCODE v44"
    source: str = ""  # 来源，如 "Ensembl", "UCSC", "NCBI"


@dataclass
class WorkSchedule:
    """实验工作制。"""

    working_days: list[str] = field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri"])
    working_hours: str = "09:00-18:00"  # 工作时间段
    core_facility_hours: str = ""  # 公共平台开放时间
    holiday_calendar: str = ""  # 假期日历（中国法定假日 / 院校假期）


@dataclass
class UserProfile:
    """SciCoreAgent 用户完整档案。

    存储用户的机构信息、可用仪器、常用物种、实验偏好等，
    用于个性化的实验方案推荐和日程推算。
    """

    profile_id: str = ""
    created_at: str = ""
    updated_at: str = ""

    # 机构
    institution: InstitutionInfo = field(default_factory=InstitutionInfo)

    # 仪器
    instruments: list[InstrumentInfo] = field(default_factory=list)

    # 常用物种
    organisms: list[OrganismInfo] = field(default_factory=list)

    # 工作制
    schedule: WorkSchedule = field(default_factory=WorkSchedule)

    # 偏好
    preferred_sequencing_platform: str = ""  # 首选测序平台
    preferred_analysis_tools: list[str] = field(default_factory=list)  # 首选分析工具
    preferred_language: str = "zh"  # 界面语言

    # 实验偏好
    default_replicates: int = 3  # 默认生物学重复数
    default_significance_threshold: float = 0.05  # 默认显著性阈值
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "institution": {
                "name": self.institution.name,
                "department": self.institution.department,
                "lab": self.institution.lab,
                "location": self.institution.location,
                "timezone": self.institution.timezone,
            },
            "instruments": [
                {
                    "name": i.name,
                    "type": i.instrument_type,
                    "model": i.model,
                    "specs": i.specs,
                    "status": i.status,
                }
                for i in self.instruments
            ],
            "organisms": [
                {
                    "name": o.name,
                    "strain": o.strain,
                    "reference_genome": o.reference_genome,
                    "annotation": o.annotation,
                    "source": o.source,
                }
                for o in self.organisms
            ],
            "schedule": {
                "working_days": self.schedule.working_days,
                "working_hours": self.schedule.working_hours,
                "core_facility_hours": self.schedule.core_facility_hours,
                "holiday_calendar": self.schedule.holiday_calendar,
            },
            "preferences": {
                "sequencing_platform": self.preferred_sequencing_platform,
                "analysis_tools": self.preferred_analysis_tools,
                "language": self.preferred_language,
                "default_replicates": self.default_replicates,
                "default_significance_threshold": self.default_significance_threshold,
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> UserProfile:
        inst = data.get("institution", {})
        prefs = data.get("preferences", {})
        sched = data.get("schedule", {})

        return cls(
            profile_id=data.get("profile_id", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            institution=InstitutionInfo(
                name=inst.get("name", ""),
                department=inst.get("department", ""),
                lab=inst.get("lab", ""),
                location=inst.get("location", ""),
                timezone=inst.get("timezone", "Asia/Shanghai"),
            ),
            instruments=[
                InstrumentInfo(
                    name=i["name"],
                    instrument_type=i.get("type", ""),
                    model=i.get("model", ""),
                    specs=i.get("specs", ""),
                    status=i.get("status", "available"),
                )
                for i in data.get("instruments", [])
            ],
            organisms=[
                OrganismInfo(
                    name=o["name"],
                    strain=o.get("strain", ""),
                    reference_genome=o.get("reference_genome", ""),
                    annotation=o.get("annotation", ""),
                    source=o.get("source", ""),
                )
                for o in data.get("organisms", [])
            ],
            schedule=WorkSchedule(
                working_days=sched.get("working_days", ["Mon", "Tue", "Wed", "Thu", "Fri"]),
                working_hours=sched.get("working_hours", "09:00-18:00"),
                core_facility_hours=sched.get("core_facility_hours", ""),
                holiday_calendar=sched.get("holiday_calendar", ""),
            ),
            preferred_sequencing_platform=prefs.get("sequencing_platform", ""),
            preferred_analysis_tools=prefs.get("analysis_tools", []),
            preferred_language=prefs.get("language", "zh"),
            default_replicates=prefs.get("default_replicates", 3),
            default_significance_threshold=prefs.get("default_significance_threshold", 0.05),
            notes=data.get("notes", ""),
        )


# ---------------------------------------------------------------------------
# Profile Manager
# ---------------------------------------------------------------------------

DEFAULT_PROFILE_PATH = Path(".scicore") / "user_profile.json"


class UserProfileManager:
    """用户档案管理器。

    Usage::

        mgr = UserProfileManager()
        mgr.set_institution("清华大学", "生物信息学实验室")
        mgr.add_instrument("Illumina NovaSeq 6000", "sequencer")
        mgr.add_organism("Homo sapiens", reference_genome="GRCh38/hg38")
        recommendations = mgr.recommend_experiment_conditions()
    """

    def __init__(self, profile_path: str | Path | None = None):
        self._path = Path(profile_path or DEFAULT_PROFILE_PATH)
        self._profile: UserProfile | None = None

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    @property
    def profile(self) -> UserProfile:
        if self._profile is None:
            self._profile = self._load()
        return self._profile

    def is_configured(self) -> bool:
        """用户是否已配置档案。"""
        p = self.profile
        return bool(p.institution.name) or bool(p.instruments) or bool(p.organisms)

    def _load(self) -> UserProfile:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return UserProfile.from_dict(data)
            except Exception:
                pass
        return UserProfile()

    def save(self) -> None:
        """持久化用户档案。"""
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        profile = self.profile
        if not profile.profile_id:
            import uuid
            profile.profile_id = uuid.uuid4().hex[:12]
            profile.created_at = now
        profile.updated_at = now
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_institution(
        self,
        name: str = "",
        department: str = "",
        lab: str = "",
        location: str = "",
    ) -> None:
        p = self.profile
        if name:
            p.institution.name = name
        if department:
            p.institution.department = department
        if lab:
            p.institution.lab = lab
        if location:
            p.institution.location = location

    def add_instrument(
        self,
        name: str,
        instrument_type: str = "",
        model: str = "",
        specs: str = "",
    ) -> None:
        p = self.profile
        # 防止重复
        existing = [i for i in p.instruments if i.name.lower() == name.lower()]
        if existing:
            return
        p.instruments.append(InstrumentInfo(
            name=name,
            instrument_type=instrument_type,
            model=model,
            specs=specs,
        ))

    def remove_instrument(self, name: str) -> bool:
        p = self.profile
        before = len(p.instruments)
        p.instruments = [i for i in p.instruments if i.name.lower() != name.lower()]
        return len(p.instruments) < before

    def add_organism(
        self,
        name: str,
        strain: str = "",
        reference_genome: str = "",
        annotation: str = "",
        source: str = "",
    ) -> None:
        p = self.profile
        existing = [o for o in p.organisms if o.name.lower() == name.lower()]
        if existing:
            return
        p.organisms.append(OrganismInfo(
            name=name,
            strain=strain,
            reference_genome=reference_genome,
            annotation=annotation,
            source=source,
        ))

    def remove_organism(self, name: str) -> bool:
        p = self.profile
        before = len(p.organisms)
        p.organisms = [o for o in p.organisms if o.name.lower() != name.lower()]
        return len(p.organisms) < before

    def set_schedule(self, working_days: list[str] | None = None, working_hours: str = "") -> None:
        p = self.profile
        if working_days:
            p.schedule.working_days = working_days
        if working_hours:
            p.schedule.working_hours = working_hours

    # ------------------------------------------------------------------
    # Recommendations — 基于用户档案的个性化推荐
    # ------------------------------------------------------------------

    def recommend_experiment_conditions(self) -> dict:
        """基于用户单位信息推荐实验条件。

        返回适合该实验室的实验方案建议。
        """
        p = self.profile
        recommendations: dict[str, Any] = {
            "profile_configured": self.is_configured(),
            "sequencing": [],
            "reference_genomes": [],
            "tools": [],
            "timeline": {},
            "warnings": [],
        }

        # 1. 基于可用仪器推荐测序策略
        sequencers = [i for i in p.instruments if i.instrument_type in ("sequencer", "sequencing")]
        if sequencers:
            for seq in sequencers:
                if "NovaSeq" in seq.name or "NovaSeq" in seq.model:
                    recommendations["sequencing"].append({
                        "instrument": seq.name,
                        "recommended_read_length": "PE150",
                        "recommended_depth": {
                            "RNA-seq (mRNA)": "20-30M reads/sample",
                            "RNA-seq (lncRNA)": "50-100M reads/sample",
                            "WGS (human 30x)": "~300M reads/sample",
                            "WES": "~50M reads/sample",
                            "ChIP-seq": "20-40M reads/sample",
                            "ATAC-seq": "25-50M reads/sample",
                        },
                        "note": "NovaSeq 适合高通量项目，注意 index 多样性",
                    })
                elif "HiSeq" in seq.name or "HiSeq" in seq.model:
                    recommendations["sequencing"].append({
                        "instrument": seq.name,
                        "recommended_read_length": "PE150",
                        "recommended_depth": {
                            "RNA-seq (mRNA)": "20-30M reads/sample",
                            "ChIP-seq": "20-40M reads/sample",
                        },
                    })
                elif "MiSeq" in seq.name or "MiSeq" in seq.model:
                    recommendations["sequencing"].append({
                        "instrument": seq.name,
                        "recommended_read_length": "PE300",
                        "recommended_depth": {
                            "16S rRNA": "50K-100K reads/sample",
                            "扩增子测序": "50K-100K reads/sample",
                        },
                        "note": "MiSeq 适合小型项目和扩增子测序",
                    })
        else:
            recommendations["warnings"].append("未配置测序仪信息，无法推荐测序策略。使用 user_profile_set 配置。")

        # 2. 基于常用物种推荐参考基因组
        for org in p.organisms:
            if org.reference_genome:
                recommendations["reference_genomes"].append({
                    "organism": org.name,
                    "strain": org.strain,
                    "reference_genome": org.reference_genome,
                    "annotation": org.annotation,
                    "source": org.source,
                })
            else:
                # 自动匹配已知参考基因组
                matched = _auto_match_reference(org.name)
                if matched:
                    recommendations["reference_genomes"].append({
                        "organism": org.name,
                        "strain": org.strain,
                        "reference_genome": matched["genome"],
                        "annotation": matched.get("annotation", ""),
                        "source": matched.get("source", ""),
                        "auto_matched": True,
                        "note": "自动匹配，请确认版本",
                    })

        # 3. 基于机构所在地推荐实验日程
        if p.institution.location:
            recommendations["timeline"] = {
                "location": p.institution.location,
                "working_days": p.schedule.working_days,
                "working_hours": p.schedule.working_hours,
                "estimated_setup_days": 1,
                "estimated_sequencing_turnaround": "7-14 个工作日（视测序平台排队情况）",
                "note": (
                    f"基于 {p.institution.location} 的工作制推算。"
                    "公共平台送样需提前预约，建议预留 1-2 周排队时间。"
                ),
            }

        # 4. 基于首选工具推荐分析方法
        if p.preferred_analysis_tools:
            recommendations["tools"] = p.preferred_analysis_tools

        return recommendations

    def recommend_for_species(self, organism_name: str) -> dict | None:
        """根据用户档案中的物种信息，推荐该物种的分析参数。"""
        p = self.profile
        for org in p.organisms:
            if org.name.lower() == organism_name.lower():
                result = {
                    "organism": org.name,
                    "strain": org.strain,
                    "reference_genome": org.reference_genome,
                    "annotation": org.annotation,
                    "source": org.source,
                }
                if not org.reference_genome:
                    matched = _auto_match_reference(org.name)
                    if matched:
                        result.update(matched)
                        result["auto_matched"] = True
                return result

        # 即使不在档案中，也尝试自动匹配
        matched = _auto_match_reference(organism_name)
        if matched:
            matched["auto_matched"] = True
            return matched
        return None

    def recommend_for_assay(self, assay_type: str) -> dict:
        """根据实验类型推荐测序深度和参数。"""
        p = self.profile
        sequencers = [i for i in p.instruments if i.instrument_type == "sequencer"]

        # 默认推荐（无仪器信息时）
        default_recs = {
            "RNA-seq (mRNA)": {
                "reads": "20-30M PE150",
                "replicates": p.default_replicates,
                "note": "推荐生物学重复 ≥3，人类/小鼠推荐 30M，植物推荐 20M",
            },
            "RNA-seq (lncRNA)": {
                "reads": "50-100M PE150",
                "replicates": p.default_replicates,
                "note": "lncRNA 表达量低，需提高测序深度",
            },
            "scRNA-seq": {
                "reads": "30K-50K reads/cell",
                "cells": "3000-10000 cells",
                "note": "10x Genomics 推荐捕获 3000-10000 个细胞",
            },
            "ChIP-seq": {
                "reads": "20-40M SE50",
                "replicates": max(2, p.default_replicates - 1),
                "note": "ChIP-seq 推荐 ≥2 个生物学重复，需 Input 对照",
            },
            "ATAC-seq": {
                "reads": "25-50M PE50",
                "replicates": max(2, p.default_replicates - 1),
                "note": "ATAC-seq 推荐 ≥2 个生物学重复",
            },
            "WGS": {
                "reads": "300M PE150 (human 30x)",
                "replicates": 1,
                "note": "全基因组测序深度 ≥30x（人类），需考虑建库 PCR 重复",
            },
            "WES": {
                "reads": "50M PE150 (human 100x)",
                "replicates": 1,
                "note": "全外显子测序目标覆盖度 100x",
            },
        }

        result = default_recs.get(assay_type, {
            "reads": "视具体实验设计而定",
            "replicates": p.default_replicates,
            "note": "请提供更具体的实验类型",
        })

        # 如果有特定测序仪，调整推荐
        if sequencers:
            seq_names = [s.name for s in sequencers]
            result["available_instruments"] = seq_names

        return result

    # ------------------------------------------------------------------
    # System Prompt 注入
    # ------------------------------------------------------------------

    def profile_prompt_context(self) -> str:
        """生成用户档案的系统提示文本。"""
        p = self.profile

        if not self.is_configured():
            return (
                "## 👤 用户档案\n\n"
                "用户档案尚未配置。使用 `user_profile_set` 工具配置单位信息，"
                "以获得个性化的实验条件推荐和日程推算。\n\n"
                "可配置信息：\n"
                "- 机构名称和实验室\n"
                "- 可用仪器（测序仪、显微镜等）\n"
                "- 常用研究物种和参考基因组\n"
                "- 实验工作制\n"
            )

        lines = ["## 👤 用户档案", ""]

        # 机构
        inst = p.institution
        if inst.name:
            parts = [inst.name]
            if inst.department:
                parts.append(inst.department)
            if inst.lab:
                parts.append(inst.lab)
            lines.append(f"**机构**: {' / '.join(parts)}")
            if inst.location:
                lines.append(f"**所在地**: {inst.location}")
            lines.append("")

        # 仪器
        if p.instruments:
            lines.append("**可用仪器**:")
            for instr in p.instruments:
                status_mark = "✅" if instr.status == "available" else "⚠️"
                spec_text = f" ({instr.specs})" if instr.specs else ""
                lines.append(f"  - {status_mark} {instr.name} ({instr.instrument_type}){spec_text}")
            lines.append("")

        # 常用物种
        if p.organisms:
            lines.append("**常用物种/参考基因组**:")
            for org in p.organisms:
                ref_text = f" → {org.reference_genome}" if org.reference_genome else " → ⚠️ 未配置参考基因组"
                strain_text = f" ({org.strain})" if org.strain else ""
                lines.append(f"  - {org.name}{strain_text}{ref_text}")
            lines.append("")

        # 偏好
        if p.preferred_sequencing_platform or p.preferred_analysis_tools:
            lines.append("**分析偏好**:")
            if p.preferred_sequencing_platform:
                lines.append(f"  - 首选测序平台: {p.preferred_sequencing_platform}")
            if p.preferred_analysis_tools:
                lines.append(f"  - 首选分析工具: {', '.join(p.preferred_analysis_tools)}")
            lines.append(f"  - 默认生物学重复数: {p.default_replicates}")
            lines.append(f"  - 默认显著性阈值: {p.default_significance_threshold}")
            lines.append("")

        # 日程
        sched = p.schedule
        lines.append("**工作制**:")
        lines.append(f"  - 工作日: {', '.join(sched.working_days)}")
        lines.append(f"  - 工作时间: {sched.working_hours}")
        if sched.core_facility_hours:
            lines.append(f"  - 公共平台开放时间: {sched.core_facility_hours}")
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_GLOBAL_PROFILE_MGR: UserProfileManager | None = None


def get_profile_manager() -> UserProfileManager:
    """获取全局用户档案管理器单例。"""
    global _GLOBAL_PROFILE_MGR
    if _GLOBAL_PROFILE_MGR is None:
        _GLOBAL_PROFILE_MGR = UserProfileManager()
    return _GLOBAL_PROFILE_MGR


# ---------------------------------------------------------------------------
# 参考基因组自动匹配表
# ---------------------------------------------------------------------------

_REFERENCE_GENOME_MAP: dict[str, dict] = {
    "homo sapiens": {
        "genome": "GRCh38/hg38",
        "annotation": "GENCODE v44",
        "source": "Ensembl/UCSC",
    },
    "mus musculus": {
        "genome": "GRCm39/mm39",
        "annotation": "GENCODE vM33",
        "source": "Ensembl/UCSC",
    },
    "rattus norvegicus": {
        "genome": "mRatBN7.2/rn7",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
    "danio rerio": {
        "genome": "GRCz11/danRer11",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
    "drosophila melanogaster": {
        "genome": "BDGP6.46/dm6",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
    "caenorhabditis elegans": {
        "genome": "WBcel235/ce11",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
    "arabidopsis thaliana": {
        "genome": "TAIR10",
        "annotation": "Araport11",
        "source": "TAIR",
    },
    "oryza sativa": {
        "genome": "IRGSP-1.0",
        "annotation": "MSU v7.0",
        "source": "MSU/RAP-DB",
    },
    "saccharomyces cerevisiae": {
        "genome": "R64-1-1/sacCer3",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
    "escherichia coli": {
        "genome": "ASM80076v1",
        "annotation": "NCBI RefSeq",
        "source": "NCBI",
    },
    "macaca mulatta": {
        "genome": "Mmul_10/rheMac10",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
    "xenopus laevis": {
        "genome": "Xenopus_laevis_v10.1/xenLae2",
        "annotation": "Ensembl 109",
        "source": "Ensembl",
    },
}


def _auto_match_reference(organism_name: str) -> dict | None:
    """根据物种名称自动匹配参考基因组。"""
    name_lower = organism_name.lower().strip()
    # 精确匹配
    if name_lower in _REFERENCE_GENOME_MAP:
        return dict(_REFERENCE_GENOME_MAP[name_lower])
    # 模糊匹配
    for key, value in _REFERENCE_GENOME_MAP.items():
        if key in name_lower or name_lower in key:
            return dict(value)
    # 中文名称匹配
    cn_map = {
        "人": "homo sapiens",
        "小鼠": "mus musculus",
        "大鼠": "rattus norvegicus",
        "斑马鱼": "danio rerio",
        "果蝇": "drosophila melanogaster",
        "线虫": "caenorhabditis elegans",
        "拟南芥": "arabidopsis thaliana",
        "水稻": "oryza sativa",
        "酵母": "saccharomyces cerevisiae",
        "大肠杆菌": "escherichia coli",
    }
    for cn, en in cn_map.items():
        if cn in organism_name:
            return dict(_REFERENCE_GENOME_MAP.get(en, {}))
    return None
