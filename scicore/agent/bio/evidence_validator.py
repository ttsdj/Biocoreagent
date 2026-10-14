"""EvidenceSpan 强制执行器 — 确保生物信息学输出可追溯到原始文献.

实现三个层面的强制：
  1. 工具输出层：所有 bio 工具的输出必须携带 evidence_spans
  2. 系统提示层：Agent 被要求标注每条生物学论断的证据来源
  3. 多 Agent 层：子 Agent 的证据链合并到父 Agent

核心原则：无证据不输出，缺失必声明，来源必追溯

Reference: SciCoreAgent 需求文档, Section 3 (安全护栏机制).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .schemas import EvidenceSpan


# ---------------------------------------------------------------------------
# Validation Result
# ---------------------------------------------------------------------------


@dataclass
class EvidenceViolation:
    """一条证据违规。"""

    field: str  # 缺少证据的字段名
    value: Any  # 字段值
    severity: str  # "error" (阻断) | "warning" (警告) | "info" (建议)
    reason: str  # 违规原因

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "value": str(self.value)[:200],
            "severity": self.severity,
            "reason": self.reason,
        }


@dataclass
class EvidenceReport:
    """证据完整性报告。"""

    passed: bool
    violations: list[EvidenceViolation] = field(default_factory=list)
    evidence_count: int = 0
    missing_count: int = 0
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": [v.to_dict() for v in self.violations],
            "evidence_count": self.evidence_count,
            "missing_count": self.missing_count,
            "summary": self.summary,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CRITICAL_FIELDS — 必须携带证据的关键字段
# ---------------------------------------------------------------------------

# 每个工具参数的证据要求级别
# "required" = 必须有 EvidenceSpan，否则阻断
# "recommended" = 建议有，缺失时警告
# "optional" = 无要求

CRITICAL_FIELDS: dict[str, dict[str, str]] = {
    # Protocol 相关
    "tools_required": {"level": "required", "category": "软件工具"},
    "databases_required": {"level": "required", "category": "数据库"},
    "software_required": {"level": "required", "category": "软件包"},
    "reference_genome": {"level": "required", "category": "参考基因组"},
    "genome_version": {"level": "required", "category": "基因组版本"},
    "annotation_version": {"level": "required", "category": "注释版本"},
    "parameters": {"level": "required", "category": "分析参数"},
    "organism": {"level": "required", "category": "物种"},
    "assay_type": {"level": "required", "category": "实验类型"},
    "version": {"level": "required", "category": "版本号"},
    # 文献相关
    "pmid": {"level": "recommended", "category": "PMID"},
    "method_name": {"level": "required", "category": "方法名称"},
    "method_description": {"level": "recommended", "category": "方法描述"},
    # 流程相关
    "workflow_tool": {"level": "required", "category": "工作流工具"},
    "workflow_step": {"level": "recommended", "category": "工作流步骤"},
}


# ---------------------------------------------------------------------------
# EvidenceValidator
# ---------------------------------------------------------------------------


class EvidenceValidator:
    """证据完整性验证器。

    验证规则：
    1. CRITICAL_FIELDS 中的 "required" 字段必须有对应的 EvidenceSpan
    2. EvidenceSpan 的 text 不能为空或只是占位符
    3. EvidenceSpan 的 source_id 必须指向真实来源
    4. 不能使用 "common knowledge", "well known" 等模糊来源
    """

    # 禁止的伪证据短语
    FORBIDDEN_PHRASES: list[str] = [
        "common knowledge",
        "well known",
        "widely used",
        "standard practice",
        "general consensus",
        "常识",
        "众所周知",
        "通用做法",
        "行业标准",
    ]

    def validate_field(
        self,
        field_name: str,
        field_value: Any,
        evidence_spans: list[EvidenceSpan] | None = None,
    ) -> EvidenceReport:
        """验证单个字段是否有充分的证据支持。

        Args:
            field_name: 字段名
            field_value: 字段值
            evidence_spans: 该字段关联的证据列表

        Returns:
            EvidenceReport: 验证报告
        """
        violations: list[EvidenceViolation] = []

        # 空值不需要证据
        if field_value is None or field_value == "" or field_value == []:
            return EvidenceReport(
                passed=True,
                violations=[],
                evidence_count=0,
                missing_count=0,
                summary=f"字段 '{field_name}' 为空，无需证据",
            )

        # 检查是否属于关键字段
        field_rule = CRITICAL_FIELDS.get(field_name, {"level": "optional", "category": "其他"})

        if field_rule["level"] == "optional":
            return EvidenceReport(passed=True, violations=[], evidence_count=0, missing_count=0,
                                 summary=f"字段 '{field_name}' 无需强制证据")

        # 检查证据是否存在
        if not evidence_spans:
            if field_rule["level"] == "required":
                violations.append(EvidenceViolation(
                    field=field_name,
                    value=field_value,
                    severity="error",
                    reason=f"关键字段 '{field_name}'（{field_rule['category']}）缺少 EvidenceSpan。"
                           f"值 '{field_value}' 必须追溯到原始文献的具体段落。",
                ))
            else:
                violations.append(EvidenceViolation(
                    field=field_name,
                    value=field_value,
                    severity="warning",
                    reason=f"建议字段 '{field_name}'（{field_rule['category']}）缺少 EvidenceSpan。",
                ))
        else:
            # 验证每个 EvidenceSpan 的质量
            for i, span in enumerate(evidence_spans):
                sub_violations = self._validate_span(span, i, field_name)
                violations.extend(sub_violations)

        passed = len([v for v in violations if v.severity == "error"]) == 0

        return EvidenceReport(
            passed=passed,
            violations=violations,
            evidence_count=len(evidence_spans or []),
            missing_count=0 if evidence_spans else 1,
            summary=f"字段 '{field_name}': "
                    f"{'✅ 通过' if passed else '❌ 未通过'} "
                    f"({len(violations)} 条违规)",
        )

    def validate_object(
        self,
        obj: dict,
        field_evidence_map: dict[str, list[EvidenceSpan]] | None = None,
    ) -> EvidenceReport:
        """验证整个对象的证据完整性。

        Args:
            obj: 待验证的对象字典
            field_evidence_map: 字段名 → EvidenceSpan 列表的映射

        Returns:
            EvidenceReport: 完整验证报告
        """
        all_violations: list[EvidenceViolation] = []
        field_evidence_map = field_evidence_map or {}

        for key, value in obj.items():
            spans = field_evidence_map.get(key, [])
            # 如果对象自带 evidence_spans，使用它
            if key == "evidence_spans":
                continue
            report = self.validate_field(key, value, spans)
            all_violations.extend(report.violations)

        errors = [v for v in all_violations if v.severity == "error"]
        warnings = [v for v in all_violations if v.severity == "warning"]

        return EvidenceReport(
            passed=len(errors) == 0,
            violations=all_violations,
            evidence_count=sum(
                1 for spans in field_evidence_map.values() for _ in spans
            ),
            missing_count=len(errors),
            summary=(
                f"对象验证: {len(all_violations)} 条违规 "
                f"({len(errors)} 错误, {len(warnings)} 警告)"
            ),
        )

    def validate_tool_output(
        self,
        tool_name: str,
        output: dict,
    ) -> EvidenceReport:
        """验证工具输出的证据完整性。

        根据工具名称确定 CRITICAL_FIELDS 子集。
        """
        violations: list[EvidenceViolation] = []

        # 根据工具类型确定关键字段
        if tool_name in ("bio_extract_protocol",):
            required_fields = ["tools_required", "databases_required",
                              "software_required", "organism", "assay_type"]
        elif tool_name in ("bio_replication_plan",):
            required_fields = ["tasks", "resources", "missing_information"]
        elif tool_name in ("bio_workflow_sketch",):
            required_fields = ["workflow_tool", "parameters"]
        elif tool_name in ("pubmed_extract_rnaseq_methods",):
            required_fields = ["method_name", "tools_required"]
        else:
            required_fields = []

        for field_name in required_fields:
            if field_name in output:
                evidence_spans = output.get("evidence_spans", [])
                field_report = self.validate_field(
                    field_name, output[field_name], evidence_spans
                )
                violations.extend(field_report.violations)

        errors = [v for v in violations if v.severity == "error"]

        return EvidenceReport(
            passed=len(errors) == 0,
            violations=violations,
            evidence_count=len(output.get("evidence_spans", [])),
            missing_count=len(errors),
            summary=(
                f"工具 '{tool_name}' 输出验证: "
                f"{'✅ 通过' if len(errors) == 0 else '❌ ' + str(len(errors)) + ' 条缺失证据'}"
            ),
        )

    def _validate_span(
        self,
        span: EvidenceSpan,
        index: int,
        field_name: str,
    ) -> list[EvidenceViolation]:
        """验证单个 EvidenceSpan 的质量。"""
        violations: list[EvidenceViolation] = []

        # 检查 text 是否为空
        if not span.text or span.text.strip() == "":
            violations.append(EvidenceViolation(
                field=f"{field_name}.evidence[{index}].text",
                value=span.text,
                severity="error",
                reason="EvidenceSpan.text 不能为空",
            ))

        # 检查 text 是否为禁止的伪证据短语
        for phrase in self.FORBIDDEN_PHRASES:
            if phrase.lower() in span.text.lower():
                violations.append(EvidenceViolation(
                    field=f"{field_name}.evidence[{index}].text",
                    value=span.text[:100],
                    severity="error",
                    reason=f"EvidenceSpan.text 包含禁止的伪证据短语 '{phrase}'。"
                           f"必须引用原文具体段落，不能使用模糊表述。",
                ))
                break

        # 检查 source_id 是否有效
        if not span.source_id or span.source_id.strip() == "":
            violations.append(EvidenceViolation(
                field=f"{field_name}.evidence[{index}].source_id",
                value=span.source_id,
                severity="error",
                reason="EvidenceSpan.source_id 不能为空，必须指向真实来源。",
            ))

        # 检查 source_type 是否有效
        valid_types = {"protocol", "paper", "readme", "workflow_doc", "other"}
        if span.source_type not in valid_types:
            violations.append(EvidenceViolation(
                field=f"{field_name}.evidence[{index}].source_type",
                value=span.source_type,
                severity="warning",
                reason=f"EvidenceSpan.source_type '{span.source_type}' 不在有效类型列表中: {valid_types}",
            ))

        # 检查 location 是否提供
        if not span.location or span.location.strip() == "":
            violations.append(EvidenceViolation(
                field=f"{field_name}.evidence[{index}].location",
                value=span.location,
                severity="warning",
                reason="EvidenceSpan.location 为空。建议提供具体位置（如 'chunk_003 lines 12-18'）",
            ))

        return violations


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

_GLOBAL_VALIDATOR: EvidenceValidator | None = None


def get_evidence_validator() -> EvidenceValidator:
    """获取全局证据验证器单例。"""
    global _GLOBAL_VALIDATOR
    if _GLOBAL_VALIDATOR is None:
        _GLOBAL_VALIDATOR = EvidenceValidator()
    return _GLOBAL_VALIDATOR


# ---------------------------------------------------------------------------
# System Prompt 注入
# ---------------------------------------------------------------------------


def evidence_guard_prompt() -> str:
    """生成证据护栏的系统提示文本，注入到 Agent system prompt 中。"""

    field_rules = "\n".join(
        f"  - **{field}** ({rule['category']}): {rule['level']}"
        for field, rule in sorted(CRITICAL_FIELDS.items())
        if rule["level"] in ("required", "recommended")
    )

    forbidden = "\n".join(
        f"  - ❌ \"{phrase}\""
        for phrase in EvidenceValidator.FORBIDDEN_PHRASES[:6]
    )

    return f"""\
# EvidenceSpan 强制追溯规则 (Evidence Guard)

核心原则：**无证据不输出，缺失必声明，来源必追溯**

## 关键字段证据要求
以下字段在输出时必须携带 EvidenceSpan 追溯到原始文献的具体段落：

{field_rules}

## 禁止的伪证据表述
以下短语不得作为证据来源（必须引用原文具体段落）：
{forbidden}

## 证据编写规范
1. 每个 EvidenceSpan 必须包含：source_id（来源标识）、source_type（来源类型）、location（原文位置）、text（原文摘录）
2. 原文未明确说明的字段，必须留空/None，不得猜测填充
3. 每个 TaskCard 和 ResourceCard 必须至少有一个 EvidenceSpan 指向其来源文档
4. 实验方法、软件工具、数据库版本、参考基因组等关键信息必须有证据支撑

## 证据不足时的行为
1. 将缺失字段列入 `missing_information` 列表
2. 如果缺失导致无法继续，填入 `cannot_proceed_reasons`
3. 不得在证据不足时生成可执行的工作流代码
4. 如实报告"信息缺失"而非填补虚构数据

## 多 Agent 证据链
1. 子 Agent 的输出必须包含 evidence_spans
2. 父 Agent 合并结果时保留所有子 Agent 的证据
3. 冲突证据：同时记录，标注冲突，由用户裁决
"""
