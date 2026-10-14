"""Agent 间通信与 Team 结果合成 — Multi-Agent 协作增强.

在现有 multiagent.py 基础上增加：
  1. Agent 间消息传递（共享消息队列）
  2. 共享工作区（Agent 共享中间文件）
  3. 自动结果合成（团队完成后合并输出）
  4. 冲突检测（两个 Agent 给出矛盾结论时）
  5. 综合报告生成（将团队输出合并为结构化报告）

Reference: SciCoreAgent 需求文档, Section 9 (多 Agent 协作体系).
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Agent Message
# ---------------------------------------------------------------------------


@dataclass
class AgentMessage:
    """Agent 间消息。"""

    message_id: str
    from_agent: str  # 发送方 job_id
    to_agent: str  # 接收方 job_id（"*" 表示广播）
    message_type: str  # "finding", "question", "blocker", "handoff", "result"
    content: str
    timestamp: str = ""
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "message_id": self.message_id,
            "from": self.from_agent,
            "to": self.to_agent,
            "type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Shared Workspace
# ---------------------------------------------------------------------------


class SharedWorkspace:
    """共享工作区 — 多个 Agent 共享的中间文件存储。

    每个 Team 拥有独立的工作区目录，Agent 可读写共享文件，
    通过文件锁保证并发安全。
    """

    DEFAULT_ROOT = Path(".scicore") / "workspaces"

    def __init__(self, team_id: str, root: Path | None = None):
        self._team_id = team_id
        self._root = (root or self.DEFAULT_ROOT) / team_id
        self._lock = Lock()

    @property
    def path(self) -> Path:
        """工作区根路径。"""
        return self._root

    def init(self) -> None:
        """初始化工作区目录。"""
        self._root.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, content: str) -> Path:
        """写入共享文件。"""
        with self._lock:
            filepath = self._safe_path(filename)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")
            return filepath

    def read(self, filename: str) -> str | None:
        """读取共享文件。"""
        filepath = self._safe_path(filename)
        if filepath.exists():
            return filepath.read_text(encoding="utf-8")
        return None

    def list_files(self) -> list[str]:
        """列出工作区所有文件。"""
        if not self._root.exists():
            return []
        return [
            str(p.relative_to(self._root))
            for p in self._root.rglob("*")
            if p.is_file()
        ]

    def clean(self) -> None:
        """清理工作区（删除所有文件）。"""
        import shutil
        if self._root.exists():
            shutil.rmtree(self._root)

    def _safe_path(self, filename: str) -> Path:
        """防止路径穿越攻击。"""
        resolved = (self._root / filename).resolve()
        if not str(resolved).startswith(str(self._root.resolve())):
            raise ValueError(f"非法的文件路径: {filename}")
        return resolved


# ---------------------------------------------------------------------------
# Team Synthesizer — 团队结果合成
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    """结果合成结果。"""

    team_id: str
    objective: str
    status: str  # "complete", "partial", "conflict"
    summary: str  # 综合摘要
    findings: list[dict]  # 各 Agent 的关键发现
    conflicts: list[dict]  # 检测到的冲突
    recommendations: list[str]  # 后续建议
    agent_outputs: dict[str, str]  # job_id → output 映射
    missing_areas: list[str]  # 未覆盖的领域
    consensus_level: str  # "high" / "medium" / "low"
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "team_id": self.team_id,
            "objective": self.objective,
            "status": self.status,
            "summary": self.summary,
            "findings": self.findings,
            "conflicts": self.conflicts,
            "recommendations": self.recommendations,
            "agent_outputs": self.agent_outputs,
            "missing_areas": self.missing_areas,
            "consensus_level": self.consensus_level,
            "created_at": self.created_at,
        }


class TeamSynthesizer:
    """团队结果合成器 — 将多个 Agent 的输出合并为综合结论。

    合成策略：
    1. 提取关键发现（事实性陈述）
    2. 交叉验证（多个 Agent 提到的事实权重更高）
    3. 冲突检测（两个 Agent 给出矛盾结论时标记）
    4. 缺失领域识别（哪些方面未被覆盖）
    5. 一致性评估（Agent 间结论的一致程度）
    """

    def synthesize(
        self,
        team_id: str,
        objective: str,
        agent_results: dict[str, dict],  # job_id → {role, status, result, error}
    ) -> SynthesisResult:
        """合成团队结果。

        Args:
            team_id: 团队 ID
            objective: 团队目标
            agent_results: 各 Agent 的结果映射

        Returns:
            SynthesisResult: 合成结果
        """
        findings: list[dict] = []
        all_outputs: dict[str, str] = {}
        errors: list[str] = []

        # 1. 提取各 Agent 的关键发现
        for job_id, result in agent_results.items():
            role = result.get("role", "unknown")
            status = result.get("status", "unknown")
            output = result.get("result", "") or ""
            error = result.get("error", "")

            all_outputs[job_id] = output

            if status == "failed":
                errors.append(f"[{role}] {error}")
                continue

            # 提取事实性陈述和关键发现
            extracted = self._extract_findings(output, role)
            for f in extracted:
                f["job_id"] = job_id
                f["role"] = role
            findings.extend(extracted)

        # 2. 交叉验证：多个 Agent 提到的发现权重更高
        findings = self._cross_validate(findings)

        # 3. 冲突检测
        conflicts = self._detect_conflicts(findings)

        # 4. 确定状态
        completed_count = sum(
            1 for r in agent_results.values() if r.get("status") == "completed"
        )
        total_count = len(agent_results)

        if conflicts:
            status = "conflict"
        elif completed_count == total_count:
            status = "complete"
        else:
            status = "partial"

        # 5. 一致性评估
        consensus_level = self._assess_consensus(findings, conflicts)

        # 6. 识别缺失领域
        missing_areas = self._identify_missing(objective, findings)

        # 7. 生成综合摘要
        summary = self._generate_summary(objective, findings, conflicts, errors, consensus_level)

        # 8. 后续建议
        recommendations = self._generate_recommendations(
            findings, conflicts, missing_areas, errors
        )

        return SynthesisResult(
            team_id=team_id,
            objective=objective,
            status=status,
            summary=summary,
            findings=findings,
            conflicts=conflicts,
            recommendations=recommendations,
            agent_outputs=all_outputs,
            missing_areas=missing_areas,
            consensus_level=consensus_level,
        )

    def _extract_findings(self, output: str, role: str) -> list[dict]:
        """从 Agent 输出中提取关键发现。"""
        if not output:
            return []

        findings: list[dict] = []

        # 查找列表项（- 或 * 开头或编号的行）
        lines = output.split("\n")
        for line in lines:
            stripped = line.strip()
            # 匹配 Markdown 列表项
            if re.match(r"^[-*]\s+", stripped):
                text = re.sub(r"^[-*]\s+", "", stripped)
                if len(text) > 20:  # 过滤太短的
                    findings.append({
                        "text": text,
                        "role": role,
                        "confidence": "medium",
                        "source": "agent_output",
                    })
            # 匹配编号列表项
            elif re.match(r"^\d+[\.\)]\s+", stripped):
                text = re.sub(r"^\d+[\.\)]\s+", "", stripped)
                if len(text) > 20:
                    findings.append({
                        "text": text,
                        "role": role,
                        "confidence": "medium",
                        "source": "agent_output",
                    })

        # 查找事实性陈述（包含关键动词/名词的行）
        fact_patterns = [
            r"(发现|找到|检测到|观察到|确认|验证)",
            r"(found|detected|observed|identified|confirmed|verified)",
            r"(STAR|DESeq2|HISAT2|FastQC|BWA|GATK|Seurat)",
            r"(GRCh38|hg38|mm10|mm39|参考基因组|reference genome)",
            r"(参数|parameter|version|版本).*[:：]",
        ]

        for line in lines:
            stripped = line.strip()
            if len(stripped) < 30:
                continue
            for pattern in fact_patterns:
                if re.search(pattern, stripped, re.IGNORECASE):
                    # 避免重复
                    if not any(f["text"] == stripped for f in findings):
                        findings.append({
                            "text": stripped[:300],
                            "role": role,
                            "confidence": "high",
                            "source": "fact_statement",
                        })

        return findings

    def _cross_validate(self, findings: list[dict]) -> list[dict]:
        """交叉验证：多个 Agent 提到的事实权重更高。"""
        # 按文本相似度分组
        validated: list[dict] = []
        grouped: list[list[int]] = []

        for i, f1 in enumerate(findings):
            if any(i in g for g in grouped):
                continue
            group = [i]
            for j, f2 in enumerate(findings[i + 1:], start=i + 1):
                if any(j in g for g in grouped):
                    continue
                sim = _text_similarity(f1["text"], f2["text"])
                if sim >= 0.5:
                    group.append(j)
            grouped.append(group)

        for group in grouped:
            items = [findings[i] for i in group]
            # 多个 Agent 提到 → 高置信度
            unique_roles = set(item.get("role", "") for item in items)
            if len(unique_roles) >= 2:
                confidence = "high"
            elif items[0].get("source") == "fact_statement":
                confidence = "high"
            else:
                confidence = items[0].get("confidence", "medium")

            validated.append({
                "text": items[0]["text"],
                "mentioned_by": len(unique_roles),
                "roles": list(unique_roles),
                "confidence": confidence,
                "occurrence_count": len(items),
            })

        return sorted(validated, key=lambda f: (f["occurrence_count"], f["mentioned_by"]), reverse=True)

    def _detect_conflicts(self, findings: list[dict]) -> list[dict]:
        """检测冲突：两个发现互相矛盾时标记。"""
        conflicts: list[dict] = []

        # 冲突关键词对
        conflict_patterns = [
            (r"(STAR|HISAT2|Bowtie2)", r"(STAR|HISAT2|Bowtie2)"),  # 不同比对工具推荐
            (r"(DESeq2|edgeR|limma)", r"(DESeq2|edgeR|limma)"),  # 不同差异分析工具
            (r"(GRCh38|hg38|hg19|GRCh37)", r"(GRCh38|hg38|hg19|GRCh37)"),  # 不同参考基因组
            (r"(paired.?end|PE)", r"(single.?end|SE)"),  # 不同测序策略
            (r"(支持|confirm|验证|成功)", r"(否定|contradict|失败|无效)"),  # 正反结论
        ]

        for i in range(len(findings)):
            for j in range(i + 1, len(findings)):
                f1, f2 = findings[i], findings[j]

                for pat_a, pat_b in conflict_patterns:
                    match_a1 = re.search(pat_a, f1["text"], re.IGNORECASE)
                    match_b1 = re.search(pat_b, f1["text"], re.IGNORECASE)
                    match_a2 = re.search(pat_a, f2["text"], re.IGNORECASE)
                    match_b2 = re.search(pat_b, f2["text"], re.IGNORECASE)

                    # 两个发现推荐了不同的工具/版本/策略
                    if match_a1 and match_b2 and match_a1.group(1) != match_b2.group(1):
                        if "contradict" not in f1["text"].lower() and "contradict" not in f2["text"].lower():
                            conflicts.append({
                                "finding_1": f1["text"][:200],
                                "finding_2": f2["text"][:200],
                                "conflict_type": "tool_recommendation",
                                "detail": f"Agent 对 {pat_a.pattern} 给出了不同推荐",
                                "resolution": "需要用户根据实验具体需求决定",
                            })
                            break

        return conflicts

    def _assess_consensus(
        self,
        findings: list[dict],
        conflicts: list[dict],
    ) -> str:
        """评估 Agent 间共识程度。"""
        if not findings:
            return "low"

        high_conf = sum(1 for f in findings if f.get("confidence") == "high")
        multi_mentioned = sum(1 for f in findings if f.get("mentioned_by", 0) >= 2)

        if conflicts:
            return "low"
        elif high_conf / max(len(findings), 1) >= 0.6 and multi_mentioned >= 3:
            return "high"
        elif multi_mentioned >= 1:
            return "medium"
        return "low"

    def _identify_missing(self, objective: str, findings: list[dict]) -> list[str]:
        """识别未覆盖的分析领域。"""
        missing: list[str] = []
        objective_lower = objective.lower()

        # 检查常见分析维度是否被覆盖
        coverage_checks = {
            "数据质量": ["quality", "qc", "fastqc", "质控", "质量"],
            "比对": ["alignment", "mapping", "比对", "mapped"],
            "定量": ["quantification", "count", "表达量", "定量"],
            "差异分析": ["differential", "deg", "差异", "差异表达"],
            "功能富集": ["enrichment", "go", "kegg", "pathway", "富集", "通路"],
            "可视化": ["visualization", "plot", "figure", "图", "可视化"],
            "批次效应": ["batch", "批次", "batch effect"],
            "重复性": ["replicate", "reproducibility", "重复", "可重复"],
        }

        all_finding_text = " ".join(f["text"].lower() for f in findings)

        for dimension, keywords in coverage_checks.items():
            if not any(kw in all_finding_text for kw in keywords):
                if any(kw in objective_lower for kw in keywords):
                    missing.append(f"{dimension}: 目标中包含但分析结果中未覆盖")

        return missing

    def _generate_summary(
        self,
        objective: str,
        findings: list[dict],
        conflicts: list[dict],
        errors: list[str],
        consensus: str,
    ) -> str:
        """生成综合摘要。"""
        parts = [f"## 团队综合报告: {objective}\n"]

        # 关键数字
        high_findings = [f for f in findings if f.get("confidence") == "high"]
        parts.append(f"- 关键发现: {len(high_findings)} 条 (高置信度)")
        parts.append(f"- 总发现: {len(findings)} 条")
        parts.append(f"- 冲突: {len(conflicts)} 条")
        parts.append(f"- 错误: {len(errors)} 条")
        parts.append(f"- 共识程度: {consensus}\n")

        # 高置信度发现
        if high_findings:
            parts.append("### 高置信度发现")
            for i, f in enumerate(high_findings[:5], 1):
                mentioned = f.get("mentioned_by", 1)
                parts.append(f"{i}. {f['text']} (被 {mentioned} 个 Agent 确认)")

        # 冲突
        if conflicts:
            parts.append("\n### ⚠️ 冲突")
            for i, c in enumerate(conflicts, 1):
                parts.append(f"{i}. {c.get('conflict_type', 'unknown')}: {c.get('detail', '')}")

        # 错误
        if errors:
            parts.append("\n### ❌ 执行错误")
            for e in errors:
                parts.append(f"- {e}")

        return "\n".join(parts)

    def _generate_recommendations(
        self,
        findings: list[dict],
        conflicts: list[dict],
        missing_areas: list[str],
        errors: list[str],
    ) -> list[str]:
        """生成后续建议。"""
        recs: list[str] = []

        if conflicts:
            recs.append(f"解决 {len(conflicts)} 条冲突后再继续分析")

        if missing_areas:
            for area in missing_areas:
                recs.append(f"补充分析: {area}")

        if errors:
            recs.append(f"修复 {len(errors)} 个 Agent 错误后重新运行")

        if not findings:
            recs.append("未获得有效发现，建议重新设计团队任务分配")

        if not recs:
            recs.append("所有分析步骤已完成，可以进行下游验证")

        return recs


# ---------------------------------------------------------------------------
# Team Report Generator
# ---------------------------------------------------------------------------


class TeamReportGenerator:
    """团队综合报告生成器 — 生成结构化的 Markdown 报告。"""

    def generate_markdown(self, synthesis: SynthesisResult) -> str:
        """生成 Markdown 格式的综合报告。"""
        lines = [
            f"# SciCoreAgent 团队分析报告",
            f"",
            f"**团队 ID**: {synthesis.team_id}",
            f"**目标**: {synthesis.objective}",
            f"**状态**: {synthesis.status}",
            f"**共识程度**: {synthesis.consensus_level}",
            f"**生成时间**: {synthesis.created_at}",
            f"",
            f"---",
            f"",
            f"## 综合摘要",
            f"",
            synthesis.summary,
            f"",
        ]

        # 发现列表
        if synthesis.findings:
            lines.extend([
                f"## 关键发现 ({len(synthesis.findings)} 条)",
                f"",
                "| # | 发现 | 来源 Agent 数 | 置信度 |",
                "|---|------|-------------|--------|",
            ])
            for i, f in enumerate(synthesis.findings[:20], 1):
                mentioned = f.get("mentioned_by", 1)
                conf = f.get("confidence", "medium")
                text = f["text"][:100].replace("|", "\\|")
                lines.append(f"| {i} | {text} | {mentioned} | {conf} |")
            lines.append("")

        # 冲突
        if synthesis.conflicts:
            lines.extend([
                f"## ⚠️ 冲突 ({len(synthesis.conflicts)} 条)",
                f"",
            ])
            for i, c in enumerate(synthesis.conflicts, 1):
                lines.extend([
                    f"### 冲突 {i}: {c.get('conflict_type', 'unknown')}",
                    f"",
                    f"**发现 A**: {c.get('finding_1', '')[:200]}",
                    f"",
                    f"**发现 B**: {c.get('finding_2', '')[:200]}",
                    f"",
                    f"**建议解决方案**: {c.get('resolution', '请用户裁决')}",
                    f"",
                ])

        # 缺失领域
        if synthesis.missing_areas:
            lines.extend([
                f"## 🔍 未覆盖领域",
                f"",
            ])
            for area in synthesis.missing_areas:
                lines.append(f"- {area}")
            lines.append("")

        # 建议
        if synthesis.recommendations:
            lines.extend([
                f"## 📋 后续建议",
                f"",
            ])
            for i, rec in enumerate(synthesis.recommendations, 1):
                lines.append(f"{i}. {rec}")
            lines.append("")

        # Agent 输出摘要
        lines.extend([
            f"## 📊 各 Agent 输出摘要",
            f"",
        ])
        for job_id, output in synthesis.agent_outputs.items():
            output_preview = output[:300].replace("\n", " ") if output else "(空)"
            lines.append(f"### {job_id}")
            lines.append(f"```")
            lines.append(output_preview)
            lines.append(f"```")
            lines.append("")

        return "\n".join(lines)

    def generate_json(self, synthesis: SynthesisResult) -> str:
        """生成 JSON 格式的综合报告。"""
        return json.dumps(synthesis.to_dict(), ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------


def _text_similarity(text1: str, text2: str) -> float:
    """计算两个文本的简单相似度（3-gram Jaccard）。"""
    if not text1 or not text2:
        return 0.0

    def ngrams(s: str, n: int = 3) -> set:
        s = re.sub(r"\s+", " ", s.lower())
        return {s[i:i + n] for i in range(len(s) - n + 1)}

    a = ngrams(text1)
    b = ngrams(text2)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
