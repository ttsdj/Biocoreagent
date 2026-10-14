"""Agent 团队协作工具 — 结果合成、冲突检测、综合报告。"""

from __future__ import annotations

import json

from .base import Tool
from ..multiagent import GLOBAL_MULTIAGENT_MANAGER
from ..agent_team import TeamSynthesizer, TeamReportGenerator, SharedWorkspace


class TeamSynthesizeTool(Tool):
    """合成团队结果的工具。"""

    name = "team_synthesize"
    description = (
        "合成多 Agent 团队的输出结果：提取关键发现、交叉验证、检测冲突、"
        "评估共识程度、识别缺失领域，生成综合报告。"
        "在 agent_team_status 返回所有 Agent 完成后调用此工具。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "team_id": {
                "type": "string",
                "description": "团队 ID（来自 agent_team_start 的返回值）",
            },
            "output_format": {
                "type": "string",
                "description": "输出格式: markdown 或 json",
                "enum": ["markdown", "json"],
                "default": "markdown",
            },
        },
        "required": ["team_id"],
    }

    def execute(self, team_id: str, output_format: str = "markdown") -> str:
        # 获取团队状态
        status = GLOBAL_MULTIAGENT_MANAGER.team_status(team_id)
        if status is None:
            return json.dumps({
                "success": False,
                "error": f"团队 '{team_id}' 不存在",
            }, ensure_ascii=False, indent=2)

        # 检查是否所有 Agent 已完成
        running = status["summary"].get("running", 0)
        if running > 0:
            return json.dumps({
                "success": False,
                "error": f"团队中还有 {running} 个 Agent 正在运行，等待完成后再合成",
                "team_status": status["status"],
            }, ensure_ascii=False, indent=2)

        # 构建 agent_results
        agent_results = {}
        for job in status["jobs"]:
            agent_results[job["job_id"]] = {
                "role": job["role"],
                "status": job["status"],
                "result": job.get("result"),
                "error": job.get("error"),
            }

        # 合成
        synthesizer = TeamSynthesizer()
        synthesis = synthesizer.synthesize(
            team_id=team_id,
            objective=status["objective"],
            agent_results=agent_results,
        )

        # 生成报告
        generator = TeamReportGenerator()

        if output_format == "json":
            report = generator.generate_json(synthesis)
        else:
            report = generator.generate_markdown(synthesis)

        return json.dumps({
            "success": True,
            "team_id": team_id,
            "synthesis": synthesis.to_dict(),
            "report": report,
            "report_format": output_format,
        }, ensure_ascii=False, indent=2)


class TeamWorkspaceTool(Tool):
    """共享工作区操作工具。"""

    name = "team_workspace"
    description = (
        "操作 Agent 团队的共享工作区：写入/读取/列出共享文件。"
        "多个 Agent 可通过此工作区交换中间数据。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "team_id": {
                "type": "string",
                "description": "团队 ID",
            },
            "action": {
                "type": "string",
                "description": "操作: write, read, list, clean",
                "enum": ["write", "read", "list", "clean"],
            },
            "filename": {
                "type": "string",
                "description": "文件名（write/read 操作需要）",
            },
            "content": {
                "type": "string",
                "description": "写入内容（write 操作需要）",
            },
        },
        "required": ["team_id", "action"],
    }

    def execute(
        self,
        team_id: str,
        action: str,
        filename: str = "",
        content: str = "",
    ) -> str:
        workspace = SharedWorkspace(team_id)
        workspace.init()

        if action == "write":
            if not filename:
                return json.dumps({"success": False, "error": "filename 不能为空"})
            path = workspace.write(filename, content)
            return json.dumps({
                "success": True,
                "action": "write",
                "filepath": str(path),
                "size": len(content),
            })

        elif action == "read":
            if not filename:
                return json.dumps({"success": False, "error": "filename 不能为空"})
            data = workspace.read(filename)
            if data is None:
                return json.dumps({"success": False, "error": f"文件 '{filename}' 不存在"})
            return json.dumps({
                "success": True,
                "action": "read",
                "filename": filename,
                "content": data,
            })

        elif action == "list":
            files = workspace.list_files()
            return json.dumps({
                "success": True,
                "action": "list",
                "team_id": team_id,
                "files": files,
                "file_count": len(files),
            })

        elif action == "clean":
            workspace.clean()
            return json.dumps({
                "success": True,
                "action": "clean",
                "message": f"工作区 '{team_id}' 已清理",
            })

        return json.dumps({"success": False, "error": f"未知操作: {action}"})
