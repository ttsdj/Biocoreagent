"""Tool registry."""

from .bash import BashTool
from .read import ReadFileTool
from .write import WriteFileTool
from .edit import EditFileTool
from .glob_tool import GlobTool
from .grep import GrepTool
from .agent import AgentTool
from .multiagent import AgentStartTool, AgentStatusTool, AgentTeamStartTool, AgentTeamStatusTool
from .agent_team import TeamSynthesizeTool, TeamWorkspaceTool
from .context import ContextStatusTool
from .workflow_governance import (
    WorkflowPlanPrepareTool,
    WorkflowPreflightCheckTool,
    WorkflowPrimitiveLedgerTool,
)
from .transcriptome import (
    TranscriptomeCapabilityListTool,
    TranscriptomeOmicVerseCheckTool,
    TranscriptomeOmicVerseDEGTool,
    TranscriptomePlanTool,
    TranscriptomePlanVerifyTool,
    TranscriptomeProvenanceAppendTool,
    TranscriptomeStateInspectTool,
)
from .remote import SSHBashTool
from .extensions import ExtensionsTool
from .mcp_config import MCPListTool, MCPRegisterTool, MCPUnregisterTool
from .skills import SkillInstallTool, SkillListTool, SkillReadTool, SkillSaveTool, SkillSearchTool
from .skill_router import (
    SkillRouteTool,
    SkillChainTool,
    SkillRouteListTool,
    SkillMarkExecutedTool,
)
from .knowledge_base import KBCrossSearchTool, KBCrossEnhanceTool, KBSuggestSkillTool
from .literature import (
    LiteratureExportXlsxTool,
    LiteratureRedBlueReviewTool,
    PubMedFetchDetailsTool,
    PubMedLiteratureReviewTool,
    PubMedSearchTool,
)
from .provenance import (
    CodeLiteratureLinkListTool,
    CodeLiteratureLinkSaveTool,
    CodeLiteratureLinkSearchTool,
)
from .user_profile import UserProfileSetTool, UserProfileGetTool, UserProfileRecommendTool
from .wiki import WikiSaveTool, WikiSearchTool, WikiListTool
from .bio import (
    BioCountMatrixInspectTool,
    BioRNASeqCompareTool,
    BioReportTool,
    BioSampleSheetInspectTool,
    BioSeqInspectTool,
    BioWorkflowSketchTool,
    FileHashTool,
    BioIngestProtocolTool,
    BioExtractProtocolTool,
    BioQueryEvidenceTool,
    BioReplicationPlanTool,
    BioPipelinePlanTool,
    BioPipelineSupportedAssaysTool,
    BioExperimentPlanTool,
    BioRDSInspectTool,
    BioRBridgeTool,
    BioDESeq2TissueVsRestTool,
    BioContingencyTestTool,
    BioRegressionTool,
    BioDESeq2QuickTool,
)

ALL_TOOLS = [
    BashTool(),
    ReadFileTool(),
    WriteFileTool(),
    EditFileTool(),
    GlobTool(),
    GrepTool(),
    AgentTool(),
    AgentStartTool(),
    AgentStatusTool(),
    AgentTeamStartTool(),
    AgentTeamStatusTool(),
    # Team collaboration tools
    TeamSynthesizeTool(),
    TeamWorkspaceTool(),
    ContextStatusTool(),
    WorkflowPreflightCheckTool(),
    WorkflowPrimitiveLedgerTool(),
    WorkflowPlanPrepareTool(),
    TranscriptomeCapabilityListTool(),
    TranscriptomeStateInspectTool(),
    TranscriptomePlanTool(),
    TranscriptomePlanVerifyTool(),
    TranscriptomeProvenanceAppendTool(),
    TranscriptomeOmicVerseCheckTool(),
    TranscriptomeOmicVerseDEGTool(),
    SSHBashTool(),
    ExtensionsTool(),
    MCPRegisterTool(),
    MCPListTool(),
    MCPUnregisterTool(),
    SkillInstallTool(),
    SkillSaveTool(),
    SkillListTool(),
    SkillReadTool(),
    SkillSearchTool(),
    # Skill router tools
    SkillRouteTool(),
    SkillChainTool(),
    SkillRouteListTool(),
    SkillMarkExecutedTool(),
    # Unified knowledge base tools
    KBCrossSearchTool(),
    KBCrossEnhanceTool(),
    KBSuggestSkillTool(),
    # Literature retrieval and red-blue evidence review
    PubMedSearchTool(),
    PubMedFetchDetailsTool(),
    PubMedLiteratureReviewTool(),
    LiteratureRedBlueReviewTool(),
    LiteratureExportXlsxTool(),
    CodeLiteratureLinkSaveTool(),
    CodeLiteratureLinkSearchTool(),
    CodeLiteratureLinkListTool(),
    # User profile tools
    UserProfileSetTool(),
    UserProfileGetTool(),
    UserProfileRecommendTool(),
    # Wiki knowledge-base tools
    WikiSaveTool(),
    WikiSearchTool(),
    WikiListTool(),
    # Existing bioinformatics tools
    BioSeqInspectTool(),
    FileHashTool(),
    BioWorkflowSketchTool(),
    BioSampleSheetInspectTool(),
    BioCountMatrixInspectTool(),
    BioRNASeqCompareTool(),
    BioReportTool(),
    # Protocol RAG tools
    BioIngestProtocolTool(),
    BioExtractProtocolTool(),
    BioQueryEvidenceTool(),
    BioReplicationPlanTool(),
    # Pipeline planning & experiment design
    BioPipelinePlanTool(),
    BioPipelineSupportedAssaysTool(),
    BioExperimentPlanTool(),
    # R integration & RDS inspection
    BioRDSInspectTool(),
    BioRBridgeTool(),
    BioDESeq2TissueVsRestTool(),
    # Pre-validated statistical analysis (OmicOS-style)
    BioContingencyTestTool(),
    BioRegressionTool(),
    BioDESeq2QuickTool(),
]


def get_tool(name: str):
    """Look up a tool by name."""
    for t in ALL_TOOLS:
        if t.name == name:
            return t
    return None
