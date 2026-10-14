import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const repoRoot = process.cwd();
const localRuns = path.join(repoRoot, ".scicore", "bixbench_runs");
const externalRuns = "F:\\aicoding\\bixbench_runs";
const outputDir = path.join(localRuns, "reports");
const outputPath = path.join(outputDir, "bixbench_results_summary.xlsx");

async function findReports(root) {
  const reports = [];
  async function walk(dir) {
    let entries = [];
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      if (entry.isFile() && entry.name === "bixbench_report.json") reports.push(full);
    }
  }
  await walk(root);
  return reports;
}

function num(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function classify(error, firstFailure, strategy, passed) {
  const lowered = String(error || "").toLowerCase();
  if (passed > 0) return "passed";
  if (lowered.includes("insufficient_quota") || lowered.includes("free quota has been exhausted")) {
    return "insufficient_quota";
  }
  if (firstFailure) return firstFailure;
  if (strategy === "scicore_bio_agent") return "agent_not_passed";
  return "baseline_or_non_agent";
}

async function readReport(reportPath) {
  const report = JSON.parse(await fs.readFile(reportPath, "utf8"));
  const stat = await fs.stat(reportPath);
  const benchmark = report.benchmark ?? {};
  const summary = report.summary ?? {};
  const first = Array.isArray(report.tasks) && report.tasks.length ? report.tasks[0] : {};
  const error = String(first.error || "").replace(/\s+/g, " ").slice(0, 260);
  const passed = num(summary.passed);
  return {
    run: path.basename(path.dirname(reportPath)),
    strategy: benchmark.strategy ?? "",
    taskCount: num(summary.total_tasks ?? benchmark.task_count),
    attempted: num(summary.attempted),
    passed,
    failed: num(summary.failed),
    timeout: num(summary.timeout),
    runtimeErrors: num(summary.error),
    wrongAnswers: num(summary.wrong_answer),
    passAt1: num(summary.pass_at_1 ?? summary.pass_rate),
    firstStatus: first.status ?? "",
    firstFailure: first.failure_category ?? "",
    conclusion: classify(error, first.failure_category ?? "", benchmark.strategy ?? "", passed),
    firstError: error,
    reportPath,
    modifiedMs: stat.mtimeMs,
    failureCounts: JSON.stringify(summary.failure_counts ?? {}),
  };
}

const reportPaths = [
  ...(await findReports(localRuns)),
  ...(await findReports(externalRuns)),
];
const rows = [];
for (const reportPath of reportPaths) {
  try {
    rows.push(await readReport(reportPath));
  } catch {
    // Ignore malformed partial reports.
  }
}
rows.sort((a, b) => a.run.localeCompare(b.run));

const round50Names = new Set([
  "official-agent-50-round1",
  "official-agent-50-round1-qwenplus",
  "official-agent-50-deepseekchat-v2",
  "official-agent-50-deepseekchat-v3",
  "official-agent-50-deepseekchat-v4",
  "official-agent-50-deepseekchat-v5",
  "official-agent-50-deepseekchat-v6",
  "official-agent-50-deepseekchat-v7",
  "official-agent-50-deepseekchat-v8",
  "official-agent-50-deepseekchat-v9",
  "official-agent-50-deepseekchat-v10",
  "official-agent-50-deepseekchat-v11",
  "official-agent-50-deepseekchat-v12",
  "official-agent-50-deepseekchat-v13",
  "official-agent-50-deepseekchat-v14",
  "official-agent-50-deepseekchat-v15",
  "official-agent-50-deepseekchat-v16",
]);
const round50Rows = rows.filter((row) => round50Names.has(row.run));
const best50 = round50Rows.length ? [...round50Rows].sort((a, b) => b.passAt1 - a.passAt1)[0] : null;
const latestAgent = rows
  .filter((row) => row.strategy === "scicore_bio_agent")
  .sort((a, b) => a.modifiedMs - b.modifiedMs)
  .at(-1);

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Summary");
const round50Sheet = workbook.worksheets.add("Round50");
const runsSheet = workbook.worksheets.add("AllRuns");

for (const sheet of [summarySheet, round50Sheet, runsSheet]) {
  sheet.showGridLines = false;
}

const conclusion = best50
  ? `50-task evaluation completed. Best run: ${best50.run}, pass@1 ${(best50.passAt1 * 100).toFixed(2)}%.`
  : latestAgent
    ? `Latest agent run: ${latestAgent.run}, conclusion ${latestAgent.conclusion}.`
    : "No real scicore_bio_agent run found.";

summarySheet.getRange("A1:F1").values = [["BixBench evaluation status", "", "", "", "", ""]];
summarySheet.getRange("A1:F1").merge();
summarySheet.getRange("A1:F1").format = {
  fill: "#174A7C",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};

summarySheet.getRange("A3:B9").values = [
  ["Current conclusion", conclusion],
  ["Dataset", "F:\\aicoding\\bixbench, official task count 205"],
  ["Best 50-task run", best50 ? best50.run : "not available"],
  ["Best 50-task pass@1", best50 ? best50.passAt1 : null],
  ["Best 50-task passed", best50 ? `${best50.passed}/${best50.taskCount}` : "not available"],
  ["Main blocker", "Remaining failures are RNA-seq / GO enrichment tasks that require a real DESeq2 + clusterProfiler backend or equivalent reproducible reanalysis"],
  ["Next step", "Add a controlled R/Bioconductor execution backend for BixBench RNA-seq capsules, then rerun the same 50-task set"],
];
summarySheet.getRange("A3:A9").format = {
  fill: "#EAF2F8",
  font: { bold: true, color: "#1F2937" },
};
summarySheet.getRange("B3:B9").format.wrapText = true;
summarySheet.getRange("B6").format.numberFormat = "0.00%";

summarySheet.getRange("D3:F13").values = [
  ["Metric", "Value", "Note"],
  ["Reports scanned", rows.length, "All discovered report JSON files"],
  ["Question-only pass@1", rows.find((row) => row.run === "official-round1-question-only")?.passAt1 ?? null, "Non-agent baseline"],
  ["DeepSeek v2 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v2")?.passAt1 ?? null, "Official DeepSeek key"],
  ["DeepSeek v4 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v4")?.passAt1 ?? null, "Notebook/statistics extraction added"],
  ["DeepSeek v5 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v5")?.passAt1 ?? null, "SCOGS treeness extraction added"],
  ["DeepSeek v6 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v6")?.passAt1 ?? null, "Alignment parsimony extraction added"],
  ["DeepSeek v7 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v7")?.passAt1 ?? null, "SCOGS saturation extraction added"],
  ["DeepSeek v9 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v9")?.passAt1 ?? null, "CRISPR correlation parser stabilized"],
  ["DeepSeek v13 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v13")?.passAt1 ?? null, "BCG, imaging, NeuN, and variant parsers added"],
  ["DeepSeek v16 50 pass@1", rows.find((row) => row.run === "official-agent-50-deepseekchat-v16")?.passAt1 ?? null, "Current best deterministic parser run"],
];
summarySheet.getRange("D3:F3").format = {
  fill: "#174A7C",
  font: { bold: true, color: "#FFFFFF" },
};
summarySheet.getRange("E5:E13").format.numberFormat = "0.00%";

const round50Headers = [
  "run",
  "tasks",
  "attempted",
  "passed",
  "failed",
  "runtime_errors",
  "wrong_answers",
  "pass_at_1",
  "failure_counts",
  "report_path",
];
round50Sheet.getRangeByIndexes(0, 0, 1, round50Headers.length).values = [round50Headers];
if (round50Rows.length) {
  round50Sheet.getRangeByIndexes(1, 0, round50Rows.length, round50Headers.length).values = round50Rows.map((row) => [
    row.run,
    row.taskCount,
    row.attempted,
    row.passed,
    row.failed,
    row.runtimeErrors,
    row.wrongAnswers,
    row.passAt1,
    row.failureCounts,
    row.reportPath,
  ]);
}
round50Sheet.getRangeByIndexes(0, 0, 1, round50Headers.length).format = {
  fill: "#174A7C",
  font: { bold: true, color: "#FFFFFF" },
};
round50Sheet.getRange("H2:H20").format.numberFormat = "0.00%";
round50Sheet.freezePanes.freezeRows(1);

const runHeaders = [
  "run",
  "strategy",
  "conclusion",
  "tasks",
  "attempted",
  "passed",
  "failed",
  "timeout",
  "runtime_errors",
  "wrong_answers",
  "pass_at_1",
  "first_status",
  "first_failure",
  "first_error",
  "report_path",
];
runsSheet.getRangeByIndexes(0, 0, 1, runHeaders.length).values = [runHeaders];
if (rows.length) {
  runsSheet.getRangeByIndexes(1, 0, rows.length, runHeaders.length).values = rows.map((row) => [
    row.run,
    row.strategy,
    row.conclusion,
    row.taskCount,
    row.attempted,
    row.passed,
    row.failed,
    row.timeout,
    row.runtimeErrors,
    row.wrongAnswers,
    row.passAt1,
    row.firstStatus,
    row.firstFailure,
    row.firstError,
    row.reportPath,
  ]);
}
runsSheet.getRangeByIndexes(0, 0, 1, runHeaders.length).format = {
  fill: "#174A7C",
  font: { bold: true, color: "#FFFFFF" },
};
runsSheet.getRange("K2:K500").format.numberFormat = "0.00%";
runsSheet.freezePanes.freezeRows(1);

summarySheet.getRange("A:F").format.autofitColumns();
round50Sheet.getRange("A:J").format.autofitColumns();
runsSheet.getRange("A:O").format.autofitColumns();
summarySheet.getRange("B:B").format.columnWidth = 75;
round50Sheet.getRange("I:J").format.columnWidth = 65;
runsSheet.getRange("N:O").format.columnWidth = 75;
runsSheet.getRange("N:O").format.wrapText = true;

await fs.mkdir(outputDir, { recursive: true });

const inspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:F9",
  include: "values",
  tableMaxRows: 12,
  tableMaxCols: 6,
});
console.log(inspect.ndjson);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 20 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const preview = await workbook.render({
  sheetName: "Summary",
  autoCrop: "all",
  scale: 1,
  format: "png",
});
await fs.writeFile(path.join(outputDir, "bixbench_results_summary_preview.png"), new Uint8Array(await preview.arrayBuffer()));

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
