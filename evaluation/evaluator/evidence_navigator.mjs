import crypto from "node:crypto";
import path from "node:path";

const TEXT_EXTENSIONS = /\.(?:md|txt|json|csv|tsv|py|mjs|js|r|jl|ipynb|yaml|yml|log)$/i;
const FILE_REFERENCE = /(?:`([^`]+)`|\b([A-Za-z0-9_.-]+\.(?:md|txt|json|csv|tsv|py|mjs|js|r|jl|ipynb|yaml|yml|log))\b)/gi;
const TABLE_REFERENCE = /\b(?:table|figure|fig\.?|section|key)\s*[:#-]?\s*([A-Za-z0-9_.-]+)/gi;
const IDENTIFIER = /\b[A-Za-z_][A-Za-z0-9_]{3,}\b/g;
const NUMBER = /(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?%?/gi;
const STOPWORDS = new Set([
  "about", "after", "analysis", "before", "between", "code", "comparison", "condition", "data",
  "discovery", "effect", "evidence", "finding", "from", "result", "results", "should", "table",
  "that", "their", "there", "these", "this", "through", "using", "with", "without",
]);

function sha256(text) {
  return crypto.createHash("sha256").update(text).digest("hex");
}

function normalizeCue(value) {
  return String(value).trim().replace(/^['"]|['"]$/g, "").toLowerCase();
}

function addCue(target, value, weight, kind) {
  const cue = normalizeCue(value);
  if (!cue || cue.length < 2) return;
  const prior = target.get(cue);
  if (!prior || prior.weight < weight) target.set(cue, { cue, weight, kind });
}

export function extractNavigationCues(discoveryText) {
  const cues = new Map();
  for (const match of String(discoveryText).matchAll(FILE_REFERENCE)) {
    const reference = match[1] ?? match[2];
    addCue(cues, reference, 40, "file_reference");
    addCue(cues, path.basename(reference), 40, "file_reference");
    addCue(cues, path.basename(reference, path.extname(reference)), 24, "file_stem");
  }
  for (const match of String(discoveryText).matchAll(TABLE_REFERENCE)) {
    addCue(cues, match[1], 36, "named_locator");
    addCue(cues, `table${match[1]}`, 40, "named_locator");
    addCue(cues, `table_${match[1]}`, 40, "named_locator");
  }
  for (const match of String(discoveryText).matchAll(NUMBER)) addCue(cues, match[0], 18, "number");
  for (const match of String(discoveryText).matchAll(IDENTIFIER)) {
    const token = normalizeCue(match[0]);
    if (!STOPWORDS.has(token)) addCue(cues, token, token.includes("_") ? 20 : 5, "term");
  }
  return [...cues.values()].sort((left, right) => right.weight - left.weight || left.cue.localeCompare(right.cue));
}

function jsonMatches(value, cues, pointer = "$") {
  const matches = [];
  if (Array.isArray(value)) {
    value.forEach((item, index) => matches.push(...jsonMatches(item, cues, `${pointer}[${index}]`)));
    return matches;
  }
  if (!value || typeof value !== "object") return matches;
  for (const [key, child] of Object.entries(value)) {
    const keyLower = key.toLowerCase();
    const score = cues.reduce((sum, item) => sum + (keyLower.includes(item.cue) ? item.weight : 0), 0);
    const childPointer = `${pointer}.${key}`;
    if (score > 0) matches.push({ pointer: childPointer, score, value: child });
    matches.push(...jsonMatches(child, cues, childPointer));
  }
  return matches;
}

function renderJsonFocus(text, cues, limit) {
  let parsed;
  try { parsed = JSON.parse(text); } catch { return null; }
  const matches = jsonMatches(parsed, cues)
    .sort((left, right) => right.score - left.score || left.pointer.localeCompare(right.pointer));
  if (!matches.length) return null;
  const blocks = [];
  let used = 0;
  for (const match of matches) {
    const body = JSON.stringify(match.value, null, 2);
    const block = `// JSON定位 ${match.pointer}\n${body}`;
    if (used + block.length > limit && blocks.length) continue;
    blocks.push(block.slice(0, Math.max(0, limit - used)));
    used += block.length;
    if (used >= limit) break;
  }
  return blocks.join("\n\n");
}

function lineScore(line, cues) {
  const lower = line.toLowerCase();
  return cues.reduce((sum, item) => sum + (lower.includes(item.cue) ? item.weight : 0), 0);
}

function mergeRanges(ranges) {
  const merged = [];
  for (const range of ranges.sort((a, b) => a.start - b.start || a.end - b.end)) {
    const last = merged.at(-1);
    if (last && range.start <= last.end + 1) last.end = Math.max(last.end, range.end);
    else merged.push({ ...range });
  }
  return merged;
}

function renderLineFocus(text, cues, limit) {
  const lines = text.split(/\r?\n/);
  const hits = lines
    .map((line, index) => ({ index, score: lineScore(line, cues) }))
    .filter((row) => row.score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, 24);
  if (!hits.length) return null;
  const ranges = mergeRanges(hits.map((hit) => ({
    start: Math.max(0, hit.index - 8),
    end: Math.min(lines.length - 1, hit.index + 12),
  })));
  const blocks = [];
  let used = 0;
  for (const range of ranges) {
    const numbered = lines.slice(range.start, range.end + 1)
      .map((line, offset) => `${range.start + offset + 1}: ${line}`)
      .join("\n");
    const block = `// 行 ${range.start + 1}-${range.end + 1}\n${numbered}`;
    if (used + block.length > limit && blocks.length) continue;
    blocks.push(block.slice(0, Math.max(0, limit - used)));
    used += block.length;
    if (used >= limit) break;
  }
  return blocks.join("\n\n");
}

export function focusEvidence(bytes, locator, discoveryText, { limit = 8000, extraCues = [] } = {}) {
  if (!TEXT_EXTENSIONS.test(locator)) {
    return {
      content: "[二进制产物：仅核验文件身份与哈希；需要专用解析器才能作为内容证据]",
      strategy: "binary_metadata_only",
      truncated: true,
    };
  }
  const text = bytes.toString("utf8");
  if (text.length <= limit) return { content: text, strategy: "complete_text", truncated: false };
  const cues = extractNavigationCues(`${discoveryText}\n${extraCues.join("\n")}`);
  const jsonFocus = /\.json$/i.test(locator) ? renderJsonFocus(text, cues, limit) : null;
  if (jsonFocus) return { content: jsonFocus, strategy: "json_pointer_focus", truncated: true };
  const lineFocus = renderLineFocus(text, cues, limit);
  if (lineFocus) return { content: lineFocus, strategy: "matched_line_windows", truncated: true };
  const head = Math.floor(limit * 0.6);
  return {
    content: `${text.slice(0, head)}\n...[无定位命中，保留头尾；共省略 ${text.length - limit} 字符]...\n${text.slice(-(limit - head))}`,
    strategy: "head_tail_fallback",
    truncated: true,
  };
}

export function stableEvidenceId(locator, expectedSha256) {
  return `file_${sha256(`${locator}\u0000${expectedSha256}`).slice(0, 20)}`;
}

export function buildEvidenceInventory(rows, discoveryText, tracePaths = new Set()) {
  const references = extractNavigationCues(discoveryText).filter((row) => row.kind === "file_reference");
  return rows.map((row) => {
    const locator = row.path;
    const base = path.basename(locator);
    const explicitlyReferenced = references.some((reference) =>
      locator.toLowerCase().includes(reference.cue) || base.toLowerCase() === reference.cue
    );
    return {
      evidence_id: stableEvidenceId(locator, row.sha256),
      locator,
      sha256: row.sha256,
      trace_bound: tracePaths.has(locator),
      explicitly_referenced: explicitlyReferenced,
      media_type: path.extname(locator).toLowerCase() || "unknown",
    };
  });
}
