import fs from "node:fs";
import path from "node:path";

const EXTENSIONS = "md|txt|json|csv|tsv|py|mjs|js|r|jl|ipynb|yaml|yml|log|html|htm|pdf|png|jpg|jpeg|svg|parquet|feather|xlsx|xls|npy|npz|h5|hdf5";
const INLINE_CODE = /`([^`\n]+)`/g;
const FILE_TOKEN = new RegExp(`(?:^|[\\s'\"=:(])((?:\\/workspace\\/|\\.?\\.?\\/)?[A-Za-z0-9_{}*?.,+@%\\/\\\\-]+\\.(?:${EXTENSIONS})(?::\\d+)?)(?=$|[\\s'\"),;])`, "gi");
const PLAIN_FILE = new RegExp(`\\b([A-Za-z0-9_{}*?.,+@%./\\\\-]+\\.(?:${EXTENSIONS})(?::\\d+)?)\\b`, "gi");
const inventoryCache = new Map();

function cleanReference(value) {
  let result = String(value).trim().replace(/\\/g, "/").replace(/:\d+$/, "").replace(/[?#].*$/, "");
  result = result.replace(/^['\"]|['\"]$/g, "").replace(/^\.\//, "");
  if (result === "workspace") return "";
  if (result.startsWith("workspace/")) result = result.slice("workspace/".length);
  return result;
}

function inside(root, file) {
  const resolvedRoot = path.resolve(root);
  const resolved = path.resolve(file);
  return resolved === resolvedRoot || resolved.startsWith(`${resolvedRoot}${path.sep}`);
}

function existingFile(file) {
  try { return fs.statSync(file).isFile(); } catch { return false; }
}

function safeWorkspaceInventory(workspaceRoot) {
  const key = path.resolve(workspaceRoot);
  if (inventoryCache.has(key)) return inventoryCache.get(key);
  const result = [];
  const ignored = new Set([".git", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache"]);
  const queue = [key];
  while (queue.length && result.length < 50000) {
    const current = queue.pop();
    let entries = [];
    try { entries = fs.readdirSync(current, { withFileTypes: true }); } catch { continue; }
    for (const entry of entries) {
      if (ignored.has(entry.name)) continue;
      const absolute = path.join(current, entry.name);
      if (!inside(key, absolute)) continue;
      if (entry.isDirectory()) queue.push(absolute);
      else if (entry.isFile()) result.push({ absolute, relative: path.relative(key, absolute).replace(/\\/g, "/") });
    }
  }
  inventoryCache.set(key, result);
  return result;
}

function globRegex(pattern) {
  const input = cleanReference(pattern).replace(/^\/workspace\//, "");
  let source = "";
  for (let index = 0; index < input.length; index += 1) {
    const rest = input.slice(index);
    const range = rest.match(/^\{(\d+)(?:\.\.|-)(\d+)\}/);
    if (range) {
      const values = [];
      for (let value = Number(range[1]); value <= Number(range[2]); value += 1) values.push(String(value));
      source += `(?:${values.join("|")})`; index += range[0].length - 1; continue;
    }
    const character = input[index];
    if (character === "*") source += "[^/]*";
    else if (character === "?") source += "[^/]";
    else source += character.replace(/[\\^$.*+?()[\]{}|]/g, "\\$&");
  }
  return new RegExp(`(?:^|/)${source}$`, "i");
}

function segmentCompatible(candidate, wanted) {
  return candidate === wanted || candidate.startsWith(`${wanted}_`) || candidate.startsWith(`${wanted}-`);
}

function relaxedSuffixMatch(relative, wantedSuffix) {
  const candidateParts = relative.toLowerCase().split("/");
  const wantedParts = wantedSuffix.toLowerCase().split("/");
  if (wantedParts.length > candidateParts.length) return false;
  const tail = candidateParts.slice(-wantedParts.length);
  return tail.every((candidate, index) => index === tail.length - 1
    ? candidate === wantedParts[index]
    : segmentCompatible(candidate, wantedParts[index]));
}

function stemTokens(value) {
  return path.basename(value).replace(/\.[^.]+$/, "").toLowerCase().split(/[^a-z0-9]+/).filter((row) => row.length >= 2 && !["file", "data", "result", "output"].includes(row));
}

function contextualCandidate(candidates, reference, contextText, workspaceRoot) {
  if (!candidates.length) return null;
  const context = String(contextText ?? "").toLowerCase();
  const wanted = stemTokens(reference);
  const scored = candidates.map((row) => {
    const relative = path.relative(workspaceRoot, row.absolute).replace(/\\/g, "/").toLowerCase();
    const tokens = new Set(relative.split(/[^a-z0-9]+/).filter((token) => token.length >= 2));
    let score = wanted.reduce((sum, token) => sum + (tokens.has(token) ? 4 : 0), 0);
    for (const token of tokens) if (token.length >= 4 && context.includes(token)) score += 1;
    return { row, score };
  }).sort((a, b) => b.score - a.score || a.row.relative.localeCompare(b.row.relative));
  if (scored[0].score >= 4 && (!scored[1] || scored[0].score - scored[1].score >= 2)) return scored[0].row;
  return null;
}

function registeredInventory(outputRoot, registeredPaths) {
  return registeredPaths.map((relative) => {
    const normalized = cleanReference(relative).replace(/^output\//, "");
    return { relative: normalized, absolute: path.resolve(outputRoot, normalized) };
  }).filter((row) => inside(outputRoot, row.absolute) && existingFile(row.absolute));
}

export function extractReportedFileReferences(text) {
  const result = [];
  const add = (value) => {
    const cleaned = cleanReference(value);
    if (cleaned && !result.includes(cleaned)) result.push(cleaned);
  };
  const source = String(text);
  for (const code of source.matchAll(INLINE_CODE)) {
    for (const match of code[1].matchAll(FILE_TOKEN)) add(match[1]);
  }
  const withoutCode = source.replace(INLINE_CODE, " ");
  for (const match of withoutCode.matchAll(PLAIN_FILE)) add(match[1]);
  return result;
}

export function resolveReportedPath(reference, { workspaceRoot, outputRoot, registeredPaths = [], contextText = "" }) {
  const cleaned = cleanReference(reference);
  const attempts = [];
  const add = (strategy, candidate) => {
    const resolved = path.resolve(candidate);
    if (!inside(workspaceRoot, resolved)) return;
    if (!attempts.some((row) => row.path === resolved)) attempts.push({ strategy, path: resolved });
  };

  if (path.isAbsolute(cleaned)) {
    if (cleaned === "/workspace" || cleaned.startsWith("/workspace/")) {
      add("container_workspace_remap", path.join(workspaceRoot, cleaned.slice("/workspace/".length)));
    }
    add("absolute_within_workspace", cleaned);
  } else {
    add("workspace_relative", path.join(workspaceRoot, cleaned));
    add("output_relative", path.join(outputRoot, cleaned.replace(/^output\//, "")));
  }

  const exact = attempts.find((row) => existingFile(row.path));
  if (exact) return { reference, cleaned, status: "resolved", strategy: exact.strategy, path: exact.path, attempted_paths: attempts.map((row) => row.path) };

  const registered = registeredInventory(outputRoot, registeredPaths);
  const workspace = safeWorkspaceInventory(workspaceRoot);
  if (/[?*{]/.test(cleaned)) {
    const matcher = globRegex(cleaned);
    const matches = [...new Map([...registered, ...workspace].filter((row) => matcher.test(row.relative)).map((row) => [row.absolute, row])).values()];
    if (matches.length) return {
      reference, cleaned, status: "resolved", strategy: "safe_glob_expansion", path: matches[0].absolute,
      matched_paths: matches.map((row) => row.absolute), attempted_paths: attempts.map((row) => row.path),
    };
  }

  const wantedSuffix = cleaned.replace(/^output\//, "").toLowerCase();
  const registeredSuffix = registered.filter((row) => row.relative.toLowerCase().endsWith(wantedSuffix));
  if (registeredSuffix.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_registered_suffix", path: registeredSuffix[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };
  const workspaceSuffix = workspace.filter((row) => row.relative.toLowerCase().endsWith(wantedSuffix));
  if (workspaceSuffix.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_workspace_suffix", path: workspaceSuffix[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };

  const registeredRelaxed = registered.filter((row) => relaxedSuffixMatch(row.relative, wantedSuffix));
  if (registeredRelaxed.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_registered_relaxed_suffix", path: registeredRelaxed[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };
  const workspaceRelaxed = workspace.filter((row) => relaxedSuffixMatch(row.relative, wantedSuffix));
  if (workspaceRelaxed.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_workspace_relaxed_suffix", path: workspaceRelaxed[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };

  const wantedBase = path.basename(cleaned).toLowerCase();
  const registeredBasename = registered.filter((row) => path.basename(row.relative).toLowerCase() === wantedBase);
  if (registeredBasename.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_registered_basename", path: registeredBasename[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };
  const workspaceBasename = workspace.filter((row) => path.basename(row.relative).toLowerCase() === wantedBase);
  if (workspaceBasename.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_workspace_basename", path: workspaceBasename[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };
  const wantedStemTokens = stemTokens(cleaned);
  const tokenCandidates = [...new Map([...registered, ...workspace]
    .filter((row) => path.extname(row.relative).toLowerCase() === path.extname(cleaned).toLowerCase())
    .filter((row) => wantedStemTokens.length && wantedStemTokens.every((token) => stemTokens(row.relative).includes(token)))
    .map((row) => [row.absolute, row])).values()];
  if (tokenCandidates.length === 1) return {
    reference, cleaned, status: "resolved", strategy: "unique_basename_token_containment", path: tokenCandidates[0].absolute,
    attempted_paths: attempts.map((row) => row.path),
  };
  const ambiguous = registeredSuffix.length > 1 ? registeredSuffix
    : workspaceSuffix.length > 1 ? workspaceSuffix
      : registeredRelaxed.length > 1 ? registeredRelaxed
        : workspaceRelaxed.length > 1 ? workspaceRelaxed
      : registeredBasename.length > 1 ? registeredBasename
        : workspaceBasename.length > 1 ? workspaceBasename : tokenCandidates;
  const contextual = contextualCandidate(ambiguous, cleaned, contextText, workspaceRoot);
  if (contextual) return {
    reference, cleaned, status: "resolved", strategy: "contextual_unique_candidate", path: contextual.absolute,
    candidate_paths: ambiguous.map((row) => row.absolute), attempted_paths: attempts.map((row) => row.path),
  };
  return {
    reference, cleaned,
    status: ambiguous.length > 1 ? "ambiguous" : "not_found_after_resolution",
    strategy: ambiguous.length > 1 ? "multiple_safe_candidates" : "all_safe_candidates_exhausted",
    candidate_paths: ambiguous.map((row) => row.absolute),
    attempted_paths: attempts.map((row) => row.path),
  };
}

export function resolveDiscoveryReferences(text, options) {
  return extractReportedFileReferences(text).map((reference) => resolveReportedPath(reference, { ...options, contextText: options.contextText ?? text }));
}
