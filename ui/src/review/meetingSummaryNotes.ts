import type { SidecarDisplayCard } from "../presentation";
import type {
  ReviewJobResponse,
  SavedTranscriptSegment,
  SessionMemorySummary,
  SummaryProjectWorkstream,
  SummaryTechnicalFinding,
} from "../types";

export type SummaryNoteKind = "key_note" | "decision" | "action" | "open_question" | "risk";

export type SummaryNoteItem = {
  id: string;
  kind: SummaryNoteKind;
  text: string;
  sourceSegmentIds: string[];
  confidence?: number;
};

export type MeetingSummaryNotesViewModel = {
  title: string;
  summaryParagraph: string;
  keyNotes: SummaryNoteItem[];
  decisions: SummaryNoteItem[];
  actions: SummaryNoteItem[];
  openQuestions: SummaryNoteItem[];
  risks: SummaryNoteItem[];
  sourceSegmentIds: string[];
  transcriptSegments: SavedTranscriptSegment[];
  warnings: string[];
};

type Candidate = SummaryNoteItem & {
  order: number;
  strength: number;
};

type CandidateGroups = Record<SummaryNoteKind, Candidate[]>;

const STALE_CONTEXT_TERMS = ["PGE", "Westwood", "Energy Lens", "Energy consulting"];

const META_NOTE_PATTERNS = [
  /\b\d+\s+(?:follow-?ups?|actions?|decisions?|open questions?|questions?|risks?|blockers?|technical findings?)\b.*\b(?:captured|confirmation|grounded|watch)\b/i,
  /\b(?:source coverage|source count|review output|meeting output|supporting cards?|validation evidence)\b/i,
  /\bthe (?:meeting|review|call) (?:covered|centered on|focused on|was about)\b/i,
  /\breview completed, but\b/i,
  /\bdid not pass the usefulness gate\b/i,
  /\bvalidate the clean transcript\b/i,
  /\btechnical findings? grounded in the transcript\b/i,
];

const WEAK_SUMMARY_PATTERNS = [
  /\breview completed, but\b/i,
  /\bdid not pass the usefulness gate\b/i,
  /\bvalidate the clean transcript\b/i,
  /\bthe (?:meeting|review|call) (?:covered|centered on|focused on|was about)\b/i,
];

const STOPWORDS = new Set([
  "a",
  "an",
  "and",
  "are",
  "as",
  "at",
  "be",
  "by",
  "for",
  "from",
  "has",
  "have",
  "in",
  "into",
  "is",
  "it",
  "of",
  "on",
  "or",
  "the",
  "this",
  "to",
  "with",
]);

export function buildMeetingSummaryNotesViewModel(
  job: ReviewJobResponse,
  cards: SidecarDisplayCard[] = [],
): MeetingSummaryNotesViewModel {
  const summary = job.summary;
  const transcriptSegments = job.clean_segments ?? [];
  const supportText = supportTextForJob(job);
  const groups: CandidateGroups = {
    key_note: [],
    decision: [],
    action: [],
    open_question: [],
    risk: [],
  };
  let order = 0;

  const push = (
    kind: SummaryNoteKind,
    textInput: unknown,
    options: { sourceSegmentIds?: string[]; confidence?: number; strength?: number } = {},
  ) => {
    let text = cleanNoteText(textInput, supportText);
    if (!text || shouldSuppressNote(text)) {
      return;
    }
    let targetKind = kind;
    if (looksLikeLogisticsFollowUp(text)) {
      targetKind = "action";
    } else if (targetKind === "risk" && looksLikeQuestionOrUncertainty(text) && !looksLikeConcern(text)) {
      targetKind = "open_question";
    } else if (targetKind === "open_question" && looksLikeConcern(text) && !looksLikeQuestionOrUncertainty(text)) {
      targetKind = "risk";
    }
    const sourceSegmentIds = distinctStrings(options.sourceSegmentIds ?? []);
    groups[targetKind].push({
      id: `${targetKind}-${order}`,
      kind: targetKind,
      text,
      sourceSegmentIds,
      confidence: normalizedConfidence(options.confidence),
      order: order++,
      strength: options.strength ?? 20,
    });
  };

  if (summary) {
    addSummaryCandidates(summary, push);
    addWorkstreamCandidates(summary, push, supportText);
    addTechnicalFindingCandidates(summary, push);
  }
  addCardCandidates(cards, push);

  const decisions = mergeCandidates(groups.decision);
  const actions = mergeCandidates(groups.action)
    .filter((item) => !noteMatchesAny(item, decisions));
  const openQuestions = mergeCandidates(groups.open_question)
    .filter((item) => !noteMatchesAny(item, [...decisions, ...actions]));
  const risks = mergeCandidates(groups.risk)
    .filter((item) => !noteMatchesAny(item, [...decisions, ...actions, ...openQuestions]));
  const primary = [...decisions, ...actions, ...openQuestions, ...risks];
  const keyNotes = buildKeyNotes(mergeCandidates(groups.key_note), primary);
  const summaryParagraph = buildSummaryParagraph(summary, [...keyNotes, ...primary], supportText);
  const sourceSegmentIds = distinctStrings([
    ...keyNotes,
    ...decisions,
    ...actions,
    ...openQuestions,
    ...risks,
  ].flatMap((item) => item.sourceSegmentIds));
  const warnings = summaryWarnings(job);

  return {
    title: cleanNoteTitle(summary?.title || job.title || "Meeting Summary", supportText) || "Meeting Summary",
    summaryParagraph,
    keyNotes,
    decisions,
    actions,
    openQuestions,
    risks,
    sourceSegmentIds,
    transcriptSegments,
    warnings,
  };
}

export function buildMeetingSummaryMarkdown(viewModel: MeetingSummaryNotesViewModel): string {
  const lines = ["# Meeting Summary", ""];
  if (viewModel.title && viewModel.title !== "Meeting Summary") {
    lines.push(`_${viewModel.title}_`, "");
  }
  lines.push("## Summary", viewModel.summaryParagraph || "No summary was captured.", "");
  appendMarkdownItems(lines, "Key Notes", viewModel.keyNotes);
  appendMarkdownItems(lines, "Decisions", viewModel.decisions);
  appendMarkdownItems(lines, "Action Items / Follow-ups", viewModel.actions, "None found.");
  appendMarkdownItems(lines, "Open Questions", viewModel.openQuestions);
  appendMarkdownItems(lines, "Risks / Concerns", viewModel.risks);
  return `${lines.join("\n").replace(/\n{3,}/g, "\n\n").trim()}\n`;
}

function addSummaryCandidates(
  summary: SessionMemorySummary,
  push: (kind: SummaryNoteKind, textInput: unknown, options?: { sourceSegmentIds?: string[]; confidence?: number; strength?: number }) => void,
) {
  const summarySourceIds = summaryList(summary.source_segment_ids);
  for (const decision of summaryList(summary.decisions)) {
    push("decision", decision, { sourceSegmentIds: summarySourceIds, strength: 62 });
  }
  for (const action of summaryList(summary.actions)) {
    push("action", action, { sourceSegmentIds: summarySourceIds, strength: 62 });
  }
  for (const question of summaryList(summary.unresolved_questions)) {
    push("open_question", question, { sourceSegmentIds: summarySourceIds, strength: 54 });
  }
  for (const risk of summaryList(summary.risks)) {
    push("risk", risk, { sourceSegmentIds: summarySourceIds, strength: 52 });
  }
  for (const keyPoint of summaryList(summary.key_points)) {
    push("key_note", keyPoint, { sourceSegmentIds: summarySourceIds, strength: 34 });
  }
  for (const action of summaryList(summary.portfolio_rollup?.bp_next_actions)) {
    push("action", action, { sourceSegmentIds: summaryList(summary.portfolio_rollup?.source_segment_ids), strength: 48 });
  }
  for (const loop of summaryList(summary.portfolio_rollup?.open_loops)) {
    push(looksLikeConcern(loop) ? "risk" : "open_question", loop, {
      sourceSegmentIds: summaryList(summary.portfolio_rollup?.source_segment_ids),
      strength: 36,
    });
  }
}

function addWorkstreamCandidates(
  summary: SessionMemorySummary,
  push: (kind: SummaryNoteKind, textInput: unknown, options?: { sourceSegmentIds?: string[]; confidence?: number; strength?: number }) => void,
  supportText: string,
) {
  for (const workstream of summaryProjectWorkstreamList(summary.project_workstreams)) {
    const sourceSegmentIds = summaryList(workstream.source_segment_ids);
    const project = cleanNoteTitle(workstream.project, supportText);
    for (const decision of summaryList(workstream.decisions)) {
      push("decision", withUsefulProjectContext(decision, project), { sourceSegmentIds, strength: 72 });
    }
    for (const action of summaryList(workstream.actions)) {
      push("action", withUsefulProjectContext(action, project), { sourceSegmentIds, strength: 74 });
    }
    for (const question of summaryList(workstream.open_questions)) {
      push("open_question", withUsefulProjectContext(question, project), { sourceSegmentIds, strength: 66 });
    }
    for (const risk of summaryList(workstream.risks)) {
      push("risk", withUsefulProjectContext(risk, project), { sourceSegmentIds, strength: 64 });
    }
    const hasDetail = [
      workstream.decisions,
      workstream.actions,
      workstream.open_questions,
      workstream.risks,
    ].some((value) => summaryList(value).length > 0);
    if (!hasDetail && supportedTopicLabel(project, supportText)) {
      push("key_note", project, { sourceSegmentIds, strength: 28 });
    }
  }
}

function withUsefulProjectContext(textInput: unknown, project: string): string {
  const text = cleanText(textInput);
  const label = cleanText(project).replace(/\.$/, "");
  if (!text || !label || !usefulTopicLabel(label)) {
    return text;
  }
  const textKey = normalizeNoteKey(text);
  const labelKey = normalizeNoteKey(label);
  if (!labelKey || textKey.includes(labelKey)) {
    return text;
  }
  return `${label}: ${text}`;
}

function addTechnicalFindingCandidates(
  summary: SessionMemorySummary,
  push: (kind: SummaryNoteKind, textInput: unknown, options?: { sourceSegmentIds?: string[]; confidence?: number; strength?: number }) => void,
) {
  for (const finding of summaryTechnicalFindingList(summary.technical_findings)) {
    const sourceSegmentIds = summaryList(finding.source_segment_ids);
    for (const risk of summaryList(finding.risks)) {
      push("risk", risk, { sourceSegmentIds, confidence: confidenceScore(finding.confidence), strength: 34 });
    }
    for (const gap of summaryList(finding.data_gaps)) {
      push("open_question", gap, { sourceSegmentIds, confidence: confidenceScore(finding.confidence), strength: 34 });
    }
    for (const recommendation of summaryList(finding.recommendations)) {
      push("action", recommendation, { sourceSegmentIds, confidence: confidenceScore(finding.confidence), strength: 36 });
    }
    for (const findingText of summaryList(finding.findings)) {
      if (looksLikeLogisticsFollowUp(findingText)) {
        push("action", findingText, { sourceSegmentIds, confidence: confidenceScore(finding.confidence), strength: 42 });
      } else if (summaryList(finding.methods).length > 0 || summaryList(finding.assumptions).length > 0) {
        push("key_note", findingText, { sourceSegmentIds, confidence: confidenceScore(finding.confidence), strength: 24 });
      }
    }
  }
}

function addCardCandidates(
  cards: SidecarDisplayCard[],
  push: (kind: SummaryNoteKind, textInput: unknown, options?: { sourceSegmentIds?: string[]; confidence?: number; strength?: number }) => void,
) {
  for (const card of cards) {
    const text = card.summary || card.title;
    const common = {
      sourceSegmentIds: distinctStrings(card.sourceSegmentIds ?? []),
      confidence: card.confidence,
    };
    switch (card.category) {
      case "action":
        push("action", text, { ...common, strength: 82 });
        break;
      case "decision":
        push("decision", text, { ...common, strength: 80 });
        break;
      case "question":
      case "clarification":
        push("open_question", text, { ...common, strength: 76 });
        break;
      case "risk":
        push("risk", text, { ...common, strength: 76 });
        break;
      case "note":
      case "contribution":
        push("key_note", text, { ...common, strength: 32 });
        break;
      default:
        break;
    }
  }
}

function buildSummaryParagraph(
  summary: SessionMemorySummary | null,
  items: SummaryNoteItem[],
  supportText: string,
): string {
  const summaryText = cleanNoteText(summary?.summary, supportText);
  if (summaryText && !WEAK_SUMMARY_PATTERNS.some((pattern) => pattern.test(summaryText))) {
    return summaryText;
  }
  const topics = distinctStrings(items.map((item) => topicPhrase(item.text)).filter(Boolean)).slice(0, 5);
  if (topics.length > 0) {
    return `Key topics were ${joinHumanList(topics)}.`;
  }
  return "No concise meeting summary was captured.";
}

function topicPhrase(text: string): string {
  const value = cleanText(text);
  const lower = value.toLowerCase();
  if (/\brfi\b/i.test(value)) {
    return "RFI log handoff";
  }
  if (/\bsiemens\b/i.test(value)) {
    return "Siemens document review";
  }
  if (/\bcache-control\b/i.test(value)) {
    return "Cache-Control directive guidance";
  }
  if (/\b(?:dns|cname|apex domain)\b/i.test(value)) {
    return "DNS/CNAME handling for apex domains";
  }
  if (/\bgithub\b/i.test(value)) {
    return "GitHub issue status";
  }
  if (/\bjabber\b/i.test(value)) {
    return "Jabber relay logistics";
  }
  if (/\b(?:browser|browsers|ros developers?|impediments?)\b/i.test(value)) {
    return "browser deployment impediments";
  }
  const prefix = value.split(":")[0]?.trim() ?? "";
  const candidate = prefix && prefix.length < 80 ? prefix : value.split(/[.;]/)[0]?.trim() ?? value;
  return sentenceCase(candidate.replace(/\.$/, ""));
}

function buildKeyNotes(baseNotes: SummaryNoteItem[], primaryItems: SummaryNoteItem[]): SummaryNoteItem[] {
  const notes = baseNotes
    .filter((item) => !noteMatchesAny(item, primaryItems))
    .slice(0, 8);
  if (notes.length >= 4) {
    return notes;
  }
  const existing = [...notes];
  for (const item of primaryItems) {
    const phrase = topicPhrase(item.text);
    if (!phrase || existing.some((candidate) => normalizeNoteKey(candidate.text) === normalizeNoteKey(phrase))) {
      continue;
    }
    existing.push({
      id: stableItemId({ ...item, kind: "key_note", text: phrase, sourceSegmentIds: item.sourceSegmentIds }),
      kind: "key_note",
      text: phrase.endsWith(".") ? phrase : `${phrase}.`,
      sourceSegmentIds: item.sourceSegmentIds,
      confidence: item.confidence,
    });
    if (existing.length >= 4) {
      break;
    }
  }
  return existing.slice(0, 8);
}

function summaryWarnings(job: ReviewJobResponse): string[] {
  const diagnostics = job.summary?.diagnostics ?? job.diagnostics ?? {};
  const flags = Array.isArray(diagnostics.usefulness_flags) ? diagnostics.usefulness_flags : [];
  const qualityFlags = Array.isArray(diagnostics.quality_flags) ? diagnostics.quality_flags : [];
  return distinctStrings([...flags, ...qualityFlags].map(cleanText).filter(Boolean));
}

function mergeCandidates(candidates: Candidate[]): SummaryNoteItem[] {
  const merged: Candidate[] = [];
  for (const candidate of candidates.sort((left, right) => left.order - right.order)) {
    const existingIndex = merged.findIndex((item) => noteItemsMatch(item, candidate));
    if (existingIndex === -1) {
      merged.push(candidate);
      continue;
    }
    merged[existingIndex] = mergeCandidate(merged[existingIndex], candidate);
  }
  return merged
    .sort((left, right) => left.order - right.order)
    .map(({ order: _order, strength: _strength, ...item }) => ({
      ...item,
      id: stableItemId(item),
    }));
}

function mergeCandidate(left: Candidate, right: Candidate): Candidate {
  const sourceSegmentIds = distinctStrings([...left.sourceSegmentIds, ...right.sourceSegmentIds]);
  const stronger = right.strength > left.strength ? right : left;
  const weaker = stronger === right ? left : right;
  return {
    ...stronger,
    text: bestNoteText(stronger.text, weaker.text),
    sourceSegmentIds,
    confidence: stronger.confidence ?? weaker.confidence,
    order: Math.min(left.order, right.order),
    strength: Math.max(left.strength, right.strength),
  };
}

function bestNoteText(left: string, right: string): string {
  const leftScore = noteTextScore(left);
  const rightScore = noteTextScore(right);
  return rightScore > leftScore ? right : left;
}

function noteTextScore(text: string): number {
  const clean = cleanText(text);
  let score = Math.min(clean.length, 240);
  if (/[.!?]$/.test(clean)) score += 20;
  if (clean.length > 260) score -= 30;
  if (META_NOTE_PATTERNS.some((pattern) => pattern.test(clean))) score -= 100;
  return score;
}

function noteMatchesAny(item: SummaryNoteItem, existing: SummaryNoteItem[]): boolean {
  return existing.some((candidate) => noteItemsMatch(item, candidate));
}

function noteItemsMatch(left: SummaryNoteItem, right: SummaryNoteItem): boolean {
  const leftKey = normalizeNoteKey(left.text);
  const rightKey = normalizeNoteKey(right.text);
  if (!leftKey || !rightKey) {
    return false;
  }
  if (leftKey === rightKey || leftKey.includes(rightKey) || rightKey.includes(leftKey)) {
    return true;
  }
  if ((left.kind === "open_question" || right.kind === "open_question") && left.kind !== right.kind) {
    return false;
  }
  if (leftKey.includes("private") !== rightKey.includes("private")) {
    return false;
  }
  if (left.kind !== right.kind) {
    return false;
  }
  const sourceOverlap = left.sourceSegmentIds.some((id) => right.sourceSegmentIds.includes(id));
  return tokenOverlap(leftKey, rightKey) >= (sourceOverlap ? 0.36 : 0.58);
}

function cleanNoteTitle(value: unknown, supportText: string): string {
  let text = applyDomainCleanup(cleanText(value));
  if (!text) {
    return "";
  }
  text = stripUnsupportedContextPrefix(text, supportText);
  text = stripDanglingPrefix(text);
  if (containsUnsupportedStaleTerm(text, supportText)) {
    return "";
  }
  return text;
}

function cleanNoteText(value: unknown, supportText: string): string {
  let text = applyDomainCleanup(cleanText(value));
  if (!text) {
    return "";
  }
  text = stripUnsupportedContextPrefix(text, supportText);
  text = stripDanglingPrefix(text);
  text = finishSentence(text);
  if (containsUnsupportedStaleTerm(text, supportText)) {
    return "";
  }
  return text;
}

function applyDomainCleanup(text: string): string {
  return text
    .replace(/\bRFI[-\s]+log\b/gi, "RFI log")
    .replace(/\bcash control private\b/gi, "Cache-Control `private`")
    .replace(/\bcash control\b/gi, "Cache-Control")
    .replace(/\bc\s*[- ]?\s*name\b/gi, "CNAME")
    .replace(/\bCNAME records?\b/gi, "CNAME")
    .replace(/\bHTTP\s+BIS\b/gi, "HTTPBIS")
    .replace(/\bROS developers?\b/g, "browser developers")
    .replace(/\bgeti\s*hub\b/gi, "GitHub")
    .replace(/\bgithub\b/gi, "GitHub")
    .replace(/\bietf\b/gi, "IETF")
    .replace(/\bdns\b/gi, "DNS")
    .replace(/\bjabber\b/gi, "Jabber")
    .replace(/\bapex domain\b/gi, "apex domain");
}

function stripUnsupportedContextPrefix(text: string, supportText: string): string {
  const match = text.match(/^([^:]{2,110}):\s*(.+)$/);
  if (!match) {
    return text;
  }
  const [, prefix, body] = match;
  if (containsUnsupportedStaleTerm(prefix, supportText)) {
    return body.trim();
  }
  return text;
}

function stripDanglingPrefix(text: string): string {
  const trimmed = text.trim();
  const dangling = trimmed.match(/^([^:]{2,90}):\s*$/);
  if (dangling) {
    const label = dangling[1].trim();
    return usefulTopicLabel(label) ? label : "";
  }
  return trimmed.replace(/:\s*$/, "");
}

function usefulTopicLabel(text: string): boolean {
  return /\b(?:RFI|Siemens|Cache-Control|CNAME|DNS|GitHub|Jabber|HTTPBIS|browser|apex domain)\b/i.test(text);
}

function finishSentence(text: string): string {
  let value = cleanText(text)
    .replace(/\s+(?:he|she|they|it|roy|the|a|an|with|between|older|recently|including)$/i, "")
    .replace(/\s+(?:in|to|for|of|with|between|across|from|despite)$/i, "")
    .replace(/\s+,\s*$/, "")
    .trim();
  if (!value) {
    return "";
  }
  if (!/[.!?`]$/.test(value)) {
    value += ".";
  }
  return value;
}

function shouldSuppressNote(text: string): boolean {
  const value = cleanText(text);
  if (!value || normalizeNoteKey(value).length < 4) {
    return true;
  }
  return META_NOTE_PATTERNS.some((pattern) => pattern.test(value));
}

function supportedTopicLabel(text: string, supportText: string): boolean {
  const key = normalizeNoteKey(text);
  if (!key || containsUnsupportedStaleTerm(text, supportText)) {
    return false;
  }
  return key.split(/\s+/).some((token) => token.length > 3 && supportText.includes(token));
}

function containsUnsupportedStaleTerm(text: string, supportText: string): boolean {
  return STALE_CONTEXT_TERMS.some((term) => (
    new RegExp(`\\b${escapeRegExp(term)}\\b`, "i").test(text)
    && !supportText.includes(normalizeNoteKey(term))
  ));
}

function looksLikeLogisticsFollowUp(text: unknown): boolean {
  const value = cleanText(text);
  return /\b(?:volunteer|relay|Jabber relay|someone was requested|step forward|need(?:ed)? to serve)\b/i.test(value);
}

function looksLikeQuestionOrUncertainty(text: unknown): boolean {
  const value = cleanText(text);
  return value.endsWith("?") || /\b(?:whether|clarify|confirm|question|considering|should|unknown|unresolved)\b/i.test(value);
}

function looksLikeConcern(text: unknown): boolean {
  const value = cleanText(text);
  return /\b(?:risk|blocker|blocked|blocking|concern|impediments?|challenge|depends?|dependency|compatibility issues?)\b/i.test(value);
}

function supportTextForJob(job: ReviewJobResponse): string {
  return normalizeNoteKey([
    job.title,
    ...(job.clean_segments ?? []).map((segment) => segment.text),
  ].join(" "));
}

function normalizeNoteKey(value: unknown): string {
  return applyDomainCleanup(cleanText(value))
    .toLowerCase()
    .replace(/[`'"]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((token) => token && !STOPWORDS.has(token))
    .join(" ")
    .trim();
}

function tokenOverlap(leftKey: string, rightKey: string): number {
  const left = new Set(leftKey.split(/\s+/).filter(Boolean));
  const right = new Set(rightKey.split(/\s+/).filter(Boolean));
  if (left.size === 0 || right.size === 0) {
    return 0;
  }
  let matches = 0;
  for (const token of left) {
    if (right.has(token)) {
      matches += 1;
    }
  }
  return matches / Math.min(left.size, right.size);
}

function stableItemId(item: SummaryNoteItem): string {
  const key = normalizeNoteKey(`${item.kind} ${item.text}`).slice(0, 80).replace(/\s+/g, "-") || "item";
  const source = item.sourceSegmentIds.slice().sort().join("-").replace(/[^a-zA-Z0-9-]+/g, "-");
  return `${item.kind}-${key}${source ? `-${source}` : ""}`;
}

function appendMarkdownItems(
  lines: string[],
  title: string,
  items: SummaryNoteItem[],
  empty = "None",
) {
  lines.push(`## ${title}`);
  if (items.length === 0) {
    lines.push(`- ${empty}`, "");
    return;
  }
  for (const item of items) {
    lines.push(`- ${item.text}`);
  }
  lines.push("");
}

function joinHumanList(items: string[]): string {
  if (items.length <= 1) {
    return items[0] ?? "";
  }
  if (items.length === 2) {
    return `${items[0]} and ${items[1]}`;
  }
  return `${items.slice(0, -1).join(", ")}, and ${items[items.length - 1]}`;
}

function summaryList(value: unknown): string[] {
  return Array.isArray(value) ? distinctStrings(value.map(cleanText).filter(Boolean)) : [];
}

function summaryProjectWorkstreamList(value: unknown): SummaryProjectWorkstream[] {
  return Array.isArray(value) ? value.filter((item): item is SummaryProjectWorkstream => Boolean(item && typeof item === "object")) : [];
}

function summaryTechnicalFindingList(value: unknown): SummaryTechnicalFinding[] {
  return Array.isArray(value) ? value.filter((item): item is SummaryTechnicalFinding => Boolean(item && typeof item === "object")) : [];
}

function distinctStrings(values: string[]): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    const text = cleanText(value);
    const key = text.toLowerCase();
    if (!text || seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(text);
  }
  return result;
}

function normalizedConfidence(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  return Math.max(0, Math.min(1, value > 1 ? value / 100 : value));
}

function confidenceScore(value: unknown): number | undefined {
  const text = cleanText(value).toLowerCase();
  if (text === "high") return 0.9;
  if (text === "medium") return 0.74;
  if (text === "low") return 0.45;
  const numeric = Number(text);
  return Number.isFinite(numeric) ? normalizedConfidence(numeric) : undefined;
}

function cleanText(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value).replace(/\s+/g, " ").trim();
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["text", "description", "body", "summary", "title", "value", "item"]) {
      const text = cleanText(record[key]);
      if (text) {
        return text;
      }
    }
  }
  return "";
}

function sentenceCase(value: string): string {
  const text = cleanText(value);
  return text ? `${text[0].toUpperCase()}${text.slice(1)}` : "";
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
