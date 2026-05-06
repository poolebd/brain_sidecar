export type TranscriptSource = "microphone" | "typed";
export type SpeakerRole = "user" | "other" | "unknown";

export type TranscriptEvent = {
  id: string;
  segmentId?: string;
  text: string;
  at: number;
  source: TranscriptSource;
  speakerRole: SpeakerRole;
  speakerLabel?: string;
  speakerConfidence?: number;
  startedAt?: number;
  endedAt?: number;
  isFinal: boolean;
};

export type SidecarCategory =
  | "action"
  | "decision"
  | "question"
  | "risk"
  | "clarification"
  | "contribution"
  | "memory"
  | "work"
  | "work_memory"
  | "web"
  | "status"
  | "note";
export type SidecarPriority = "low" | "normal" | "high";

export type SidecarDisplayCard = {
  id: string;
  category: SidecarCategory;
  title: string;
  summary: string;
  whyRelevant?: string;
  source?: string;
  sourceType?: string;
  sourceLabel?: string;
  qualityLabel?: string;
  date?: string;
  rawText?: string;
  at: number;
  score?: number;
  confidence?: number;
  sourceSegmentIds?: string[];
  evidenceQuote?: string;
  owner?: string;
  dueDate?: string;
  missingInfo?: string;
  cardKey?: string;
  suggestedSay?: string;
  suggestedAsk?: string;
  ephemeral?: boolean;
  expiresAt?: number;
  pinned?: boolean;
  provisional?: boolean;
  explicitlyRequested?: boolean;
  priority?: SidecarPriority;
  sources?: { title: string; url?: string; path?: string }[];
  citations?: string[];
  debugMetadata?: Record<string, unknown>;
};

export type SidecarFilterOptions = {
  maxCards?: number;
  minScore?: number;
  now?: number;
};

export type WorkCardBucket = "actions" | "decisions" | "questions" | "context";

export type WorkCardGroups = Record<WorkCardBucket, SidecarDisplayCard[]>;

export type MeetingOutputBucket = "actions" | "decisions" | "questions" | "risks" | "notes";

export type MeetingOutputGroups = Record<MeetingOutputBucket, SidecarDisplayCard[]>;

export type SidecarQualityMetrics = {
  acceptedCurrentCount: number;
  workMemoryCount: number;
  manualCount: number;
  sourceIdCoverage: number;
  evidenceQuoteCoverage: number;
};

export type LiveFieldRow = {
  id: string;
  at: number;
  transcript?: TranscriptEvent;
  cards: SidecarDisplayCard[];
  kind: "transcript" | "sidecar";
};

const HIGH_CONFIDENCE_USER_SPEAKER = 0.85;
const DEFAULT_MIN_SCORE = 0.65;
const RECENT_TRANSCRIPT_MS = 5 * 60 * 1000;
const DEFAULT_MAX_CARDS = 3;

export function getTranscriptDisplayLabel(event: TranscriptEvent): string {
  if (event.source === "typed") {
    return "You";
  }
  if (event.speakerLabel && (event.speakerConfidence ?? 0) >= HIGH_CONFIDENCE_USER_SPEAKER) {
    return event.speakerLabel;
  }
  if (event.speakerRole === "user" && (event.speakerConfidence ?? 0) >= HIGH_CONFIDENCE_USER_SPEAKER) {
    return "Me";
  }
  return "Mic";
}

export function buildSidecarDisplayCards(
  candidates: SidecarDisplayCard[],
  transcriptEvents: TranscriptEvent[],
  options: SidecarFilterOptions = {},
): SidecarDisplayCard[] {
  const now = options.now ?? Date.now();
  const minScore = options.minScore ?? DEFAULT_MIN_SCORE;
  const maxCards = options.maxCards ?? DEFAULT_MAX_CARDS;
  const recentTranscript = transcriptEvents
    .filter((event) => now - event.at <= RECENT_TRANSCRIPT_MS)
    .map((event) => event.text)
    .join(" ");

  const filtered = candidates
    .filter((card) => card.pinned || !card.expiresAt || card.expiresAt > now)
    .filter((card) => isDisplayWorthy(card, recentTranscript, minScore))
    .sort(compareSidecarCards);

  const deduped: SidecarDisplayCard[] = [];
  for (const card of filtered) {
    if (deduped.some((existing) => areDuplicateCards(existing, card))) {
      continue;
    }
    deduped.push(card);
  }

  if (!Number.isFinite(maxCards)) {
    return deduped;
  }
  return deduped.slice(0, Math.max(0, maxCards));
}

export function groupWorkCards(cards: SidecarDisplayCard[]): WorkCardGroups {
  const groups: WorkCardGroups = {
    actions: [],
    decisions: [],
    questions: [],
    context: [],
  };

  for (const card of cards) {
    groups[workBucketForCard(card)].push(card);
  }

  return groups;
}

export function groupMeetingOutputCards(cards: SidecarDisplayCard[]): MeetingOutputGroups {
  const groups: MeetingOutputGroups = {
    actions: [],
    decisions: [],
    questions: [],
    risks: [],
    notes: [],
  };

  for (const card of cards) {
    if (!isCurrentMeetingCard(card)) {
      continue;
    }
    if (card.category === "action") {
      groups.actions.push(card);
    } else if (card.category === "decision") {
      groups.decisions.push(card);
    } else if (["question", "clarification"].includes(card.category)) {
      groups.questions.push(card);
    } else if (card.category === "risk") {
      groups.risks.push(card);
    } else {
      groups.notes.push(card);
    }
  }

  return groups;
}

export function isCurrentMeetingCard(card: SidecarDisplayCard): boolean {
  const sourceType = card.sourceType ?? String(card.debugMetadata?.sourceType ?? "");
  if (card.explicitlyRequested) {
    return false;
  }
  return sourceType === "transcript" || sourceType === "model_fallback";
}

export function isWorkMemoryCard(card: SidecarDisplayCard): boolean {
  const sourceType = card.sourceType ?? String(card.debugMetadata?.sourceType ?? "");
  return card.category === "work_memory" || sourceType === "work_memory" || sourceType === "work_memory_project";
}

export function isManualCard(card: SidecarDisplayCard): boolean {
  return Boolean(card.explicitlyRequested);
}

export function displaySourceLabel(sourceType: string, explicitlyRequested = false): string {
  const normalized = sourceType.trim().toLowerCase();
  let label = "Local memory";
  if (normalized === "transcript" || normalized === "model_fallback") {
    label = "Current meeting";
  } else if (normalized === "work_memory" || normalized === "work_memory_project") {
    label = "Work memory";
  } else if (normalized === "saved_transcript" || normalized === "session") {
    label = "Past transcript";
  } else if (["local_file", "file", "document_chunk"].includes(normalized)) {
    label = "Local file";
  } else if (normalized === "brave_web" || normalized === "web") {
    label = "Web";
  }
  return explicitlyRequested ? `Manual query / ${label}` : label;
}

export function qualityLabelForCard(card: SidecarDisplayCard): string {
  if (isCurrentMeetingCard(card) && card.evidenceQuote && (card.sourceSegmentIds?.length ?? 0) > 0) {
    return "Evidence-backed";
  }
  if (isWorkMemoryCard(card)) {
    return "Memory context";
  }
  if (card.explicitlyRequested) {
    return "Manual result";
  }
  if (typeof card.confidence === "number") {
    return `${Math.round(card.confidence * 100)}% confidence`;
  }
  return "";
}

export function buildSidecarQualityMetrics(cards: SidecarDisplayCard[]): SidecarQualityMetrics {
  const currentCards = cards.filter(isCurrentMeetingCard);
  return {
    acceptedCurrentCount: currentCards.length,
    workMemoryCount: cards.filter(isWorkMemoryCard).length,
    manualCount: cards.filter(isManualCard).length,
    sourceIdCoverage: percentWith(currentCards, (card) => (card.sourceSegmentIds?.length ?? 0) > 0),
    evidenceQuoteCoverage: percentWith(currentCards, (card) => Boolean(card.evidenceQuote?.trim())),
  };
}

export function buildConsultingBriefMarkdown(cards: SidecarDisplayCard[], title = "Consulting Brief"): string {
  const currentCards = cards.filter((card) => isCurrentMeetingCard(card) && !card.provisional);
  const memoryCards = cards.filter((card) => isWorkMemoryCard(card) && !card.provisional);
  const groups = groupMeetingOutputCards(currentCards);
  const lines: string[] = [`# ${title}`, ""];

  appendBriefSection(lines, "Actions", groups.actions);
  appendBriefSection(lines, "Decisions / Instructions", groups.decisions);
  appendBriefSection(lines, "Open Questions / Clarifications", groups.questions);
  appendBriefSection(lines, "Risks", groups.risks);
  appendBriefSection(lines, "Other Meeting Notes", groups.notes);
  appendBriefSection(lines, "Relevant Work Memory", memoryCards, { memory: true });

  return `${lines.join("\n").trim()}\n`;
}

export function buildLiveFieldRows(
  transcriptEvents: TranscriptEvent[],
  cards: SidecarDisplayCard[],
): LiveFieldRow[] {
  const transcripts = transcriptEvents
    .filter((event) => event.text.trim())
    .sort((left, right) => left.at - right.at);
  const rows = transcripts.map((event): LiveFieldRow => ({
    id: `row:${event.id}`,
    at: event.at,
    transcript: event,
    cards: [],
    kind: "transcript",
  }));
  const rowByTranscriptKey = new Map<string, LiveFieldRow>();

  for (const row of rows) {
    const transcript = row.transcript;
    if (!transcript) {
      continue;
    }
    rowByTranscriptKey.set(transcript.id, row);
    if (transcript.segmentId) {
      rowByTranscriptKey.set(transcript.segmentId, row);
    }
  }

  const sortedCards = [...cards].sort(compareLiveFieldCards);
  const unpairedRows: LiveFieldRow[] = [];
  for (const card of sortedCards) {
    const pairedRow = rowForCard(card, rowByTranscriptKey, transcripts);
    if (pairedRow) {
      pairedRow.cards.push(card);
      continue;
    }
    unpairedRows.push({
      id: `row:sidecar:${card.cardKey ?? card.id}`,
      at: card.at,
      cards: [card],
      kind: "sidecar",
    });
  }

  for (const row of rows) {
    row.cards.sort(compareLiveFieldCards);
  }

  return [...rows, ...unpairedRows].sort(compareLiveFieldRows);
}

export function normalizeDisplayText(text: string): string {
  return text
    .toLowerCase()
    .replace(/&/g, " and ")
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isDisplayWorthy(card: SidecarDisplayCard, recentTranscript: string, minScore: number): boolean {
  if (String(card.debugMetadata?.sourceType ?? "") === "transcript_segment") {
    return false;
  }

  const override = Boolean(card.pinned || card.explicitlyRequested || card.priority === "high");
  const text = `${card.title} ${card.summary} ${card.whyRelevant ?? ""}`;
  const normalized = normalizeDisplayText(text);
  const tokenCount = normalized ? normalized.split(" ").length : 0;

  if (!override && tokenCount < 4) {
    return false;
  }
  if (!override && typeof card.score === "number" && Number.isFinite(card.score) && card.score < minScore) {
    return false;
  }
  if (!override && isTranscriptEcho(text, recentTranscript)) {
    return false;
  }
  return true;
}

function workBucketForCard(card: SidecarDisplayCard): WorkCardBucket {
  const backendLabel = normalizeDisplayText(String(card.debugMetadata?.backendLabel ?? ""));
  if (card.category === "action" || backendLabel === "action") {
    return "actions";
  }
  if (card.category === "decision" || backendLabel === "decision") {
    return "decisions";
  }
  if (["question", "clarification"].includes(card.category) || backendLabel === "question") {
    return "questions";
  }
  return "context";
}

function isTranscriptEcho(candidateText: string, transcriptText: string): boolean {
  const candidate = normalizeDisplayText(candidateText);
  const transcript = normalizeDisplayText(transcriptText);
  if (!candidate || !transcript) {
    return false;
  }

  const candidateTokens = uniqueTokens(candidate);
  const transcriptTokens = uniqueTokens(transcript);
  if (candidateTokens.size < 6 || transcriptTokens.size < 6) {
    return false;
  }

  const tokenOverlap = overlapRatio(candidateTokens, transcriptTokens);
  const trigramOverlap = overlapRatio(new Set(trigrams(candidate)), new Set(trigrams(transcript)));
  const lengthRatio = candidate.length / Math.max(1, transcript.length);
  return tokenOverlap >= 0.78 && trigramOverlap >= 0.58 && lengthRatio >= 0.55 && lengthRatio <= 1.45;
}

function compareSidecarCards(left: SidecarDisplayCard, right: SidecarDisplayCard): number {
  const scoreDelta = cardRank(right) - cardRank(left);
  if (scoreDelta !== 0) {
    return scoreDelta;
  }
  return right.at - left.at;
}

function rowForCard(
  card: SidecarDisplayCard,
  rowByTranscriptKey: Map<string, LiveFieldRow>,
  transcripts: TranscriptEvent[],
): LiveFieldRow | null {
  for (const sourceSegmentId of card.sourceSegmentIds ?? []) {
    const row = rowByTranscriptKey.get(sourceSegmentId);
    if (row) {
      return row;
    }
  }

  if (!card.explicitlyRequested) {
    return null;
  }

  for (let index = transcripts.length - 1; index >= 0; index -= 1) {
    const transcript = transcripts[index];
    if (transcript.source === "typed" && transcript.at <= card.at + 2_000) {
      return rowByTranscriptKey.get(transcript.id) ?? null;
    }
  }
  return null;
}

function compareLiveFieldRows(left: LiveFieldRow, right: LiveFieldRow): number {
  const atDelta = left.at - right.at;
  if (atDelta !== 0) {
    return atDelta;
  }
  if (left.kind !== right.kind) {
    return left.kind === "transcript" ? -1 : 1;
  }
  return left.id.localeCompare(right.id);
}

function compareLiveFieldCards(left: SidecarDisplayCard, right: SidecarDisplayCard): number {
  const rankDelta = cardRank(right) - cardRank(left);
  if (rankDelta !== 0) {
    return rankDelta;
  }
  const atDelta = left.at - right.at;
  if (atDelta !== 0) {
    return atDelta;
  }
  return left.id.localeCompare(right.id);
}

function cardRank(card: SidecarDisplayCard): number {
  const override = (card.pinned ? 500 : 0)
    + (card.explicitlyRequested ? 140 : 0)
    + (card.priority === "high" ? 120 : card.priority === "low" ? -40 : 0);
  const category = card.category === "note"
    ? 42
    : card.category === "web"
      ? 36
      : card.category === "work" || card.category === "work_memory"
        ? 32
        : ["action", "decision", "risk", "clarification", "contribution"].includes(card.category)
          ? 44
          : 20;
  const relevance = typeof card.score === "number" && Number.isFinite(card.score)
    ? card.score * 100
    : 70;
  return override + category + relevance;
}

function percentWith(cards: SidecarDisplayCard[], predicate: (card: SidecarDisplayCard) => boolean): number {
  if (cards.length === 0) {
    return 1;
  }
  return Math.round((cards.filter(predicate).length / cards.length) * 1000) / 1000;
}

function appendBriefSection(
  lines: string[],
  heading: string,
  cards: SidecarDisplayCard[],
  options: { memory?: boolean } = {},
): void {
  lines.push(`## ${heading}`, "");
  if (cards.length === 0) {
    lines.push("- None", "");
    return;
  }
  for (const card of cards) {
    lines.push(`- **${card.title}**: ${card.summary}`);
    const details = [
      card.owner ? `Owner: ${card.owner}` : "",
      card.dueDate ? `Due: ${card.dueDate}` : "",
      card.missingInfo ? `Missing: ${card.missingInfo}` : "",
    ].filter(Boolean);
    if (details.length > 0) {
      lines.push(`  - ${details.join(" · ")}`);
    }
    if (!options.memory && card.evidenceQuote) {
      lines.push(`  - Evidence: "${card.evidenceQuote}"`);
    }
    if (!options.memory && (card.sourceSegmentIds?.length ?? 0) > 0) {
      lines.push(`  - Source segments: ${card.sourceSegmentIds!.join(", ")}`);
    }
    if (options.memory && (card.citations?.length ?? 0) > 0) {
      lines.push(`  - Citations: ${card.citations!.join("; ")}`);
    }
  }
  lines.push("");
}

function areDuplicateCards(left: SidecarDisplayCard, right: SidecarDisplayCard): boolean {
  if (left.cardKey && right.cardKey && left.cardKey === right.cardKey) {
    return true;
  }
  const leftKey = normalizeDisplayText(`${left.title} ${left.summary}`);
  const rightKey = normalizeDisplayText(`${right.title} ${right.summary}`);
  if (!leftKey || !rightKey) {
    return false;
  }
  if (leftKey === rightKey) {
    return true;
  }
  const leftTokens = uniqueTokens(leftKey);
  const rightTokens = uniqueTokens(rightKey);
  return overlapRatio(leftTokens, rightTokens) >= 0.92;
}

function uniqueTokens(text: string): Set<string> {
  return new Set(text.split(" ").filter((token) => token.length > 1));
}

function trigrams(text: string): string[] {
  const compact = text.replace(/\s+/g, " ");
  if (compact.length <= 3) {
    return compact ? [compact] : [];
  }
  const result: string[] = [];
  for (let index = 0; index <= compact.length - 3; index += 1) {
    result.push(compact.slice(index, index + 3));
  }
  return result;
}

function overlapRatio(left: Set<string>, right: Set<string>): number {
  if (left.size === 0 || right.size === 0) {
    return 0;
  }
  let shared = 0;
  for (const item of left) {
    if (right.has(item)) {
      shared += 1;
    }
  }
  return shared / Math.min(left.size, right.size);
}
