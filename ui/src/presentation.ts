export type TranscriptSource = "microphone" | "typed";
export type SpeakerRole = "user" | "other" | "unknown";

export type TranscriptEvent = {
  id: string;
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
  date?: string;
  rawText?: string;
  at: number;
  score?: number;
  confidence?: number;
  sourceSegmentIds?: string[];
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
    .filter((card) => !card.expiresAt || card.expiresAt > now)
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
