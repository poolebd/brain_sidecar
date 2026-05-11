export type TranscriptSource = "microphone" | "typed";
export type SpeakerRole = "user" | "other" | "unknown";

export type TranscriptEvent = {
  id: string;
  segmentId?: string;
  replacesSegmentId?: string;
  sourceSegmentIds?: string[];
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
  | "reference"
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

export type LiveSignalKind =
  | "thread"
  | "attention"
  | "question"
  | "action"
  | "risk"
  | "decision"
  | "suggestion";

export type LiveSignal = {
  id: string;
  kind: LiveSignalKind;
  title: string;
  body: string;
  evidenceQuote: string;
  sourceEventIds: string[];
  sourceSegmentIds: string[];
  priority: SidecarPriority;
  provisional: boolean;
  at: number;
  suggestedSay?: string;
  suggestedAsk?: string;
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
  contextCount: number;
  manualCount: number;
  sourceIdCoverage: number;
  evidenceQuoteCoverage: number;
};

export type MeetingContractBrief = {
  goal: string;
  mode: "quiet" | "balanced" | "assertive";
  reminders: string[];
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
const DEFAULT_LIVE_SIGNAL_WINDOW_MS = 90_000;
const DEFAULT_MAX_LIVE_SIGNALS = 5;
const LIVE_SIGNAL_EVIDENCE_CHARS = 220;
const DEFAULT_ADDRESSED_TO_NAMES = ["Brandon", "Brandon Poole", "BP"];
const QUESTION_SIGNAL_RE = /(\?|(?:\bcan\s+we\b|\bshould\s+we\b|\bdo\s+we\b|\bare\s+we\b|\bwho\s+owns\b|\bwhat\b|\bwhen\b|\bwhere\b|\bwhy\b|\bhow\b|\bclarify\b|\bconfirm\b))/i;
const ACTION_SIGNAL_RE = /\b(?:please|need\s+to|needs\s+to|will|i'?ll|i\s+will|we\s+should|send|review|confirm|follow\s+up|schedule|own(?:s|ed|er|ership)?)\b/i;
const RISK_SIGNAL_RE = /\b(?:risk|blocked|blocking|dependency|depends\s+on|waiting\s+on|deadline|constraint|issue|concern|assumption)\b/i;
const DECISION_SIGNAL_RE = /\b(?:decided|decision|agreed|approved|settled|sign-?off|we\s+will\s+proceed)\b/i;
const DIRECT_ADDRESS_PREFIX_RE = "(?:hey|hi|hello|ok|okay|alright|all\\s+right)";
const DIRECT_ADDRESS_REQUEST_RE = "(?:can|could|would|will\\s+you|please|do\\s+you|are\\s+you|have\\s+you|should\\s+you|need\\s+you|confirm|clarify|review|send|own|take|answer|handle|check|follow\\s+up)";
const DIRECT_ADDRESS_THIRD_PERSON_RE = "(?:is|was|has|had|goes|said|says)\\b";

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

export function isManualCard(card: SidecarDisplayCard): boolean {
  return Boolean(card.explicitlyRequested);
}

export function isCompanyReferenceCard(card: SidecarDisplayCard): boolean {
  return card.sourceType === "company_ref" || card.category === "reference";
}

export function displaySourceLabel(sourceType: string, explicitlyRequested = false): string {
  const normalized = sourceType.trim().toLowerCase();
  let label = "Local memory";
  if (normalized === "transcript" || normalized === "model_fallback") {
    label = "Current meeting";
  } else if (normalized === "saved_transcript" || normalized === "session") {
    label = "Past transcript";
  } else if (["local_file", "file", "document_chunk"].includes(normalized)) {
    label = "Technical reference";
  } else if (normalized === "company_ref") {
    label = "Company reference";
  } else if (normalized === "brave_web" || normalized === "web") {
    label = "Web";
  }
  return explicitlyRequested ? `Manual query / ${label}` : label;
}

export function qualityLabelForCard(card: SidecarDisplayCard): string {
  if (isCompanyReferenceCard(card)) {
    return "Reference context";
  }
  if (isCurrentMeetingCard(card) && card.evidenceQuote && (card.sourceSegmentIds?.length ?? 0) > 0) {
    return "Evidence-backed";
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
    contextCount: cards.filter((card) => !isCurrentMeetingCard(card) && !isManualCard(card)).length,
    manualCount: cards.filter(isManualCard).length,
    sourceIdCoverage: percentWith(currentCards, (card) => (card.sourceSegmentIds?.length ?? 0) > 0),
    evidenceQuoteCoverage: percentWith(currentCards, (card) => Boolean(card.evidenceQuote?.trim())),
  };
}

export function buildConsultingBriefMarkdown(
  cards: SidecarDisplayCard[],
  title = "Consulting Brief",
  contract?: MeetingContractBrief | null,
): string {
  const currentCards = cards.filter((card) => isCurrentMeetingCard(card) && !card.provisional);
  const contextCards = cards.filter((card) => (
    !isCurrentMeetingCard(card)
    && !isManualCard(card)
    && !card.provisional
  ));
  const manualCards = cards.filter((card) => isManualCard(card) && !card.provisional);
  const groups = groupMeetingOutputCards(currentCards);
  const lines: string[] = [`# ${title}`, ""];

  appendContractSection(lines, contract);
  appendBriefSection(lines, "Actions", groups.actions);
  appendBriefSection(lines, "Decisions / Instructions", groups.decisions);
  appendBriefSection(lines, "Open Questions / Clarifications", groups.questions);
  appendBriefSection(lines, "Risks", groups.risks);
  appendSuggestedLanguageSection(lines, currentCards);
  appendBriefSection(lines, "Other Meeting Notes", groups.notes);
  appendBriefSection(lines, "Technical References / Web Context", contextCards, { memory: true });
  appendBriefSection(lines, "Manual Query Results", manualCards, { memory: true });
  appendEvidenceIndex(lines, currentCards);

  return `${lines.join("\n").trim()}\n`;
}

export function buildLiveSignals(
  transcriptEvents: TranscriptEvent[],
  cards: SidecarDisplayCard[],
  options: {
    now?: number;
    windowMs?: number;
    maxSignals?: number;
    addressedToNames?: string[];
  } = {},
): LiveSignal[] {
  const now = options.now ?? Date.now();
  const windowMs = options.windowMs ?? DEFAULT_LIVE_SIGNAL_WINDOW_MS;
  const maxSignals = Math.max(0, options.maxSignals ?? DEFAULT_MAX_LIVE_SIGNALS);
  const addressedToNames = addressedToNamesForSignals(options.addressedToNames);
  const recentEvents = transcriptEvents
    .filter((event) => event.text.trim() && now - event.at <= windowMs)
    .sort((left, right) => left.at - right.at);
  const finalEvents = recentEvents.filter((event) => event.isFinal);
  const partialEvents = recentEvents.filter((event) => !event.isFinal);
  const signals: LiveSignal[] = [];
  const seen = new Set<string>();

  for (const event of [...finalEvents].reverse()) {
    const text = event.text.trim();
    if (isAddressedToMeEvent(event, addressedToNames)) {
      appendSignal(signals, seen, signalFromTranscript(event, "attention", {
        title: "Addressed to me",
        body: "Someone appears to be speaking to Brandon/BP.",
        priority: "high",
      }));
    }
    if (QUESTION_SIGNAL_RE.test(text)) {
      appendSignal(signals, seen, signalFromTranscript(event, "question", {
        title: "Clarification cue",
        body: "A question or clarification is active in the latest transcript.",
        priority: "normal",
      }));
    }
    if (hasActionSignalCue(text)) {
      appendSignal(signals, seen, signalFromTranscript(event, "action", {
        title: "Action cue",
        body: "A follow-up, owner, or timing cue appeared in the transcript.",
        priority: "high",
      }));
    }
    if (RISK_SIGNAL_RE.test(text)) {
      appendSignal(signals, seen, signalFromTranscript(event, "risk", {
        title: "Risk cue",
        body: "A risk, blocker, dependency, or assumption needs attention.",
        priority: "high",
      }));
    }
    if (DECISION_SIGNAL_RE.test(text)) {
      appendSignal(signals, seen, signalFromTranscript(event, "decision", {
        title: "Decision cue",
        body: "The transcript sounds like a decision or approval was stated.",
        priority: "high",
      }));
    }
    if (signals.length >= maxSignals) {
      break;
    }
  }

  for (const card of [...cards].sort(compareSidecarCards)) {
    const signal = signalFromCard(card);
    if (signal) {
      appendSignal(signals, seen, signal);
    }
    if (signals.length >= maxSignals) {
      break;
    }
  }

  const thread = finalEvents.length > 0
    ? signalFromThread(finalEvents.slice(-4), false)
    : partialEvents.length > 0
      ? signalFromThread([partialEvents[partialEvents.length - 1]], true)
      : null;
  if (thread) {
    appendSignal(signals, seen, thread);
  }

  return signals.slice(0, maxSignals);
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
    for (const sourceSegmentId of transcript.sourceSegmentIds ?? []) {
      rowByTranscriptKey.set(sourceSegmentId, row);
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

function appendSignal(signals: LiveSignal[], seen: Set<string>, signal: LiveSignal | null): void {
  if (!signal || !signal.evidenceQuote.trim()) {
    return;
  }
  const key = `${signal.kind}:${normalizeDisplayText(signal.evidenceQuote)}`;
  if (!key || seen.has(key)) {
    return;
  }
  seen.add(key);
  signals.push(signal);
}

function hasActionSignalCue(text: string): boolean {
  const normalized = text.replace(/\brate\s+schedule\b/gi, "rate profile");
  return ACTION_SIGNAL_RE.test(normalized);
}

function addressedToNamesForSignals(names: string[] | undefined): string[] {
  const requestedNames = names && names.length > 0 ? names : DEFAULT_ADDRESSED_TO_NAMES;
  return uniqueStrings(
    requestedNames
      .map((name) => name.trim())
      .filter((name) => name.length >= 2),
  ).sort((left, right) => right.length - left.length);
}

function isAddressedToMeEvent(event: TranscriptEvent, addressedToNames: string[]): boolean {
  return event.isFinal
    && event.source === "microphone"
    && event.speakerRole !== "user"
    && hasDirectAddressCue(event.text, addressedToNames);
}

function hasDirectAddressCue(text: string, addressedToNames: string[]): boolean {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) {
    return false;
  }
  for (const name of addressedToNames) {
    const namePattern = nameRegexSource(name);
    const optionalPrefix = `(?:${DIRECT_ADDRESS_PREFIX_RE}\\s+)?`;
    const startName = `${optionalPrefix}\\b${namePattern}\\b`;
    const punctuatedVocative = new RegExp(`^${startName}\\s*[,;:!-]\\s*(?!${DIRECT_ADDRESS_THIRD_PERSON_RE})\\S+`, "i");
    const requestAfterName = new RegExp(`^${startName}\\s+${DIRECT_ADDRESS_REQUEST_RE}\\b`, "i");
    const greetingOnly = new RegExp(`^\\s*(?:hey|hi|hello)\\s+\\b${namePattern}\\b(?:\\s*[,;:!-]|\\s*$|\\s+${DIRECT_ADDRESS_REQUEST_RE}\\b)`, "i");
    const trailingVocative = new RegExp(`\\b${DIRECT_ADDRESS_REQUEST_RE}\\b[^.!?]{0,120}\\b${namePattern}\\b\\s*[?.!,]?$`, "i");
    if (
      punctuatedVocative.test(clean)
      || requestAfterName.test(clean)
      || greetingOnly.test(clean)
      || trailingVocative.test(clean)
    ) {
      return true;
    }
  }
  return false;
}

function nameRegexSource(name: string): string {
  return name
    .trim()
    .split(/\s+/)
    .map(escapeRegex)
    .join("\\s+");
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function signalFromTranscript(
  event: TranscriptEvent,
  kind: Exclude<LiveSignalKind, "thread" | "suggestion">,
  details: {
    title: string;
    body: string;
    priority: SidecarPriority;
  },
): LiveSignal {
  const evidenceQuote = compactSignalText(event.text);
  return {
    id: stableLiveSignalId(kind, event.id, evidenceQuote),
    kind,
    title: details.title,
    body: details.body,
    evidenceQuote,
    sourceEventIds: [event.id],
    sourceSegmentIds: sourceSegmentIdsForEvent(event),
    priority: details.priority,
    provisional: false,
    at: event.at,
  };
}

function signalFromCard(card: SidecarDisplayCard): LiveSignal | null {
  if (!card.suggestedAsk && !card.suggestedSay) {
    return null;
  }
  const evidenceQuote = compactSignalText(card.evidenceQuote ?? "");
  if (!evidenceQuote) {
    return null;
  }
  return {
    id: stableLiveSignalId("suggestion", card.cardKey ?? card.id, evidenceQuote),
    kind: "suggestion",
    title: card.title || "Suggested contribution",
    body: card.suggestedAsk ? "Suggested ask from current meeting evidence." : "Suggested say-this from current meeting evidence.",
    evidenceQuote,
    sourceEventIds: [],
    sourceSegmentIds: [...(card.sourceSegmentIds ?? [])],
    priority: card.priority ?? "normal",
    provisional: Boolean(card.provisional),
    at: card.at,
    suggestedAsk: card.suggestedAsk,
    suggestedSay: card.suggestedSay,
  };
}

function signalFromThread(events: TranscriptEvent[], provisional: boolean): LiveSignal | null {
  const usableEvents = events.filter((event) => event.text.trim());
  if (usableEvents.length === 0) {
    return null;
  }
  const latest = usableEvents[usableEvents.length - 1];
  const evidenceQuote = compactSignalText(usableEvents.map((event) => event.text).join(" "));
  if (!evidenceQuote) {
    return null;
  }
  return {
    id: stableLiveSignalId("thread", latest.id, evidenceQuote),
    kind: "thread",
    title: provisional ? "Live preview" : "Current thread",
    body: evidenceQuote,
    evidenceQuote,
    sourceEventIds: usableEvents.map((event) => event.id),
    sourceSegmentIds: uniqueStrings(usableEvents.flatMap(sourceSegmentIdsForEvent)),
    priority: "low",
    provisional,
    at: latest.at,
  };
}

function sourceSegmentIdsForEvent(event: TranscriptEvent): string[] {
  return uniqueStrings([
    event.segmentId,
    ...(event.sourceSegmentIds ?? []),
  ].filter(Boolean) as string[]);
}

function compactSignalText(text: string, limit = LIVE_SIGNAL_EVIDENCE_CHARS): string {
  const clean = text.replace(/\s+/g, " ").trim();
  if (clean.length <= limit) {
    return clean;
  }
  const boundary = clean.lastIndexOf(" ", limit - 3);
  const end = boundary > 80 ? boundary : limit - 3;
  return `${clean.slice(0, end).trim()}...`;
}

function stableLiveSignalId(kind: LiveSignalKind, sourceId: string, evidenceQuote: string): string {
  const evidence = normalizeDisplayText(evidenceQuote).slice(0, 72).replace(/\s+/g, "-");
  const source = normalizeDisplayText(sourceId).slice(0, 48).replace(/\s+/g, "-") || "source";
  return `signal:${kind}:${source}:${evidence || "evidence"}`;
}

function uniqueStrings(values: string[]): string[] {
  const result: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    if (!value || seen.has(value)) {
      continue;
    }
    seen.add(value);
    result.push(value);
  }
  return result;
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
      : card.category === "reference"
        ? 34
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

function appendContractSection(lines: string[], contract?: MeetingContractBrief | null): void {
  lines.push("## Meeting Goal", "");
  lines.push(contract?.goal ? contract.goal : "- None", "");
  lines.push("## Contract Reminders", "");
  if (!contract?.reminders.length) {
    lines.push("- None", "");
    return;
  }
  lines.push(`- Mode: ${contract.mode}`);
  for (const reminder of contract.reminders) {
    lines.push(`- ${reminder}`);
  }
  lines.push("");
}

function appendSuggestedLanguageSection(lines: string[], cards: SidecarDisplayCard[]): void {
  const suggestions = cards.filter((card) => card.suggestedSay || card.suggestedAsk);
  lines.push("## Suggested Follow-up Language", "");
  if (suggestions.length === 0) {
    lines.push("- None", "");
    return;
  }
  for (const card of suggestions) {
    const label = card.suggestedAsk ? "Ask" : "Say";
    lines.push(`- **${card.title}** (${label}): ${card.suggestedAsk ?? card.suggestedSay}`);
  }
  lines.push("");
}

function appendEvidenceIndex(lines: string[], cards: SidecarDisplayCard[]): void {
  const evidenceCards = cards.filter((card) => card.evidenceQuote || (card.sourceSegmentIds?.length ?? 0) > 0);
  lines.push("## Evidence Index", "");
  if (evidenceCards.length === 0) {
    lines.push("- None", "");
    return;
  }
  for (const card of evidenceCards) {
    const sourceIds = card.sourceSegmentIds?.length ? ` [${card.sourceSegmentIds.join(", ")}]` : "";
    lines.push(`- **${card.title}**${sourceIds}`);
    if (card.evidenceQuote) {
      lines.push(`  - Evidence: "${card.evidenceQuote}"`);
    }
  }
  lines.push("");
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
