import type { SidecarDisplayCard } from "../presentation";
import type {
  ReviewJobResponse,
  SavedTranscriptSegment,
  SessionMemorySummary,
  SummaryProjectWorkstream,
  SummaryTechnicalFinding,
} from "../types";

export type MeetingCaptureItemKind =
  | "action"
  | "decision"
  | "open_question"
  | "risk"
  | "fact"
  | "technical_finding"
  | "parking_lot";

export type MeetingCaptureItem = {
  id: string;
  kind: MeetingCaptureItemKind;
  title: string;
  body: string;
  owner?: string;
  dueDate?: string;
  project?: string;
  confidence?: number;
  evidenceQuote?: string;
  sourceSegmentIds: string[];
  sourceLabels?: string[];
  lowConfidence?: boolean;
  reason?: string;
};

export type MeetingCaptureViewModel = {
  title: string;
  snapshot: string[];
  actions: MeetingCaptureItem[];
  decisions: MeetingCaptureItem[];
  openQuestions: MeetingCaptureItem[];
  risks: MeetingCaptureItem[];
  facts: MeetingCaptureItem[];
  technicalFindings: MeetingCaptureItem[];
  parkingLot: MeetingCaptureItem[];
  transcriptSegments: SavedTranscriptSegment[];
  diagnostics?: Record<string, unknown>;
};

type Candidate = MeetingCaptureItem & {
  order: number;
  strength: number;
};

type CandidateGroups = Record<MeetingCaptureItemKind, Candidate[]>;

const LOW_CONFIDENCE_THRESHOLD = 0.72;

const FILLER_PHRASES = [
  "the meeting covered",
  "the review centered on",
  "the review focused on",
  "the call covered",
  "follow up on this",
  "discussed project",
  "summary uses",
  "review output",
  "meeting output",
];

const DROSS_PATTERNS = [
  /\bthe (meeting|review|call) (covered|centered on|focused on|was about)\b/i,
  /\bfollow up on this\b/i,
  /\bdiscussed project\b/i,
  /\bextended evidence row\b/i,
  /\bsupporting cards?\b/i,
  /\bprimary meeting summary\b/i,
  /\breview grounded\b/i,
  /\bsummary uses corrected transcript/i,
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

export function buildMeetingCaptureViewModel(
  job: ReviewJobResponse,
  cards: SidecarDisplayCard[] = [],
): MeetingCaptureViewModel {
  const summary = job.summary;
  const transcriptSegments = job.clean_segments ?? [];
  const segmentTextKeys = new Set(transcriptSegments.map((segment) => normalizeCaptureText(segment.text)).filter(Boolean));
  const sourceProjectMap = buildSourceProjectMap(summary);
  const sourceLabelMap = buildSourceLabelMap(transcriptSegments);
  const projectNames = distinctStrings([
    ...summaryList(summary?.projects),
    ...summaryProjectWorkstreamList(summary?.project_workstreams).map((workstream) => cleanCaptureText(workstream.project)),
  ]);
  const primaryProject = projectNames.length === 1 ? projectNames[0] : "";
  const groups: CandidateGroups = {
    action: [],
    decision: [],
    open_question: [],
    risk: [],
    fact: [],
    technical_finding: [],
    parking_lot: [],
  };
  let order = 0;

  const push = (
    kind: MeetingCaptureItemKind,
    titleInput: unknown,
    options: Partial<MeetingCaptureItem> & { strength?: number; body?: string } = {},
  ) => {
    const title = cleanCaptureText(titleInput);
    const body = cleanCaptureText(options.body) || title;
    if (!title || shouldSuppressCaptureText(kind, `${title} ${body}`, options)) {
      return;
    }
    const sourceSegmentIds = distinctStrings(options.sourceSegmentIds ?? []);
    const inferredProject = inferProjectFromSources(sourceSegmentIds, sourceProjectMap) || primaryProject;
    const confidence = normalizedConfidence(options.confidence);
    const lowConfidence = Boolean(
      options.lowConfidence
        || (typeof confidence === "number" && confidence < LOW_CONFIDENCE_THRESHOLD)
        || unknownSpeakerOwnership(title, body, options.owner),
    );
    const candidate: Candidate = {
      id: `${kind}-${order}`,
      kind,
      title,
      body,
      owner: usefulOwner(options.owner) || inferOwner(title, body),
      dueDate: cleanCaptureText(options.dueDate) || inferDueDate(`${title} ${body}`),
      project: cleanCaptureText(options.project) || inferredProject,
      confidence,
      evidenceQuote: cleanCaptureText(options.evidenceQuote),
      sourceSegmentIds,
      sourceLabels: sourceLabelsForIds(sourceSegmentIds, sourceLabelMap),
      lowConfidence,
      reason: cleanCaptureText(options.reason),
      order: order++,
      strength: options.strength ?? 20,
    };
    if (kind !== "parking_lot" && lowConfidence) {
      groups.parking_lot.push({
        ...candidate,
        id: `parking_lot-${candidate.order}`,
        kind: "parking_lot",
        reason: candidate.reason || "Needs confirmation before treating as capture.",
        strength: Math.max(5, candidate.strength - 20),
      });
      return;
    }
    if (kind === "technical_finding" && mostlyTranscriptEcho(candidate, segmentTextKeys)) {
      groups.parking_lot.push({
        ...candidate,
        id: `parking_lot-${candidate.order}`,
        kind: "parking_lot",
        reason: "Transcript echo, not a distinct technical finding.",
        strength: Math.max(5, candidate.strength - 20),
      });
      return;
    }
    groups[kind].push(candidate);
  };

  if (summary) {
    addSummaryCandidates(summary, push);
    addWorkstreamCandidates(summary, push);
    addTechnicalFindingCandidates(summary, push, segmentTextKeys);
    addFactCandidates(summary, push);
  }
  addCardCandidates(cards, push, sourceProjectMap, primaryProject);

  const actions = mergeCandidates(groups.action);
  const decisions = mergeCandidates(groups.decision);
  const openQuestions = mergeCandidates(groups.open_question);
  const risks = mergeCandidates(groups.risk)
    .filter((item) => !captureItemMatchesAny(item, [...actions, ...decisions, ...openQuestions]));
  const canonical = [...actions, ...decisions, ...openQuestions, ...risks];
  const facts = mergeCandidates(groups.fact)
    .filter((item) => !captureItemMatchesAny(item, canonical))
    .slice(0, 8);
  const technicalFindings = mergeCandidates(groups.technical_finding)
    .filter((item) => !captureItemMatchesAny(item, [...canonical, ...facts]))
    .slice(0, 6);
  const parkingLot = mergeCandidates(groups.parking_lot)
    .filter((item) => !captureItemMatchesAny(item, [...canonical, ...facts, ...technicalFindings]))
    .slice(0, 12);

  return {
    title: cleanCaptureText(summary?.title) || cleanCaptureText(job.title) || "Meeting Capture",
    snapshot: buildSnapshot(summary, [...canonical, ...facts, ...technicalFindings]),
    actions,
    decisions,
    openQuestions,
    risks,
    facts,
    technicalFindings,
    parkingLot,
    transcriptSegments,
    diagnostics: summary?.diagnostics ?? job.diagnostics,
  };
}

export function buildMeetingCaptureMarkdown(viewModel: MeetingCaptureViewModel): string {
  const lines: string[] = ["# Meeting Capture", ""];
  if (viewModel.title && viewModel.title !== "Meeting Capture") {
    lines.push(`_${viewModel.title}_`, "");
  }
  appendMarkdownBullets(lines, "Snapshot", viewModel.snapshot);
  appendMarkdownItems(lines, "Follow-ups / Actions", viewModel.actions, { action: true });
  appendMarkdownItems(lines, "Decisions", viewModel.decisions);
  appendMarkdownItems(lines, "Open Questions / Unknowns", viewModel.openQuestions);
  appendMarkdownItems(lines, "Risks / Blockers", viewModel.risks);
  appendMarkdownItems(lines, "Important Facts / Context", viewModel.facts);
  appendMarkdownItems(lines, "Technical Findings", viewModel.technicalFindings);
  appendMarkdownItems(lines, "Parking Lot / Low Confidence", viewModel.parkingLot);
  lines.push("", "## Source Notes");
  const sourceIds = distinctStrings([
    ...viewModel.actions,
    ...viewModel.decisions,
    ...viewModel.openQuestions,
    ...viewModel.risks,
    ...viewModel.facts,
    ...viewModel.technicalFindings,
    ...viewModel.parkingLot,
  ].flatMap((item) => item.sourceSegmentIds));
  if (sourceIds.length === 0) {
    lines.push("- None");
  } else {
    for (const sourceId of sourceIds) {
      const segment = viewModel.transcriptSegments.find((candidate) => segmentIds(candidate).includes(sourceId));
      const label = segment ? sourceLabel(segment) : sourceId;
      lines.push(`- ${label}`);
    }
  }
  return `${lines.join("\n").replace(/\n{3,}/g, "\n\n").trim()}\n`;
}

export function normalizeCaptureText(value: unknown): string {
  let text = cleanCaptureText(value).toLowerCase();
  for (const phrase of FILLER_PHRASES) {
    text = text.replaceAll(phrase, " ");
  }
  return text
    .replace(/['"]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((token) => token && !STOPWORDS.has(token))
    .join(" ")
    .trim();
}

function addSummaryCandidates(
  summary: SessionMemorySummary,
  push: (kind: MeetingCaptureItemKind, titleInput: unknown, options?: Partial<MeetingCaptureItem> & { strength?: number; body?: string }) => void,
) {
  const rollup = summary.portfolio_rollup;
  for (const action of summaryList(rollup?.bp_next_actions)) {
    push("action", action, { sourceSegmentIds: summaryList(rollup?.source_segment_ids), strength: 34 });
  }
  for (const loop of summaryList(rollup?.open_loops)) {
    push(looksLikeQuestion(loop) ? "open_question" : "risk", loop, {
      sourceSegmentIds: summaryList(rollup?.source_segment_ids),
      strength: 30,
    });
  }
  for (const dependency of summaryList(rollup?.cross_project_dependencies)) {
    push("risk", dependency, {
      body: cleanCaptureText(dependency).replace(/^[^:]{2,28}:\s*/, ""),
      sourceSegmentIds: summaryList(rollup?.source_segment_ids),
      reason: "Cross-project dependency",
      strength: 30,
    });
  }
  for (const action of summaryList(summary.actions)) {
    push("action", action, { strength: 44 });
  }
  for (const decision of summaryList(summary.decisions)) {
    push("decision", decision, { strength: 44 });
  }
  for (const question of summaryList(summary.unresolved_questions)) {
    push("open_question", question, { strength: 44 });
  }
  for (const risk of summaryList(summary.risks)) {
    push("risk", risk, { strength: 44 });
  }
}

function addWorkstreamCandidates(
  summary: SessionMemorySummary,
  push: (kind: MeetingCaptureItemKind, titleInput: unknown, options?: Partial<MeetingCaptureItem> & { strength?: number; body?: string }) => void,
) {
  for (const workstream of summaryProjectWorkstreamList(summary.project_workstreams)) {
    const project = cleanCaptureText(workstream.project);
    const owners = summaryList(workstream.owners);
    const owner = owners[0] ?? "";
    const dueDate = inferDueDate(workstream.next_checkpoint);
    const sourceSegmentIds = summaryList(workstream.source_segment_ids);
    for (const action of summaryList(workstream.actions)) {
      push("action", action, { project, owner, dueDate, sourceSegmentIds, strength: 68 });
    }
    for (const decision of summaryList(workstream.decisions)) {
      push("decision", decision, { project, owner, sourceSegmentIds, strength: 66 });
    }
    for (const question of summaryList(workstream.open_questions)) {
      push("open_question", question, { project, owner, sourceSegmentIds, strength: 64 });
    }
    for (const risk of summaryList(workstream.risks)) {
      push("risk", risk, { project, owner, sourceSegmentIds, strength: 64 });
    }
  }
}

function addTechnicalFindingCandidates(
  summary: SessionMemorySummary,
  push: (kind: MeetingCaptureItemKind, titleInput: unknown, options?: Partial<MeetingCaptureItem> & { strength?: number; body?: string }) => void,
  segmentTextKeys: Set<string>,
) {
  for (const finding of summaryTechnicalFindingList(summary.technical_findings)) {
    const sourceSegmentIds = summaryList(finding.source_segment_ids);
    const topic = cleanCaptureText(finding.topic || finding.question || "Technical finding");
    for (const risk of summaryList(finding.risks)) {
      push("risk", risk, { sourceSegmentIds, reason: topic, strength: 42 });
    }
    for (const gap of summaryList(finding.data_gaps)) {
      push("open_question", gap, { sourceSegmentIds, reason: `Data gap: ${topic}`, strength: 42 });
    }
    if (!isGroundedTechnicalFinding(finding, segmentTextKeys)) {
      const fallback = cleanCaptureText(
        summaryList(finding.findings)[0]
          || summaryList(finding.reference_context)[0]
          || finding.question
          || finding.topic,
      );
      if (fallback) {
        push("parking_lot", fallback, {
          sourceSegmentIds,
          reason: "Weak technical finding; needs confirmation.",
          strength: 18,
        });
      }
      continue;
    }
    const bodyParts = [
      ...prefixList("Recommendation", summaryList(finding.recommendations)),
      ...prefixList("Method", summaryList(finding.methods)),
      ...prefixList("Assumption", summaryList(finding.assumptions)),
      ...prefixList("Finding", summaryList(finding.findings)),
    ];
    const confidence = confidenceScore(finding.confidence);
    push("technical_finding", topic, {
      body: bodyParts.slice(0, 4).join(" "),
      sourceSegmentIds,
      confidence,
      strength: 58 + (summaryList(finding.recommendations).length > 0 ? 8 : 0),
    });
  }
}

function addFactCandidates(
  summary: SessionMemorySummary,
  push: (kind: MeetingCaptureItemKind, titleInput: unknown, options?: Partial<MeetingCaptureItem> & { strength?: number; body?: string }) => void,
) {
  const sourceSegmentIds = summaryList(summary.source_segment_ids);
  for (const keyPoint of summaryList(summary.key_points)) {
    push("fact", keyPoint, { sourceSegmentIds, strength: 28 });
  }
}

function addCardCandidates(
  cards: SidecarDisplayCard[],
  push: (kind: MeetingCaptureItemKind, titleInput: unknown, options?: Partial<MeetingCaptureItem> & { strength?: number; body?: string }) => void,
  sourceProjectMap: Map<string, string>,
  primaryProject: string,
) {
  for (const card of cards) {
    const sourceSegmentIds = distinctStrings(card.sourceSegmentIds ?? []);
    const project = inferProjectFromSources(sourceSegmentIds, sourceProjectMap) || primaryProject;
    const common = {
      body: card.summary,
      owner: card.owner,
      dueDate: card.dueDate,
      project,
      confidence: card.confidence,
      evidenceQuote: card.evidenceQuote,
      sourceSegmentIds,
      sourceLabels: card.sourceLabel ? [card.sourceLabel] : undefined,
    };
    const lowConfidence = typeof card.confidence === "number" && card.confidence < LOW_CONFIDENCE_THRESHOLD;
    if (lowConfidence || card.missingInfo) {
      push("parking_lot", card.title || card.summary, {
        ...common,
        lowConfidence,
        reason: card.missingInfo ? `Missing: ${card.missingInfo}` : "Low-confidence card.",
        strength: 24,
      });
      continue;
    }
    switch (card.category) {
      case "action":
        push("action", card.title || card.summary, { ...common, strength: 84 });
        break;
      case "decision":
        push("decision", card.title || card.summary, { ...common, strength: 82 });
        break;
      case "question":
      case "clarification":
        push("open_question", card.title || card.summary, { ...common, strength: 80 });
        break;
      case "risk":
        push("risk", card.title || card.summary, { ...common, strength: 80 });
        break;
      case "note":
      case "contribution":
        if (sourceSegmentIds.length > 0) {
          push("fact", card.title || card.summary, { ...common, strength: 34 });
        }
        break;
      default:
        break;
    }
  }
}

function mergeCandidates(candidates: Candidate[]): MeetingCaptureItem[] {
  const merged: Candidate[] = [];
  for (const candidate of candidates.sort((left, right) => left.order - right.order)) {
    const index = merged.findIndex((item) => captureItemsMatch(item, candidate));
    if (index === -1) {
      merged.push(candidate);
      continue;
    }
    merged[index] = mergeCandidate(merged[index], candidate);
  }
  return merged
    .sort((left, right) => left.order - right.order)
    .map(({ order: _order, strength: _strength, ...item }) => ({
      ...item,
      id: stableCaptureItemId(item),
    }));
}

function mergeCandidate(left: Candidate, right: Candidate): Candidate {
  const stronger = right.strength > left.strength ? right : left;
  const weaker = stronger === right ? left : right;
  const sourceSegmentIds = distinctStrings([...left.sourceSegmentIds, ...right.sourceSegmentIds]);
  return {
    ...left,
    title: stronger.title || weaker.title,
    body: stronger.body || weaker.body,
    owner: usefulOwner(stronger.owner) || usefulOwner(weaker.owner),
    dueDate: stronger.dueDate || weaker.dueDate,
    project: stronger.project || weaker.project,
    confidence: stronger.confidence ?? weaker.confidence,
    evidenceQuote: stronger.evidenceQuote || weaker.evidenceQuote,
    sourceSegmentIds,
    sourceLabels: distinctStrings([...(left.sourceLabels ?? []), ...(right.sourceLabels ?? [])]),
    lowConfidence: Boolean(left.lowConfidence && right.lowConfidence),
    reason: stronger.reason || weaker.reason,
    order: Math.min(left.order, right.order),
    strength: Math.max(left.strength, right.strength),
  };
}

function captureItemsMatch(left: MeetingCaptureItem, right: MeetingCaptureItem): boolean {
  if (left.kind !== right.kind && left.kind !== "parking_lot" && right.kind !== "parking_lot") {
    return false;
  }
  const leftKey = normalizeCaptureText(`${left.title} ${left.body}`);
  const rightKey = normalizeCaptureText(`${right.title} ${right.body}`);
  if (!leftKey || !rightKey) {
    return false;
  }
  if (leftKey === rightKey || leftKey.includes(rightKey) || rightKey.includes(leftKey)) {
    return true;
  }
  const overlap = tokenOverlap(leftKey, rightKey);
  const sourceOverlap = left.sourceSegmentIds.some((id) => right.sourceSegmentIds.includes(id));
  return overlap >= (sourceOverlap ? 0.42 : 0.62);
}

function captureItemMatchesAny(item: MeetingCaptureItem, existing: MeetingCaptureItem[]): boolean {
  return existing.some((candidate) => captureItemsMatch(item, candidate));
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

function buildSnapshot(summary: SessionMemorySummary | null, primaryItems: MeetingCaptureItem[]): string[] {
  const bullets: string[] = [];
  for (const keyPoint of summaryList(summary?.key_points)) {
    const text = cleanCaptureText(keyPoint);
    if (!text || DROSS_PATTERNS.some((pattern) => pattern.test(text))) {
      continue;
    }
    const candidate: MeetingCaptureItem = {
      id: "snapshot-candidate",
      kind: "fact",
      title: text,
      body: text,
      sourceSegmentIds: summaryList(summary?.source_segment_ids),
    };
    if (!captureItemMatchesAny(candidate, primaryItems)) {
      bullets.push(text);
    }
  }
  const counts = [
    [primaryItems.filter((item) => item.kind === "action").length, "follow-up captured", "follow-ups captured"],
    [primaryItems.filter((item) => item.kind === "decision").length, "decision captured", "decisions captured"],
    [primaryItems.filter((item) => item.kind === "open_question").length, "open question needs confirmation", "open questions need confirmation"],
    [primaryItems.filter((item) => item.kind === "risk").length, "risk or blocker to watch", "risks or blockers to watch"],
    [primaryItems.filter((item) => item.kind === "technical_finding").length, "technical finding grounded in the transcript", "technical findings grounded in the transcript"],
  ] as const;
  for (const [count, single, plural] of counts) {
    if (bullets.length >= 6) {
      break;
    }
    if (count > 0) {
      bullets.push(`${count} ${count === 1 ? single : plural}.`);
    }
  }
  return distinctStrings(bullets).slice(0, 6);
}

function shouldSuppressCaptureText(
  kind: MeetingCaptureItemKind,
  text: string,
  options: Partial<MeetingCaptureItem>,
): boolean {
  const value = cleanCaptureText(text);
  if (!value) {
    return true;
  }
  if (DROSS_PATTERNS.some((pattern) => pattern.test(value))) {
    return kind !== "parking_lot";
  }
  const normalized = normalizeCaptureText(value);
  if (normalized.length < 8) {
    return true;
  }
  if (kind === "action" && /\b(follow up|circle back|discuss)\b/i.test(value) && !options.owner && !options.dueDate && !options.project) {
    return true;
  }
  return false;
}

function isGroundedTechnicalFinding(finding: SummaryTechnicalFinding, segmentTextKeys: Set<string>): boolean {
  const recommendations = summaryList(finding.recommendations);
  const methods = summaryList(finding.methods);
  const assumptions = summaryList(finding.assumptions);
  const dataGaps = summaryList(finding.data_gaps);
  const findings = summaryList(finding.findings);
  const hasTechnicalSignal = recommendations.length > 0
    || methods.length > 0
    || assumptions.length > 0
    || dataGaps.length > 0
    || Boolean(cleanCaptureText(finding.confidence));
  if (!hasTechnicalSignal) {
    return false;
  }
  const combined = normalizeCaptureText([...recommendations, ...methods, ...assumptions, ...dataGaps, ...findings].join(" "));
  if (!combined) {
    return false;
  }
  for (const segmentKey of segmentTextKeys) {
    if (segmentKey && (combined === segmentKey || tokenOverlap(combined, segmentKey) > 0.82)) {
      return recommendations.length > 0 || methods.length > 0 || dataGaps.length > 0;
    }
  }
  const referenceOnly = findings.length === 0
    && recommendations.length === 0
    && methods.length === 0
    && assumptions.length === 0
    && summaryList(finding.reference_context).length > 0;
  return !referenceOnly;
}

function mostlyTranscriptEcho(item: MeetingCaptureItem, segmentTextKeys: Set<string>): boolean {
  const key = normalizeCaptureText(`${item.title} ${item.body}`);
  if (!key) {
    return true;
  }
  for (const segmentKey of segmentTextKeys) {
    if (segmentKey && (key === segmentKey || tokenOverlap(key, segmentKey) > 0.88)) {
      return true;
    }
  }
  return false;
}

function buildSourceProjectMap(summary: SessionMemorySummary | null): Map<string, string> {
  const map = new Map<string, string>();
  for (const workstream of summaryProjectWorkstreamList(summary?.project_workstreams)) {
    const project = cleanCaptureText(workstream.project);
    if (!project) {
      continue;
    }
    for (const sourceId of summaryList(workstream.source_segment_ids)) {
      map.set(sourceId, project);
    }
  }
  return map;
}

function buildSourceLabelMap(segments: SavedTranscriptSegment[]): Map<string, string> {
  const map = new Map<string, string>();
  for (const segment of segments) {
    const label = sourceLabel(segment);
    for (const id of segmentIds(segment)) {
      map.set(id, label);
    }
  }
  return map;
}

function sourceLabelsForIds(ids: string[], sourceLabelMap: Map<string, string>): string[] {
  return distinctStrings(ids.map((id) => sourceLabelMap.get(id) ?? id));
}

function sourceLabel(segment: SavedTranscriptSegment): string {
  const speaker = cleanCaptureText(segment.speaker_label) || "Speaker";
  const time = `${formatCaptureTime(segment.start_s)}-${formatCaptureTime(segment.end_s)}`;
  return `${segment.id}: ${speaker} ${time}`;
}

function segmentIds(segment: SavedTranscriptSegment): string[] {
  return distinctStrings([segment.id, ...(segment.source_segment_ids ?? [])]);
}

function inferProjectFromSources(sourceSegmentIds: string[], sourceProjectMap: Map<string, string>): string {
  for (const sourceId of sourceSegmentIds) {
    const project = sourceProjectMap.get(sourceId);
    if (project) {
      return project;
    }
  }
  return "";
}

function usefulOwner(owner: unknown): string {
  const value = cleanCaptureText(owner);
  if (!value || /unknown speaker/i.test(value)) {
    return "";
  }
  return value;
}

function unknownSpeakerOwnership(title: string, body: string, owner: unknown): boolean {
  return /unknown speaker/i.test(`${cleanCaptureText(owner)} ${title} ${body}`);
}

function inferOwner(title: string, body: string): string {
  const text = `${title} ${body}`;
  const ownerMatch = text.match(/\b(?:owner|owned by|owns|assigned to)\s*:?\s*([A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.'-]*){0,2})/);
  if (ownerMatch?.[1]) {
    return ownerMatch[1].trim();
  }
  const leadMatch = text.match(/^([A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.'-]*){0,2})\s+(?:will|owns|to)\b/);
  return leadMatch?.[1]?.trim() ?? "";
}

function inferDueDate(value: unknown): string {
  const text = cleanCaptureText(value);
  const match = text.match(/\b(today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|next week|this week|eod|end of day|by [A-Z]?[a-z]+(?:day)?|by \d{1,2}\/\d{1,2})\b/i);
  if (!match) {
    return "";
  }
  return match[0].replace(/^by\s+/i, "").trim();
}

function looksLikeQuestion(value: unknown): boolean {
  const text = cleanCaptureText(value);
  return text.endsWith("?") || /\b(confirm|clarify|question|whether|who owns|which)\b/i.test(text);
}

function confidenceScore(value: unknown): number | undefined {
  const text = cleanCaptureText(value).toLowerCase();
  if (!text) {
    return undefined;
  }
  if (text === "high") return 0.9;
  if (text === "medium") return 0.74;
  if (text === "low") return 0.45;
  const numeric = Number(text);
  return Number.isFinite(numeric) ? normalizedConfidence(numeric) : undefined;
}

function normalizedConfidence(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return undefined;
  }
  if (value > 1) {
    return Math.max(0, Math.min(1, value / 100));
  }
  return Math.max(0, Math.min(1, value));
}

function stableCaptureItemId(item: MeetingCaptureItem): string {
  const key = normalizeCaptureText(`${item.kind} ${item.title} ${item.body}`).slice(0, 72).replace(/\s+/g, "-");
  const source = item.sourceSegmentIds.slice().sort().join("-").replace(/[^a-zA-Z0-9-]+/g, "-");
  return `${item.kind}-${key || "item"}${source ? `-${source}` : ""}`;
}

function cleanCaptureText(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value).replace(/\s+/g, " ").trim();
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["text", "description", "body", "summary", "title", "value", "item"]) {
      const text = cleanCaptureText(record[key]);
      if (text) {
        return text;
      }
    }
  }
  return "";
}

function summaryList(value: unknown): string[] {
  return Array.isArray(value) ? distinctStrings(value.map(cleanCaptureText).filter(Boolean)) : [];
}

function summaryProjectWorkstreamList(value: unknown): SummaryProjectWorkstream[] {
  return Array.isArray(value) ? value.filter((item): item is SummaryProjectWorkstream => Boolean(item && typeof item === "object")) : [];
}

function summaryTechnicalFindingList(value: unknown): SummaryTechnicalFinding[] {
  return Array.isArray(value) ? value.filter((item): item is SummaryTechnicalFinding => Boolean(item && typeof item === "object")) : [];
}

function distinctStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const value of values) {
    const text = cleanCaptureText(value);
    const key = text.toLowerCase();
    if (!text || seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(text);
  }
  return result;
}

function prefixList(prefix: string, values: string[]): string[] {
  return values.map((value) => `${prefix}: ${value}`);
}

function appendMarkdownBullets(lines: string[], title: string, items: string[]) {
  lines.push(`## ${title}`);
  if (items.length === 0) {
    lines.push("- None");
    return;
  }
  for (const item of items) {
    lines.push(`- ${item}`);
  }
}

function appendMarkdownItems(
  lines: string[],
  title: string,
  items: MeetingCaptureItem[],
  options: { action?: boolean } = {},
) {
  lines.push("", `## ${title}`);
  if (items.length === 0) {
    lines.push(options.action ? "- No follow-ups captured." : "- None");
    return;
  }
  for (const item of items) {
    const metadata = [
      item.owner ? `Owner: ${item.owner}` : "",
      item.dueDate ? `Due: ${item.dueDate}` : "",
      item.project ? `Project: ${item.project}` : "",
      item.lowConfidence ? "Low confidence" : "",
      item.reason,
    ].filter(Boolean);
    const body = item.body && item.body !== item.title ? `: ${item.body}` : "";
    lines.push(`- **${item.title}**${body}${metadata.length > 0 ? ` (${metadata.join("; ")})` : ""}`);
    if (item.evidenceQuote) {
      lines.push(`  - Evidence: "${item.evidenceQuote}"`);
    }
    if (item.sourceSegmentIds.length > 0) {
      lines.push(`  - Sources: ${item.sourceSegmentIds.join(", ")}`);
    }
  }
}

function formatCaptureTime(seconds: number): string {
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const remainder = safe % 60;
  return `${minutes}:${String(remainder).padStart(2, "0")}`;
}
