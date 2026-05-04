import { expect, test } from "@playwright/test";
import {
  buildSidecarDisplayCards,
  getTranscriptDisplayLabel,
  groupWorkCards,
  type SidecarDisplayCard,
  type TranscriptEvent,
} from "../src/presentation";

const now = Date.parse("2026-05-04T12:00:00Z");

test("labels microphone and trusted typed transcript neutrally", () => {
  expect(getTranscriptDisplayLabel(transcript("mic-1", "hello", { source: "microphone" }))).toBe("Mic");
  expect(getTranscriptDisplayLabel(transcript("typed-1", "apollo", { source: "typed" }))).toBe("You");
  expect(getTranscriptDisplayLabel(transcript("mic-user", "hello", {
    source: "microphone",
    speakerRole: "user",
    speakerConfidence: 0.91,
  }))).toBe("Me");
  expect(getTranscriptDisplayLabel(transcript("bp", "hello", {
    source: "microphone",
    speakerRole: "user",
    speakerLabel: "BP",
    speakerConfidence: 0.91,
  }))).toBe("BP");
  expect(getTranscriptDisplayLabel(transcript("mic-low", "hello", {
    source: "microphone",
    speakerRole: "user",
    speakerConfidence: 0.6,
  }))).toBe("Mic");
});

test("filters low score cards unless pinned, explicit, or high priority", () => {
  const cards = buildSidecarDisplayCards([
    card("low", { score: 0.42, summary: "A weak note about low relevance." }),
    card("pinned", { score: 0.2, pinned: true, summary: "Pinned context about the control room decision." }),
    card("explicit", { score: 0.3, explicitlyRequested: true, summary: "Explicit search result about a launch checklist." }),
    card("priority", { score: 0.1, priority: "high", summary: "High priority reminder about a safety handoff." }),
  ], [], { maxCards: Number.POSITIVE_INFINITY, now });

  expect(cards.map((item) => item.id)).not.toContain("low");
  expect(cards.map((item) => item.id)).toEqual(expect.arrayContaining(["pinned", "explicit", "priority"]));
});

test("keeps pinned cards visible after expiry", () => {
  const cards = buildSidecarDisplayCards([
    card("expired", { expiresAt: now - 1_000, summary: "Expired unpinned context about a rollout decision." }),
    card("pinned-expired", {
      expiresAt: now - 1_000,
      pinned: true,
      summary: "Pinned expired context about a rollback owner remains useful.",
    }),
  ], [], { maxCards: Number.POSITIVE_INFINITY, now });

  expect(cards.map((item) => item.id)).toEqual(["pinned-expired"]);
});

test("suppresses transcript echoes, duplicates, and raw transcript segments", () => {
  const recent = transcript("recent", "We need to follow up with the platform team tomorrow about deployment readiness.", {
    at: now - 60_000,
  });
  const cards = buildSidecarDisplayCards([
    card("echo", { summary: "We need to follow up with the platform team tomorrow about deployment readiness." }),
    card("dupe-a", { title: "Platform follow-up", summary: "Schedule a readiness sync with the platform team." }),
    card("dupe-b", { title: "Platform follow-up", summary: "Schedule a readiness sync with the platform team." }),
    card("raw", { debugMetadata: { sourceType: "transcript_segment", sourceId: "seg-1" } }),
    card("useful", { title: "Deployment risk", summary: "Previous rollout notes mention readiness reviews before platform launches." }),
  ], [recent], { maxCards: Number.POSITIVE_INFINITY, now });

  expect(cards.map((item) => item.id)).toEqual(expect.arrayContaining(["dupe-a", "useful"]));
  expect(cards.map((item) => item.id)).not.toEqual(expect.arrayContaining(["echo", "dupe-b", "raw"]));
});

test("limits visible cards to three by default", () => {
  const cards = buildSidecarDisplayCards([
    card("one", { summary: "First useful card about deployment readiness planning." }),
    card("two", { summary: "Second useful card about budget review preparation." }),
    card("three", { summary: "Third useful card about design review ownership." }),
    card("four", { summary: "Fourth useful card about schedule risk tracking." }),
  ], [], { now });

  expect(cards).toHaveLength(3);
});

test("groups useful cards into daily work buckets", () => {
  const groups = groupWorkCards([
    card("action", { category: "note", debugMetadata: { backendLabel: "action" } }),
    card("decision", { category: "note", debugMetadata: { backendLabel: "decision" } }),
    card("question", { category: "note", debugMetadata: { backendLabel: "question" } }),
    card("topic", { category: "note", debugMetadata: { backendLabel: "topic" } }),
    card("memory", { category: "memory" }),
    card("web", { category: "web", debugMetadata: { backendLabel: "topic" } }),
  ]);

  expect(groups.actions.map((item) => item.id)).toEqual(["action"]);
  expect(groups.decisions.map((item) => item.id)).toEqual(["decision"]);
  expect(groups.questions.map((item) => item.id)).toEqual(["question"]);
  expect(groups.context.map((item) => item.id)).toEqual(["topic", "memory", "web"]);
});

function transcript(id: string, text: string, overrides: Partial<TranscriptEvent> = {}): TranscriptEvent {
  return {
    id,
    text,
    at: now,
    source: "microphone",
    speakerRole: "unknown",
    isFinal: true,
    ...overrides,
  };
}

function card(id: string, overrides: Partial<SidecarDisplayCard> = {}): SidecarDisplayCard {
  return {
    id,
    category: "memory",
    title: "Useful context",
    summary: "Previous notes describe a useful project pattern for this discussion.",
    at: now,
    score: 0.88,
    ...overrides,
  };
}
