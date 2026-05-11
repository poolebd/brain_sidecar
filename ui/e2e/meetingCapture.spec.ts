import { expect, test } from "@playwright/test";
import type { SidecarDisplayCard } from "../src/presentation";
import { buildMeetingCaptureMarkdown, buildMeetingCaptureViewModel } from "../src/review/meetingCapture";
import type { ReviewJobResponse } from "../src/types";

test("deduplicates Review output into one meeting capture model", () => {
  const capture = buildMeetingCaptureViewModel(reviewJob(), reviewCards());

  expect(capture.actions).toHaveLength(1);
  expect(capture.actions[0]).toMatchObject({
    title: "Send RFI log",
    owner: "BP",
    dueDate: "Monday",
    project: "Apollo",
    evidenceQuote: "BP will send the RFI log to Greg by Monday.",
  });
  expect(capture.actions[0].sourceSegmentIds).toEqual(["seg-1"]);

  expect(capture.decisions).toHaveLength(1);
  expect(capture.decisions[0].title).toBe("Review path owner");
  expect(capture.decisions[0].body).toBe("Sunil owns the Siemens document review path.");

  expect(capture.technicalFindings.map((item) => item.title)).not.toContain("Weak reference echo");
  expect(capture.parkingLot.map((item) => item.title)).toContain("External reference context might matter.");
  expect(capture.snapshot.join(" ")).not.toContain("BP will send the RFI log to Greg by Monday.");
});

test("copies a clean Meeting Capture markdown artifact", () => {
  const capture = buildMeetingCaptureViewModel(reviewJob(), reviewCards());
  const markdown = buildMeetingCaptureMarkdown(capture);

  expect(markdown).toContain("# Meeting Capture");
  expect(markdown).toContain("## Snapshot");
  expect(markdown).toContain("## Follow-ups / Actions");
  expect(markdown).toContain("## Decisions");
  expect(markdown).toContain("## Open Questions / Unknowns");
  expect(markdown).toContain("## Risks / Blockers");
  expect(markdown).toContain("## Important Facts / Context");
  expect(markdown).toContain("## Technical Findings");
  expect(markdown).toContain("## Parking Lot / Low Confidence");
  expect(markdown).toContain("## Source Notes");
  expect(markdown).not.toContain("## Reference Context");
  expect(markdown).not.toContain("Contract Reminders");
  expect(countOccurrences(markdown, "Send RFI log")).toBe(1);
});

function reviewJob(): ReviewJobResponse {
  return {
    id: "reviewjob-test",
    job_id: "reviewjob-test",
    source: "upload",
    title: "Apollo handoff",
    filename: "apollo.webm",
    status: "completed_awaiting_validation",
    validation_status: "pending",
    save_result: false,
    session_id: null,
    duration_seconds: 20,
    raw_segment_count: 3,
    corrected_segment_count: 3,
    progress_percent: 100,
    steps: [],
    clean_segments: [
      {
        id: "seg-1",
        text: "BP will send the RFI log to Greg by Monday.",
        start_s: 0,
        end_s: 6,
        source_segment_ids: ["seg-1"],
        speaker_label: "BP",
      },
      {
        id: "seg-2",
        text: "Sunil owns the Siemens document review path.",
        start_s: 6,
        end_s: 12,
        source_segment_ids: ["seg-2"],
        speaker_label: "Sunil",
      },
      {
        id: "seg-3",
        text: "External reference context might matter.",
        start_s: 12,
        end_s: 18,
        source_segment_ids: ["seg-3"],
        speaker_label: "Other speaker",
      },
    ],
    meeting_cards: [],
    summary: {
      session_id: "reviewjob-test",
      title: "Apollo document handoff",
      summary: "The review centered on Apollo follow-up.",
      key_points: [
        "BP will send the RFI log to Greg by Monday.",
        "Sunil owns the Siemens document review path.",
      ],
      topics: ["apollo"],
      projects: ["Apollo"],
      project_workstreams: [
        {
          project: "Apollo",
          actions: ["BP will send the RFI log to Greg by Monday."],
          decisions: ["Sunil owns the Siemens document review path."],
          risks: ["The handoff depends on the RFI log reaching Greg on time."],
          open_questions: ["Confirm Greg's preferred handoff format."],
          owners: ["BP"],
          next_checkpoint: "Monday RFI sendout",
          source_segment_ids: ["seg-1"],
        },
      ],
      technical_findings: [
        {
          topic: "Weak reference echo",
          findings: ["External reference context might matter."],
          reference_context: ["Current public web context"],
          source_segment_ids: ["seg-3"],
        },
      ],
      decisions: ["Sunil owns the Siemens document review path."],
      actions: ["BP will send the RFI log to Greg by Monday."],
      unresolved_questions: ["Confirm Greg's preferred handoff format."],
      risks: ["The handoff depends on the RFI log reaching Greg on time."],
      entities: ["BP", "Greg", "Sunil"],
      lessons: [],
      source_segment_ids: ["seg-1", "seg-2", "seg-3"],
      created_at: 1,
      updated_at: 1,
    },
    raw_audio_retained: true,
    diagnostics: {},
    created_at: 1,
    updated_at: 1,
  };
}

function reviewCards(): SidecarDisplayCard[] {
  return [
    {
      id: "card-action",
      category: "action",
      title: "Send RFI log",
      summary: "BP owns sending the RFI log to Greg by Monday.",
      at: 1,
      confidence: 0.9,
      sourceSegmentIds: ["seg-1"],
      evidenceQuote: "BP will send the RFI log to Greg by Monday.",
      owner: "BP",
      dueDate: "Monday",
    },
    {
      id: "card-decision",
      category: "decision",
      title: "Review path owner",
      summary: "Sunil owns the Siemens document review path.",
      at: 1,
      confidence: 0.84,
      sourceSegmentIds: ["seg-2"],
    },
  ];
}

function countOccurrences(value: string, needle: string): number {
  return value.split(needle).length - 1;
}
