import { expect, test } from "@playwright/test";
import type { SidecarDisplayCard } from "../src/presentation";
import { buildMeetingSummaryMarkdown, buildMeetingSummaryNotesViewModel } from "../src/review/meetingSummaryNotes";
import type { ReviewJobResponse } from "../src/types";

test("builds quick meeting notes without the Meeting Capture taxonomy", () => {
  const notes = buildMeetingSummaryNotesViewModel(reviewJob(), reviewCards());
  const allText = notesText(notes);

  expect(notes.title).toBe("Apollo handoff");
  expect(notes.summaryParagraph).toContain("RFI log");
  expect(notes.noteGroups).toHaveLength(0);
  expect(notes.decisions.map((item) => item.text)).toContain("Sunil owns the Siemens document review path.");
  expect(notes.actions).toHaveLength(1);
  expect(notes.actions[0].text).toContain("BP will send the RFI log to Greg by Monday");
  expect(notes.openQuestions.map((item) => item.text)).toContain("Confirm Greg's preferred handoff format.");
  expect(notes.risks.map((item) => item.text)).toContain("The handoff depends on the RFI log reaching Greg on time.");
  expect(allText).not.toContain("Technical Findings");
  expect(allText).not.toContain("Parking Lot");
  expect(countOccurrences(allText, "BP will send the RFI log")).toBe(1);
});

test("copies a clean Meeting Summary markdown artifact", () => {
  const notes = buildMeetingSummaryNotesViewModel(reviewJob(), reviewCards());
  const markdown = buildMeetingSummaryMarkdown(notes);

  expect(markdown).toContain("# Meeting Summary");
  expect(markdown).toContain("_Apollo handoff_");
  expect(markdown).toContain("## Summary");
  expect(markdown).not.toContain("## Notes");
  expect(markdown).not.toContain("### Other notes");
  expect(markdown).toContain("## Decisions");
  expect(markdown).toContain("## Follow-ups");
  expect(markdown).toContain("## Open questions");
  expect(markdown).toContain("## Risks / concerns");
  expect(markdown).not.toContain("## Key Notes");
  expect(markdown).not.toContain("## Action Items / Follow-ups");
  expect(markdown).not.toContain("## Technical Findings");
  expect(markdown).not.toContain("## Source Notes");
  expect(markdown).not.toContain("Contract Reminders");
  expect(countOccurrences(markdown, "BP will send the RFI log")).toBe(1);
});

test("normalizes the IETF HTTPBIS fixture and suppresses local context leakage", () => {
  const notes = buildMeetingSummaryNotesViewModel(ietfReviewJob(), []);
  const allText = notesText(notes);
  const groupTitles = notes.noteGroups.map((group) => group.title);

  expect(notes.title).toBe("IETF 103 HTTPBIS technical meeting (20 min)");
  expect(notes.summaryParagraph).toContain("The HTTPBIS discussion focused on");
  expect(notes.summaryParagraph).toContain("Cache-Control");
  expect(groupTitles).toEqual(expect.arrayContaining([
    "Cache-Control",
    "DNS / CNAME",
    "Browser deployment",
    "GitHub issues",
    "Jabber relay",
  ]));
  expect(allText).toContain("Cache-Control");
  expect(allText).toContain("Cache-Control `private`");
  expect(allText).toContain("CNAME");
  expect(allText).toContain("HTTPBIS");
  expect(allText).toContain("IETF");
  expect(allText).toContain("DNS");
  expect(allText).toContain("Jabber");
  expect(allText).toContain("GitHub");
  expect(allText).not.toContain("Cash Control");
  expect(allText).not.toContain("PGE");
  expect(allText).not.toContain("Westwood");
  expect(allText).not.toContain("Energy Lens");
  expect(notes.actions.some((item) => /Jabber relay/i.test(item.text))).toBe(true);
  expect(notes.risks.some((item) => /Rayb has drafted/i.test(item.text))).toBe(false);
  expect(notes.risks.some((item) => /browser.*impediments/i.test(item.text))).toBe(true);
  expect(notes.openQuestions.some((item) => /quoted.*Cache-Control/i.test(item.text))).toBe(true);
  expect(notes.openQuestions.some((item) => /technical findings/i.test(item.text))).toBe(false);

  const markdown = buildMeetingSummaryMarkdown(notes);
  expect(markdown).toContain("### Cache-Control");
  expect(markdown).toContain("### DNS / CNAME");
  expect(markdown).toContain("### Browser deployment");
  expect(markdown).toContain("### GitHub issues");
  expect(markdown).toContain("### Jabber relay");
  expect(markdown).toContain("## Follow-ups");
  expect(markdown).not.toContain("PGE");
  expect(markdown).not.toContain("Westwood");
  expect(markdown).not.toContain("Energy Lens");
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
        text: "Confirm Greg's preferred handoff format. The handoff depends on the RFI log reaching Greg on time.",
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
      summary: "BP owns the RFI-log sendout to Greg by Monday, and Sunil owns the Siemens document review path.",
      key_points: [
        "BP will send the RFI log to Greg by Monday.",
        "Sunil owns the Siemens document review path.",
        "1 decision captured.",
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
      summary: "BP will send the RFI log to Greg by Monday.",
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

function ietfReviewJob(): ReviewJobResponse {
  return {
    id: "review_4e18455454934dc0",
    job_id: "review_4e18455454934dc0",
    source: "upload",
    title: "IETF 103 HTTPBIS technical meeting (20 min)",
    filename: "ietf103-httpbis-20min.mp3",
    status: "completed_awaiting_validation",
    validation_status: "pending",
    save_result: false,
    session_id: null,
    duration_seconds: 1200,
    raw_segment_count: 117,
    corrected_segment_count: 117,
    progress_percent: 100,
    steps: [],
    clean_segments: [
      {
        id: "ietf-seg-1",
        text: "Welcome to IETF HTTPBIS. We need one Jabber relay volunteer for the meeting.",
        start_s: 128,
        end_s: 168,
        source_segment_ids: ["ietf-seg-1"],
        speaker_label: "Unknown speaker",
      },
      {
        id: "ietf-seg-2",
        text: "The cache control private directive and quoted Cache-Control compatibility are being discussed.",
        start_s: 500,
        end_s: 540,
        source_segment_ids: ["ietf-seg-2"],
        speaker_label: "Unknown speaker",
      },
      {
        id: "ietf-seg-3",
        text: "The DNS draft addresses the CNAME problem at the apex domain, and the GitHub issue status was reviewed.",
        start_s: 650,
        end_s: 710,
        source_segment_ids: ["ietf-seg-3"],
        speaker_label: "Unknown speaker",
      },
      {
        id: "ietf-seg-4",
        text: "Browser developers should discuss deployment impediments for the HTTPBIS work.",
        start_s: 800,
        end_s: 840,
        source_segment_ids: ["ietf-seg-4"],
        speaker_label: "Unknown speaker",
      },
    ],
    meeting_cards: [],
    summary: {
      session_id: "review_4e18455454934dc0",
      title: "IETF 103 HTTPBIS technical meeting (20 min)",
      summary: "Review completed, but the generated meeting summary did not pass the usefulness gate. Validate the clean transcript and supporting evidence before approval.",
      key_points: [
        "Cash Control Directive Requirement Downgrade:",
        "Evaluate retention of 'cash control private' directive: The group is considering whether to keep the 'cash control private' directive in the caching specification.",
        "Clarify Quoted Cash Control Directive Generation: There's a discussion on whether to formally restrict senders from generating cash control directives in the quoted form, due to potential compatibility issues with older",
        "Apex Domain CNAME Limitation Solution:",
        "Rayb's Draft Record Type: Rayb has drafted a new record type to allow web browsers to find content without a CNAME record, specifically addressing the inability to use a CNAME at the apex of a domain. He",
        "ROS Developer Deployment Impediments: It was suggested that ROS developers be consulted about technical impediments to deploying the service in their browsers.",
        "GitHub Issue Status:",
        "PGE / Westwood relay coordination: A volunteer is needed to serve as a Jabber relay for the meeting. Someone was requested to step forward.",
      ],
      topics: ["HTTPBIS", "Cache-Control", "DNS"],
      projects: ["PGE / Westwood relay coordination"],
      project_workstreams: [
        {
          project: "Cash Control Directive Requirement Downgrade",
          decisions: ["The team decided to change the cash control directive requirement in the caching document from a firm requirement to an advisory note for clients, allowing them to choose whether or not to honor the directives."],
          source_segment_ids: ["ietf-seg-2"],
        },
        {
          project: "Clarify Quoted Cash Control Directive Generation",
          open_questions: ["There's a discussion on whether to formally restrict senders from generating cash control directives in the quoted form, due to potential compatibility issues with older"],
          risks: ["There's a discussion on whether to formally restrict senders from generating cash control directives in the quoted form, due to potential compatibility issues with older"],
          source_segment_ids: ["ietf-seg-2"],
        },
        {
          project: "Rayb's Draft Record Type",
          open_questions: ["Rayb has drafted a new record type to allow web browsers to find content without a CNAME record, specifically addressing the inability to use a CNAME at the apex of a domain. He"],
          source_segment_ids: ["ietf-seg-3"],
        },
        {
          project: "ROS Developer Deployment Impediments",
          open_questions: ["It was suggested that ROS developers be consulted about technical impediments to deploying the service in their browsers."],
          source_segment_ids: ["ietf-seg-4"],
        },
        {
          project: "PGE / Westwood relay coordination",
          open_questions: ["A volunteer is needed to serve as a Jabber relay for the meeting. Someone was requested to step forward."],
          source_segment_ids: ["ietf-seg-1"],
        },
      ],
      technical_findings: [
        {
          topic: "PGE / Westwood relay coordination",
          findings: ["A volunteer is needed to serve as a Jabber relay for the meeting. Someone was requested to step forward."],
          confidence: "medium",
          source_segment_ids: ["ietf-seg-1"],
        },
      ],
      decisions: ["The team decided to change the cash control directive requirement in the caching document from a firm requirement to an advisory note for clients, allowing them to choose whether or not to honor the directives."],
      actions: [],
      unresolved_questions: [
        "The group is considering whether to keep the 'cash control private' directive in the caching specification.",
        "There's a discussion on whether to formally restrict senders from generating cash control directives in the quoted form, due to potential compatibility issues with older",
        "A volunteer is needed to serve as a Jabber relay for the meeting. Someone was requested to step forward.",
      ],
      risks: ["There's a discussion on whether to formally restrict senders from generating cash control directives in the quoted form, due to potential compatibility issues with older"],
      entities: ["IETF", "HTTPBIS", "DNS", "GitHub", "Jabber"],
      lessons: [],
      diagnostics: {
        usefulness_status: "low_usefulness",
        usefulness_score: 0.45,
        usefulness_flags: ["low_usefulness"],
        quality_flags: ["missing_obvious_actions"],
      },
      source_segment_ids: ["ietf-seg-1", "ietf-seg-2", "ietf-seg-3", "ietf-seg-4"],
      created_at: 1,
      updated_at: 1,
    },
    raw_audio_retained: true,
    diagnostics: {},
    created_at: 1,
    updated_at: 1,
  };
}

function notesText(notes: ReturnType<typeof buildMeetingSummaryNotesViewModel>): string {
  return [
    notes.title,
    notes.summaryParagraph,
    ...notes.noteGroups.flatMap((group) => [group.title, ...group.items.map((item) => item.text)]),
    ...notes.keyNotes.map((item) => item.text),
    ...notes.decisions.map((item) => item.text),
    ...notes.actions.map((item) => item.text),
    ...notes.openQuestions.map((item) => item.text),
    ...notes.risks.map((item) => item.text),
  ].join("\n");
}

function countOccurrences(value: string, needle: string): number {
  return value.split(needle).length - 1;
}
