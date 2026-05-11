import { useState, type ChangeEvent, type RefObject } from "react";
import { buildConsultingBriefMarkdown, type MeetingOutputBucket, type SidecarDisplayCard, type TranscriptEvent } from "../presentation";
import { FilePickAction } from "../shared/FilePickAction";
import type {
  ReviewJobResponse,
  ReviewStep,
  SessionMemorySummary,
  SummaryPortfolioRollup,
  SummaryProjectWorkstream,
  SummaryReferenceContext,
  SummaryReviewMetrics,
  SummaryTechnicalFinding,
} from "../types";

const TEST_AUDIO_ACCEPT = "audio/*,.wav,.mp3,.m4a,.aac,.flac,.ogg,.mp4,.webm";
const MEETING_OUTPUT_SECTIONS: { bucket: MeetingOutputBucket; title: string; empty: string }[] = [
  { bucket: "actions", title: "Actions", empty: "No actions yet." },
  { bucket: "decisions", title: "Decisions", empty: "No decisions yet." },
  { bucket: "questions", title: "Questions / Clarifications", empty: "No open questions yet." },
  { bucket: "risks", title: "Risks", empty: "No risks yet." },
  { bucket: "notes", title: "Other Notes", empty: "No other meeting notes yet." },
];

type ReviewPageProps = {
  title: string;
  job: ReviewJobResponse | null;
  jobs: ReviewJobResponse[];
  busy: string;
  error: string;
  transcriptEvents: TranscriptEvent[];
  cards: SidecarDisplayCard[];
  groups: Record<MeetingOutputBucket, SidecarDisplayCard[]>;
  expandedCards: Set<string>;
  highlightedSourceIds: Set<string>;
  showDebugMetadata: boolean;
  transcriptScrollRef: RefObject<HTMLDivElement | null>;
  onTitleChange: (value: string) => void;
  onFilePick: (event: ChangeEvent<HTMLInputElement>) => void;
  onNewReview: () => void;
  onRecoverLatest: () => void;
  onCancel: () => void;
  onApprove: () => void;
  onDiscard: () => void;
  onSelectJob: (job: ReviewJobResponse) => void;
  onToggleCard: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onCopyCard: (card: SidecarDisplayCard) => void;
  onCopyTranscript: () => void;
  onCopyBrief: () => void;
  onOpenSession: () => void;
};

export function ReviewPage({
  title,
  job,
  jobs,
  busy,
  error,
  transcriptEvents,
  cards,
  groups,
  expandedCards,
  highlightedSourceIds,
  showDebugMetadata,
  transcriptScrollRef,
  onTitleChange,
  onFilePick,
  onNewReview,
  onRecoverLatest,
  onCancel,
  onApprove,
  onDiscard,
  onSelectJob,
  onToggleCard,
  onJumpToEvidence,
  onCopyCard,
  onCopyTranscript,
  onCopyBrief,
  onOpenSession,
}: ReviewPageProps) {
  const completed = job?.status === "completed";
  const awaitingValidation = job?.status === "completed_awaiting_validation";
  const approved = job?.status === "approved";
  const reviewReady = awaitingValidation || approved || completed;
  const running = Boolean(job && !isReviewTerminal(job.status) && job.status !== "completed_awaiting_validation");
  const summaryPrimary = Boolean(job?.summary && reviewReady);
  const hasReviewEvidence = Boolean(transcriptEvents.length > 0 || cards.length > 0 || reviewReady);
  const [transcriptOpen, setTranscriptOpen] = useState(false);
  const revealTranscriptAndJump = (card: SidecarDisplayCard) => {
    setTranscriptOpen(true);
    window.setTimeout(() => onJumpToEvidence(card), 0);
  };

  return (
    <main className="page review-page" aria-label="Review">
      <ReviewHeader
        job={job}
        jobs={jobs}
        busy={busy}
        reviewReady={reviewReady}
        awaitingValidation={awaitingValidation}
        approved={approved}
        summaryPrimary={summaryPrimary}
        onNewReview={onNewReview}
        onRecoverLatest={onRecoverLatest}
        onCancel={onCancel}
        onApprove={onApprove}
        onDiscard={onDiscard}
        onOpenSession={onOpenSession}
        onSelectJob={onSelectJob}
      />

      {(error || job?.error) && (
        <section className="errors" role="alert">
          <p>{error || job?.error}</p>
        </section>
      )}

      {!job && (
        <>
          <ReviewSetupControl
            title={title}
            job={job}
            busy={busy}
            onTitleChange={onTitleChange}
            onFilePick={onFilePick}
          />
          <section className="empty-state review-empty-state" aria-label="No review loaded">
            <h2>{busy === "loading-latest" ? "Looking for the latest review" : "No review loaded"}</h2>
            <p>{busy === "loading-latest" ? "Checking whether a review is already in flight." : "Upload audio to generate a clean transcript and meeting summary."}</p>
          </section>
        </>
      )}

      {job && summaryPrimary && (
        <section className="review-summary-workbench" aria-label="Review workspace">
          <section className="review-summary-primary" aria-label="Review summary validation">
            <SummaryValidationToolbar
              job={job}
              busy={busy}
              reviewReady={reviewReady}
              awaitingValidation={awaitingValidation}
              approved={approved}
              cardsLength={cards.length}
              onCopyBrief={onCopyBrief}
              onApprove={onApprove}
              onDiscard={onDiscard}
              onOpenSession={onOpenSession}
            />
            <MeetingSummaryBrief summary={job.summary!} cards={cards} />
          </section>

          <aside className="review-completed-controls" aria-label="Review status and source tools">
            <details
              className="review-ops-details review-transcript-details"
              aria-label="Transcript and sources"
              open={transcriptOpen}
              onToggle={(event) => setTranscriptOpen(event.currentTarget.open)}
            >
              <summary>Transcript & sources</summary>
              <TranscriptEvidencePanel
                job={job}
                running={running}
                reviewReady={reviewReady}
                transcriptEvents={transcriptEvents}
                highlightedSourceIds={highlightedSourceIds}
                transcriptScrollRef={transcriptScrollRef}
                onCopyTranscript={onCopyTranscript}
              />
            </details>
          </aside>

          <SupportingCardsPanel
            job={job}
            groups={groups}
            cards={cards}
            expandedCards={expandedCards}
            showDebugMetadata={showDebugMetadata}
            reviewReady={reviewReady}
            summaryPrimary={summaryPrimary}
            onToggleCard={onToggleCard}
            onJumpToEvidence={revealTranscriptAndJump}
            onCopyCard={onCopyCard}
            onCopyBrief={onCopyBrief}
          />
        </section>
      )}

      {job && !summaryPrimary && (
        <section className="review-active-workbench" aria-label="Review workspace">
          {job.summary ? <MeetingSummaryBrief summary={job.summary} cards={cards} /> : <ReviewSummaryPlaceholder job={job} running={running} />}

          <ReviewStatusDetails job={job} busy={busy} error={error} />
          <ReviewProgressDetails job={job} />

          {hasReviewEvidence && (
            <>
              <SupportingCardsPanel
                job={job}
                groups={groups}
                cards={cards}
                expandedCards={expandedCards}
                showDebugMetadata={showDebugMetadata}
                reviewReady={reviewReady}
                summaryPrimary={summaryPrimary}
                onToggleCard={onToggleCard}
                onJumpToEvidence={onJumpToEvidence}
                onCopyCard={onCopyCard}
                onCopyBrief={onCopyBrief}
              />
              <TranscriptEvidencePanel
                job={job}
                running={running}
                reviewReady={reviewReady}
                transcriptEvents={transcriptEvents}
                highlightedSourceIds={highlightedSourceIds}
                transcriptScrollRef={transcriptScrollRef}
                onCopyTranscript={onCopyTranscript}
              />
            </>
          )}

          <aside className="review-completed-controls review-secondary-controls" aria-label="Review upload controls">
            <details className="review-ops-details review-upload-details" aria-label="Queue another review">
              <summary>Upload another file</summary>
              <ReviewSetupControl
                title={title}
                job={job}
                busy={busy}
                onTitleChange={onTitleChange}
                onFilePick={onFilePick}
              />
            </details>
          </aside>
        </section>
      )}
    </main>
  );
}

function ReviewHeader({
  job,
  jobs,
  busy,
  reviewReady,
  awaitingValidation,
  approved,
  summaryPrimary,
  onNewReview,
  onRecoverLatest,
  onCancel,
  onApprove,
  onDiscard,
  onOpenSession,
  onSelectJob,
}: {
  job: ReviewJobResponse | null;
  jobs: ReviewJobResponse[];
  busy: string;
  reviewReady: boolean;
  awaitingValidation: boolean;
  approved: boolean;
  summaryPrimary: boolean;
  onNewReview: () => void;
  onRecoverLatest: () => void;
  onCancel: () => void;
  onApprove: () => void;
  onDiscard: () => void;
  onOpenSession: () => void;
  onSelectJob: (job: ReviewJobResponse) => void;
}) {
  const canceled = job?.status === "canceled";
  const cancelable = Boolean(job && (job.status === "queued" || job.status === "paused_for_live"));
  const [queueOpen, setQueueOpen] = useState(false);
  return (
    <header className="page-header review-page-header">
      <div>
        <p className="label">Review</p>
        <h1>Meeting Summary</h1>
        <span>Turn audio into a validated summary, then save it into Sessions.</span>
      </div>
      <div className="page-header-actions">
        <ReviewQueueMenu open={queueOpen} jobs={jobs} onOpenChange={setQueueOpen} />
        {job && (
          <span className={`status-pill ${job.status === "error" ? "danger" : reviewReady ? "good" : canceled ? "warning" : "busy"} ${job ? "summary-hidden" : ""}`}>
            <span aria-hidden="true" />
            {reviewStatusLabel(job.status)}
          </span>
        )}
        {job && (
          <button className="secondary" type="button" onClick={onNewReview} disabled={busy === "uploading"}>
            New audio
          </button>
        )}
        {!job && <button className="secondary" type="button" onClick={onRecoverLatest} disabled={busy === "loading-latest" || busy === "uploading"}>
          {busy === "loading-latest" ? "Finding..." : "Latest"}
        </button>}
        {cancelable && (
          <button className="danger ghost" type="button" onClick={onCancel} disabled={busy === "canceling"}>
            {busy === "canceling" ? "Canceling..." : "Cancel"}
          </button>
        )}
        {awaitingValidation && !summaryPrimary && (
          <>
            <button className="primary" type="button" onClick={onApprove} disabled={busy === "approving"}>{busy === "approving" ? "Approving..." : "Approve"}</button>
            <button className="secondary" type="button" onClick={onDiscard} disabled={busy === "discarding"}>{busy === "discarding" ? "Discarding..." : "Discard"}</button>
          </>
        )}
        {approved && job?.session_id && !summaryPrimary && (
          <button className="secondary" type="button" onClick={onOpenSession}>Open Session</button>
        )}
      </div>
      {queueOpen && jobs.length > 0 && (
        <div className="review-header-queue-drawer">
          <ReviewQueuePanel job={job} jobs={jobs} onSelectJob={onSelectJob} />
        </div>
      )}
    </header>
  );
}

function ReviewQueueMenu({
  jobs,
  open,
  onOpenChange,
}: {
  jobs: ReviewJobResponse[];
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  if (jobs.length === 0) {
    return null;
  }
  return (
    <details
      className="review-header-queue"
      aria-label="Review queue menu"
      open={open}
      onToggle={(event) => onOpenChange(event.currentTarget.open)}
    >
      <summary>Queue ({jobs.length})</summary>
    </details>
  );
}

function ReviewQueuePanel({
  job,
  jobs,
  onSelectJob,
}: {
  job: ReviewJobResponse | null;
  jobs: ReviewJobResponse[];
  onSelectJob: (job: ReviewJobResponse) => void;
}) {
  return (
    <section className="review-queue-panel" aria-label="Review queue">
      <div className="section-heading-row">
        <h2>Queue</h2>
        <span className="chip">{jobs.length} jobs</span>
      </div>
      <div className="review-queue-list">
        {jobs.map((item) => (
          <button
            key={reviewJobId(item)}
            type="button"
            className={`review-queue-item ${job && reviewJobId(job) === reviewJobId(item) ? "active" : ""}`}
            onClick={() => onSelectJob(item)}
          >
            <span>{displayReviewTitle(item.title)}</span>
            <strong>{reviewStatusLabel(item.status)}</strong>
            <small>{item.queue_position ? `Queue ${item.queue_position}` : item.message || item.phase || item.source}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function ReviewSetupControl({
  title,
  job,
  busy,
  onTitleChange,
  onFilePick,
}: {
  title: string;
  job: ReviewJobResponse | null;
  busy: string;
  onTitleChange: (value: string) => void;
  onFilePick: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <section className="review-control-panel" aria-label="Review setup">
      <label className="field field-wide">
        <span>Title</span>
        <input
          aria-label="Review title"
          value={title}
          onChange={(event) => onTitleChange(event.target.value)}
          placeholder="Optional review title"
          disabled={busy === "uploading"}
        />
      </label>
      <FilePickAction
        label="Review audio upload"
        actionLabel={job ? "Upload another" : "Upload audio"}
        accept={TEST_AUDIO_ACCEPT}
        path={job?.filename ?? ""}
        busy={busy === "uploading"}
        disabled={busy === "uploading"}
        onChange={onFilePick}
      />
      <div className="review-retention-note" aria-label="Review retention">
        <strong>Manual approval required</strong>
        <span>Temporary audio is retained locally until approve, discard, cancel, or cleanup.</span>
      </div>
    </section>
  );
}

function SummaryValidationToolbar({
  job,
  busy,
  reviewReady,
  awaitingValidation,
  approved,
  cardsLength,
  onCopyBrief,
  onApprove,
  onDiscard,
  onOpenSession,
}: {
  job: ReviewJobResponse;
  busy: string;
  reviewReady: boolean;
  awaitingValidation: boolean;
  approved: boolean;
  cardsLength: number;
  onCopyBrief: () => void;
  onApprove: () => void;
  onDiscard: () => void;
  onOpenSession: () => void;
}) {
  const usefulness = reviewUsefulness(job);
  const needsManualValidation = usefulness?.status === "low_usefulness" || usefulness?.status === "needs_repair";
  const validationTitle = approved
    ? "Saved to Sessions"
    : awaitingValidation
      ? needsManualValidation ? "Needs manual validation" : "Needs validation"
      : "Summary ready";
  const validationDetail = needsManualValidation
    ? `${usefulness?.flagsLabel || "Usefulness flag"} found. Validate transcript evidence before saving.`
    : "Check the summary and evidence before saving.";
  return (
    <div className="review-summary-toolbar">
      <div>
        <p className="label">Decision</p>
        <h2>{validationTitle}</h2>
        <span>{validationDetail}</span>
      </div>
      <div className="review-summary-actions" aria-label="Summary validation actions">
        <button className="secondary" type="button" disabled={!reviewReady || (!job.summary && cardsLength === 0)} onClick={onCopyBrief}>Copy brief</button>
        {awaitingValidation && (
          <>
            <button className="primary" type="button" onClick={onApprove} disabled={busy === "approving"}>{busy === "approving" ? "Approving..." : "Approve"}</button>
            <button className="secondary" type="button" onClick={onDiscard} disabled={busy === "discarding"}>{busy === "discarding" ? "Discarding..." : "Discard"}</button>
          </>
        )}
        {approved && job.session_id && (
          <button className="secondary" type="button" onClick={onOpenSession}>Open Session</button>
        )}
      </div>
    </div>
  );
}

function ReviewStatusDetails({
  job,
  busy,
  error,
}: {
  job: ReviewJobResponse | null;
  busy: string;
  error: string;
}) {
  if (!job && !busy && !error) {
    return null;
  }
  const progressPercent = reviewOverallProgress(job, busy);
  const feedback = reviewFeedback(job, busy, error);
  const asrLabel = reviewAsrLabel(job);
  const speakerIdentityLabel = reviewSpeakerIdentityLabel(job);
  const usefulness = reviewUsefulness(job);
  const durationLabel = job?.duration_seconds ? formatPlaybackClock(job.duration_seconds) : null;
  const activeStep = reviewActiveStep(job);
  const elapsedLabel = reviewElapsedLabel(job);
  return (
    <section className="review-status-panel" aria-label="Review status">
      <div className="review-status-main">
        <div>
          <p className="label">Status</p>
          <strong>{feedback.title}</strong>
          <span>{feedback.detail}</span>
        </div>
        <div className="review-status-chips" aria-label="Review metadata">
          {job?.filename && <span className="chip">{job.filename}</span>}
          <span className="chip">{job?.validation_status === "approved" ? "Approved" : job?.validation_status === "discarded" ? "Discarded" : "Awaiting approval"}</span>
          {usefulness && <span className={`chip ${usefulness.tone}`}>{usefulness.label}</span>}
          <span className="chip">{job?.temporary_audio_retained ? "Temporary audio retained" : job?.audio_deleted_at ? "Audio deleted" : "No temporary audio"}</span>
          {job?.audio_delete_error && <span className="chip danger">Cleanup failed</span>}
          {activeStep?.current && activeStep.total && <span className="chip">{activeStep.current}/{activeStep.total} {activeStep.unit || "step"}</span>}
          {durationLabel && <span className="chip">{durationLabel}</span>}
          {elapsedLabel && <span className="chip">{elapsedLabel}</span>}
        </div>
      </div>
      <div
        className="review-progress-track"
        role="progressbar"
        aria-label="Review overall progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={progressPercent}
      >
        <span style={{ width: `${progressPercent}%` }} />
      </div>
      {job && <details className="review-runtime-details" aria-label="Review runtime details">
        <summary>Runtime details</summary>
        <div className="chip-row">
          <span className="chip">{asrLabel}</span>
          {speakerIdentityLabel && <span className="chip">{speakerIdentityLabel}</span>}
          <span className="chip">{job.temporary_audio_retained ? "temporary audio retained" : "audio deleted"}</span>
          {job.raw_segment_count != null && <span className="chip">{job.raw_segment_count} raw segments</span>}
          {job.corrected_segment_count != null && <span className="chip">{job.corrected_segment_count} clean segments</span>}
          {usefulness?.scoreLabel && <span className={`chip ${usefulness.tone}`}>{usefulness.scoreLabel}</span>}
          {usefulness?.flagsLabel && <span className="chip warning">{usefulness.flagsLabel}</span>}
          {job.audio_deleted_at && <span className="chip">deleted {formatDateTime(job.audio_deleted_at)}</span>}
          {job.audio_delete_error && <span className="chip danger">{job.audio_delete_error}</span>}
        </div>
      </details>}
    </section>
  );
}

function ReviewValidationUtilityStrip({
  job,
  busy,
  error,
  showDebugMetadata,
}: {
  job: ReviewJobResponse;
  busy: string;
  error: string;
  showDebugMetadata: boolean;
}) {
  const progressPercent = reviewOverallProgress(job, busy);
  const feedback = reviewFeedback(job, busy, error);
  const asrLabel = reviewAsrLabel(job);
  const speakerIdentityLabel = reviewSpeakerIdentityLabel(job);
  const usefulness = reviewUsefulness(job);
  const durationLabel = job.duration_seconds ? formatPlaybackClock(job.duration_seconds) : "";
  const elapsedLabel = reviewElapsedLabel(job);
  const completedSteps = job.steps.filter((step) => step.status === "completed").length;
  const currentStep = reviewActiveStep(job)
    ?? [...job.steps].reverse().find((step) => step.status === "completed")
    ?? job.steps[0];
  const chips = [
    job.filename,
    job.validation_status === "approved" ? "Approved" : job.validation_status === "discarded" ? "Discarded" : "Awaiting approval",
    usefulness?.label,
    job.temporary_audio_retained ? "Temporary audio retained" : job.audio_deleted_at ? "Audio deleted" : "No temporary audio",
    `${progressPercent}% complete`,
    `${completedSteps}/${job.steps.length} steps`,
    asrLabel,
    speakerIdentityLabel,
    job.raw_segment_count != null ? `${job.raw_segment_count} raw segments` : "",
    job.corrected_segment_count != null ? `${job.corrected_segment_count} clean segments` : "",
    durationLabel,
    elapsedLabel,
    usefulness?.scoreLabel,
    usefulness?.flagsLabel,
  ].filter((value): value is string => Boolean(value));

  return (
    <section className="review-validation-strip" aria-label="Review status">
      <div className="review-validation-strip-head">
        <div>
          <p className="label">Status</p>
          <strong>{feedback.title}</strong>
          <span>{currentStep?.message || feedback.detail || job.message || "Review output ready for validation."}</span>
        </div>
        <div
          className="review-progress-track"
          role="progressbar"
          aria-label="Review overall progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={progressPercent}
        >
          <span style={{ width: `${progressPercent}%` }} />
        </div>
      </div>
      <section className="review-validation-strip-chips" aria-label="Review progress">
        {chips.map((chip, index) => (
          <span key={`${chip}-${index}`} className={`chip ${chip === usefulness?.label ? usefulness?.tone ?? "" : ""}`}>
            {chip}
          </span>
        ))}
        {job.audio_delete_error && <span className="chip danger">{job.audio_delete_error}</span>}
      </section>
      {showDebugMetadata && (
        <details className="review-runtime-details" aria-label="Review runtime details">
          <summary>Debug runtime</summary>
          <div className="chip-row">
            {job.audio_deleted_at && <span className="chip">deleted {formatDateTime(job.audio_deleted_at)}</span>}
            {job.audio_delete_error && <span className="chip danger">{job.audio_delete_error}</span>}
            {job.steps.map((step) => (
              <span key={step.key} className={`review-step-pill ${step.status}`}>
                {step.label}: {reviewStepStatusLabel(step.status)}
              </span>
            ))}
          </div>
        </details>
      )}
    </section>
  );
}

function ReviewProgressDetails({ job, compact = false }: { job: ReviewJobResponse | null; compact?: boolean }) {
  if (!job) {
    return null;
  }
  if (compact) {
    const completedSteps = job.steps.filter((step) => step.status === "completed").length;
    const currentStep = job.steps.find((step) => step.status === "running")
      ?? [...job.steps].reverse().find((step) => step.status === "completed")
      ?? job.steps[0];
    return (
      <section className="review-progress review-progress-compact" aria-label="Review progress">
        <div className="review-progress-compact-head">
          <div>
            <p className="label">Processing</p>
            <strong>{currentStep?.label ?? "Review"}</strong>
            <span>{currentStep?.message || job.message || "Processing status is ready."}</span>
          </div>
          <span className="chip">{completedSteps}/{job.steps.length} steps</span>
        </div>
        <div
          className="review-progress-track"
          role="progressbar"
          aria-label="Review processing progress"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={reviewOverallProgress(job, "")}
        >
          <span style={{ width: `${reviewOverallProgress(job, "")}%` }} />
        </div>
        <div className="review-progress-compact-steps" aria-label="Review step status">
          {job.steps.map((step) => (
            <span key={step.key} className={`review-step-pill ${step.status}`}>
              {step.label}: {reviewStepStatusLabel(step.status)}
            </span>
          ))}
        </div>
      </section>
    );
  }
  return (
    <section className="review-progress" aria-label="Review progress">
      {job.steps.map((step) => (
        <article key={step.key} className={`review-step ${step.status}`}>
          <div className="review-step-head">
            <span>{step.label}</span>
            <strong>{reviewStepStatusLabel(step.status)}</strong>
          </div>
          <div
            className="review-step-bar"
            role="progressbar"
            aria-label={`${step.label} progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={reviewStepProgress(step)}
          >
            <span style={{ width: `${reviewStepProgress(step)}%` }} />
          </div>
          <small>
            {step.message || "Pending"}
            {step.current && step.total ? ` (${step.current}/${step.total} ${step.unit || "items"})` : ""}
          </small>
        </article>
      ))}
    </section>
  );
}

function ReviewSummaryPlaceholder({ job, running }: { job: ReviewJobResponse; running: boolean }) {
  return (
    <section className="review-summary-placeholder" aria-label="Meeting Summary">
      <p className="label">Meeting Summary</p>
      <h2>{running ? job.title || "Meeting summary is being generated" : "No meeting summary yet"}</h2>
      <p>{running ? "Review is still conditioning the transcript and extracting meeting output." : "This review has no summary output yet."}</p>
    </section>
  );
}

function TranscriptEvidencePanel({
  job,
  running,
  reviewReady,
  transcriptEvents,
  highlightedSourceIds,
  transcriptScrollRef,
  onCopyTranscript,
}: {
  job: ReviewJobResponse;
  running: boolean;
  reviewReady: boolean;
  transcriptEvents: TranscriptEvent[];
  highlightedSourceIds: Set<string>;
  transcriptScrollRef: RefObject<HTMLDivElement | null>;
  onCopyTranscript: () => void;
}) {
  return (
    <section className="review-transcript-panel" aria-label="Clean Transcript">
      <div className="section-heading-row">
        <h2>Transcript Evidence</h2>
        <div className="button-row">
          <span className="chip">{job.corrected_segment_count} segments</span>
          <button className="secondary" type="button" disabled={!reviewReady || transcriptEvents.length === 0} onClick={onCopyTranscript}>Copy transcript</button>
        </div>
      </div>
      <div className="review-transcript-scroll scroll" ref={transcriptScrollRef as RefObject<HTMLDivElement>}>
        {transcriptEvents.length === 0 ? (
          <p className="empty-note">{running ? "Transcription is running." : "No transcript segments yet."}</p>
        ) : transcriptEvents.map((event) => {
          const ids = new Set([event.segmentId ?? event.id, ...(event.sourceSegmentIds ?? [])]);
          const highlighted = Array.from(ids).some((id) => highlightedSourceIds.has(id));
          return (
            <article
              key={event.id}
              className={`saved-transcript-row ${highlighted ? "highlight" : ""}`}
              data-transcript-id={event.segmentId ?? event.id}
              data-source-segment-ids={(event.sourceSegmentIds ?? []).join(" ")}
            >
              <span>{formatSecondsFromMs(event.startedAt)}-{formatSecondsFromMs(event.endedAt)}</span>
              <strong>{event.speakerLabel ?? "Speaker"}</strong>
              <p>{event.text}</p>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function SupportingCardsPanel({
  job,
  groups,
  cards,
  expandedCards,
  showDebugMetadata,
  reviewReady,
  summaryPrimary,
  onToggleCard,
  onJumpToEvidence,
  onCopyCard,
  onCopyBrief,
}: {
  job: ReviewJobResponse;
  groups: Record<MeetingOutputBucket, SidecarDisplayCard[]>;
  cards: SidecarDisplayCard[];
  expandedCards: Set<string>;
  showDebugMetadata: boolean;
  reviewReady: boolean;
  summaryPrimary: boolean;
  onToggleCard: (id: string) => void;
  onJumpToEvidence: (card: SidecarDisplayCard) => void;
  onCopyCard: (card: SidecarDisplayCard) => void;
  onCopyBrief: () => void;
}) {
  const visibleSections = MEETING_OUTPUT_SECTIONS.filter((section) => (groups[section.bucket] ?? []).length > 0);
  const validationCards = summaryPrimary ? reviewValidationEvidenceCards(job.summary ?? null, cards) : [];
  const coveredCardCount = summaryPrimary ? Math.max(0, cards.length - validationCards.length) : 0;
  const panelLabel = summaryPrimary ? "Validation Evidence Cards" : "Supporting Meeting Cards";
  const validationEmpty = summaryPrimary && validationCards.length === 0;
  return (
    <section className={`review-output-panel review-supporting-stack ${validationEmpty ? "review-validation-empty" : ""}`} aria-label={panelLabel}>
      <div className="section-heading-row">
        <h2>{summaryPrimary ? "Validation Evidence" : "Evidence Cards"}</h2>
        <div className="button-row">
          <span className="chip">
            {summaryPrimary ? `${validationCards.length} shown${coveredCardCount > 0 ? ` · ${coveredCardCount} covered` : ""}` : `${cards.length} items`}
          </span>
          {!summaryPrimary && (
            <button className="secondary" type="button" disabled={!reviewReady || (!job.summary && cards.length === 0)} onClick={onCopyBrief}>Copy brief</button>
          )}
        </div>
      </div>
      <div className="review-card-list scroll">
        {summaryPrimary ? (
          validationCards.length === 0 ? (
            <p className="empty-note">
              {cards.length > 0
                ? "No separate validation flags; repeated action and decision cards are covered by the summary."
                : reviewReady ? "No validation evidence cards were produced." : "Validation evidence is pending."}
            </p>
          ) : (
            <div className="review-validation-list">
              {validationCards.map((item) => (
                <ReviewValidationEvidenceCard
                  key={`review-validation-${item.card.id}`}
                  item={item}
                  expanded={expandedCards.has(item.card.id)}
                  showDebugMetadata={showDebugMetadata}
                  onToggle={() => onToggleCard(item.card.id)}
                  onJumpToEvidence={() => onJumpToEvidence(item.card)}
                  onCopy={() => onCopyCard(item.card)}
                />
              ))}
            </div>
          )
        ) : visibleSections.length === 0 ? (
          <p className="empty-note">{reviewReady ? "No grounded meeting cards were produced." : "Meeting output is pending."}</p>
        ) : visibleSections.map((section) => (
          <section key={section.bucket} className="work-note-section meeting-output-section" aria-label={`${section.title} review`}>
            <div className="work-note-section-head">
              <h3>{section.title}</h3>
              <span>{(groups[section.bucket] ?? []).length}</span>
            </div>
            <div className="context-preview-list">
              {(groups[section.bucket] ?? []).map((card) => (
                <ReviewContextCard
                  key={`review-${card.id}`}
                  card={card}
                  expanded={expandedCards.has(card.id)}
                  showDebugMetadata={showDebugMetadata}
                  onToggle={() => onToggleCard(card.id)}
                  onJumpToEvidence={() => onJumpToEvidence(card)}
                  onCopy={() => onCopyCard(card)}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}

export function MeetingSummaryBrief({
  summary,
  cards = [],
  compact = false,
}: {
  summary: SessionMemorySummary;
  cards?: SidecarDisplayCard[];
  compact?: boolean;
}) {
  const brief = buildReviewBriefModel(summary, cards);
  const visibleSections = [
    { key: "actions", title: "Actions", items: brief.actions },
    { key: "decisions", title: "Decisions", items: brief.decisions },
    { key: "open-questions", title: "Open Questions", items: brief.openQuestions },
    { key: "risks", title: "Risks / Watch Items", items: brief.risks },
  ] as const;
  return (
    <section className={`meeting-summary-brief ${compact ? "compact" : ""}`} aria-label="Meeting Summary">
      <section className="meeting-summary-hero" aria-label="Executive Summary">
        <div>
          <p className="label">Executive Summary</p>
          <h2>{formatReviewDisplayText(brief.title)}</h2>
          <p>{formatReviewDisplayText(brief.executiveSummary)}</p>
        </div>
        <div className="meeting-summary-evidence" aria-label="Summary evidence">
          <strong>{brief.sourceCount || "No"} sources</strong>
          <span>{formatReviewDisplayText(brief.coverageNote || "Grounded in reviewed transcript evidence.")}</span>
          {brief.primaryProject && <span>{brief.projectMode === "multi" ? `${brief.projectCount} workstreams` : brief.primaryProject}</span>}
        </div>
        </section>

      <div className="meeting-summary-brief-sections">
        {visibleSections.map((section) => (
          <ReviewBriefItemSection
            key={section.key}
            title={section.title}
            items={section.items}
            compact={compact}
            groupByProject={!compact && brief.projectMode === "multi"}
          />
        ))}
      </div>

      {!compact && <ReviewBriefReferenceSection items={brief.referenceContext} />}
      {!compact && (
        <ReviewBriefItemSection
          title="Technical Notes"
          items={brief.technicalNotes}
          compact={compact}
          groupByProject={false}
        />
      )}
    </section>
  );
}

type ReviewBriefProjectMode = "none" | "single" | "multi";
type ReviewBriefItemKind = "action" | "decision" | "question" | "risk" | "technical";

type ReviewBriefItem = {
  id: string;
  kind: ReviewBriefItemKind;
  text: string;
  project?: string;
  owner?: string;
  dueDate?: string;
  context?: string;
  sourceIds: string[];
};

type ReviewBriefCandidate = ReviewBriefItem & {
  strength: number;
  order: number;
};

type ReviewBriefReferenceItem = {
  id: string;
  kind: string;
  title: string;
  body?: string;
  citation?: string;
  sourceIds: string[];
};

type ReviewBriefModel = {
  title: string;
  executiveSummary: string;
  coverageNote: string;
  sourceCount: number;
  projectMode: ReviewBriefProjectMode;
  projectCount: number;
  primaryProject: string;
  actions: ReviewBriefItem[];
  decisions: ReviewBriefItem[];
  openQuestions: ReviewBriefItem[];
  risks: ReviewBriefItem[];
  referenceContext: ReviewBriefReferenceItem[];
  technicalNotes: ReviewBriefItem[];
};

type ReviewBriefCandidateGroups = Record<ReviewBriefItemKind, ReviewBriefCandidate[]>;

function ReviewBriefItemSection({
  title,
  items,
  compact,
  groupByProject,
}: {
  title: string;
  items: ReviewBriefItem[];
  compact: boolean;
  groupByProject: boolean;
}) {
  const visibleItems = items.slice(0, compact ? 3 : items.length);
  if (visibleItems.length === 0) {
    return null;
  }
  const groups = groupByProject ? groupReviewBriefItemsByProject(visibleItems) : [];
  return (
    <section className="meeting-summary-brief-section" aria-label={title}>
      <div className="section-heading-row">
        <h3>{title}</h3>
        <span>{visibleItems.length}</span>
      </div>
      {groups.length > 0 ? (
        <div className="meeting-summary-brief-groups">
          {groups.map((group) => (
            <section key={group.project} className="meeting-summary-brief-group">
              <h4>{formatReviewDisplayText(group.project)}</h4>
              <ReviewBriefList items={group.items} showProject={false} />
            </section>
          ))}
        </div>
      ) : (
        <ReviewBriefList items={visibleItems} showProject={!compact} />
      )}
    </section>
  );
}

function ReviewBriefList({ items, showProject }: { items: ReviewBriefItem[]; showProject: boolean }) {
  return (
    <ul className="meeting-summary-brief-list">
      {items.map((item) => (
        <li key={item.id} className="meeting-summary-brief-item">
          <span>{formatReviewDisplayText(item.text)}</span>
          {(showProject && item.project) || item.owner || item.dueDate || item.context ? (
            <div className="meeting-summary-brief-item-meta">
              {showProject && item.project && <span>{formatReviewDisplayText(item.project)}</span>}
              {item.owner && <span>Owner: {formatReviewDisplayText(item.owner)}</span>}
              {item.dueDate && <span>Due: {formatReviewDisplayText(item.dueDate)}</span>}
              {item.context && <span>{formatReviewDisplayText(item.context)}</span>}
            </div>
          ) : null}
        </li>
      ))}
    </ul>
  );
}

function ReviewBriefReferenceSection({ items }: { items: ReviewBriefReferenceItem[] }) {
  if (items.length === 0) {
    return null;
  }
  return (
    <section className="meeting-summary-context meeting-summary-reference-strip" aria-label="Reference Context">
      <div className="section-heading-row">
        <h3>Reference Context</h3>
        <span className="chip">{items.length} used</span>
      </div>
      <div className="meeting-summary-context-list">
        {items.slice(0, 6).map((item) => (
          <article key={item.id}>
            <span>{referenceContextKindLabel(item.kind)}</span>
            <strong>{formatReviewDisplayText(item.title || "Review context")}</strong>
            {item.body && <p>{formatReviewDisplayText(item.body)}</p>}
            <small>{item.sourceIds.length || 1} source{(item.sourceIds.length || 1) === 1 ? "" : "s"}</small>
          </article>
        ))}
      </div>
    </section>
  );
}

function buildReviewBriefModel(summary: SessionMemorySummary, cards: SidecarDisplayCard[] = []): ReviewBriefModel {
  let order = 0;
  const candidates: ReviewBriefCandidateGroups = {
    action: [],
    decision: [],
    question: [],
    risk: [],
    technical: [],
  };
  const sourceProjectMap = new Map<string, string>();
  const projectNames: string[] = [];
  const workstreams = summaryProjectWorkstreamList(summary.project_workstreams);
  for (const workstream of workstreams) {
    const project = cleanBriefText(workstream.project);
    if (!project) {
      continue;
    }
    projectNames.push(project);
    for (const sourceId of summaryList(workstream.source_segment_ids)) {
      sourceProjectMap.set(sourceId, project);
    }
  }
  for (const project of summaryList(summary.projects)) {
    projectNames.push(project);
  }
  const primaryProject = distinctList(projectNames)[0] ?? "";

  const push = (
    kind: ReviewBriefItemKind,
    text: unknown,
    options: {
      project?: string;
      owner?: string;
      dueDate?: string;
      context?: string;
      sourceIds?: string[];
      strength?: number;
    } = {},
  ) => {
    const value = cleanBriefText(text);
    if (!value) {
      return;
    }
    const sourceIds = distinctList(options.sourceIds ?? []);
    const project = options.project || inferProjectFromSources(sourceIds, sourceProjectMap) || (projectNames.length === 1 ? primaryProject : "");
    candidates[kind].push({
      id: `${kind}-${order}`,
      kind,
      text: value,
      project,
      owner: cleanBriefText(options.owner) || inferBriefOwner(value),
      dueDate: cleanBriefText(options.dueDate) || inferBriefDueDate(value),
      context: cleanBriefText(options.context),
      sourceIds,
      strength: options.strength ?? 20,
      order: order++,
    });
  };

  const summarySourceIds = summaryList(summary.source_segment_ids);
  const rollup = summaryPortfolioRollup(summary.portfolio_rollup, summary);
  for (const action of summaryList(rollup.bp_next_actions)) {
    push("action", action, { sourceIds: summaryList(rollup.source_segment_ids), strength: 34 });
  }
  for (const loop of summaryList(rollup.open_loops)) {
    push(reviewOpenLoopLooksLikeQuestion(loop) ? "question" : "risk", loop, { sourceIds: summaryList(rollup.source_segment_ids), strength: 28 });
  }
  for (const dependency of summaryList(rollup.cross_project_dependencies)) {
    push("risk", dependency, { sourceIds: summaryList(rollup.source_segment_ids), context: "Dependency", strength: 30 });
  }

  for (const workstream of workstreams) {
    const project = cleanBriefText(workstream.project);
    const owners = summaryList(workstream.owners);
    const owner = owners[0] ?? "";
    const sourceIds = summaryList(workstream.source_segment_ids);
    const dueDate = inferBriefDueDate(workstream.next_checkpoint ?? "");
    for (const action of summaryList(workstream.actions)) {
      push("action", action, { project, owner, dueDate, sourceIds, strength: 68 });
    }
    for (const decision of summaryList(workstream.decisions)) {
      push("decision", decision, { project, owner, sourceIds, strength: 66 });
    }
    for (const question of summaryList(workstream.open_questions)) {
      push("question", question, { project, owner, sourceIds, strength: 64 });
    }
    for (const risk of summaryList(workstream.risks)) {
      push("risk", risk, { project, owner, sourceIds, strength: 64 });
    }
  }

  for (const action of summaryList(summary.actions)) {
    push("action", action, { sourceIds: summarySourceIds, strength: 44 });
  }
  for (const decision of summaryList(summary.decisions)) {
    push("decision", decision, { sourceIds: summarySourceIds, strength: 44 });
  }
  for (const question of summaryList(summary.unresolved_questions)) {
    push("question", question, { sourceIds: summarySourceIds, strength: 44 });
  }
  for (const risk of summaryList(summary.risks)) {
    push("risk", risk, { sourceIds: summarySourceIds, strength: 44 });
  }

  for (const finding of summaryTechnicalFindingList(summary.technical_findings)) {
    const sourceIds = summaryList(finding.source_segment_ids);
    const context = cleanBriefText(finding.topic || finding.question || "Technical note");
    for (const risk of [...summaryList(finding.risks), ...summaryList(finding.data_gaps)]) {
      push("risk", risk, { sourceIds, context, strength: 42 });
    }
    for (const assumption of summaryList(finding.assumptions)) {
      push("technical", assumption, { sourceIds, context: context ? `Basis: ${context}` : "Basis", strength: 34 });
    }
    for (const method of summaryList(finding.methods)) {
      push("technical", method, { sourceIds, context: context ? `Method: ${context}` : "Method", strength: 34 });
    }
    for (const recommendation of summaryList(finding.recommendations)) {
      push("technical", recommendation, { sourceIds, context: context ? `Recommendation: ${context}` : "Recommendation", strength: 36 });
    }
    for (const reference of summaryList(finding.reference_context)) {
      push("technical", reference, { sourceIds, context: context ? `Reference: ${context}` : "Reference", strength: 26 });
    }
  }

  enrichBriefCandidatesFromCards(candidates, cards, sourceProjectMap);

  const actions = mergeBriefCandidates(candidates.action);
  const decisions = mergeBriefCandidates(candidates.decision);
  const openQuestions = mergeBriefCandidates(candidates.question);
  const risks = mergeBriefCandidates(candidates.risk)
    .filter((item) => ![...actions, ...decisions, ...openQuestions].some((canonical) => reviewBriefItemsMatch(item, canonical)));
  const canonicalItems = [...actions, ...decisions, ...openQuestions, ...risks];
  const technicalNotes = mergeBriefCandidates(candidates.technical)
    .filter((item) => !canonicalItems.some((canonical) => reviewBriefItemsMatch(item, canonical)));
  const referenceContext = buildReviewBriefReferenceContext(summary, canonicalItems, technicalNotes);
  const itemProjects = distinctList([...canonicalItems.map((item) => item.project ?? ""), ...projectNames].filter(Boolean));
  const sourceIds = distinctList([
    ...summarySourceIds,
    ...canonicalItems.flatMap((item) => item.sourceIds),
    ...technicalNotes.flatMap((item) => item.sourceIds),
    ...referenceContext.flatMap((item) => item.sourceIds),
  ]);

  return {
    title: summary.title || "Review summary",
    executiveSummary: summary.summary || summaryList(summary.key_points)[0] || "No executive summary was produced.",
    coverageNote: summaryList(summary.coverage_notes)[0] || "",
    sourceCount: sourceIds.length,
    projectMode: itemProjects.length > 1 ? "multi" : itemProjects.length === 1 ? "single" : "none",
    projectCount: itemProjects.length,
    primaryProject: itemProjects[0] ?? primaryProject,
    actions,
    decisions,
    openQuestions,
    risks,
    referenceContext,
    technicalNotes,
  };
}

function groupReviewBriefItemsByProject(items: ReviewBriefItem[]): { project: string; items: ReviewBriefItem[] }[] {
  const groups = new Map<string, ReviewBriefItem[]>();
  for (const item of items) {
    const project = item.project || "General follow-up";
    groups.set(project, [...(groups.get(project) ?? []), item]);
  }
  return Array.from(groups.entries()).map(([project, groupItems]) => ({ project, items: groupItems }));
}

function cleanBriefText(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value).trim();
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    for (const key of ["text", "description", "body", "summary", "title", "value", "item"]) {
      const text = cleanBriefText(record[key]);
      if (text) {
        return text;
      }
    }
  }
  return "";
}

function inferProjectFromSources(sourceIds: string[], sourceProjectMap: Map<string, string>): string {
  for (const sourceId of sourceIds) {
    const project = sourceProjectMap.get(sourceId);
    if (project) {
      return project;
    }
  }
  return "";
}

function inferBriefOwner(value: string): string {
  const text = cleanBriefText(value);
  const match = text.match(/^([A-Z][A-Za-z0-9&.-]*(?:\s+[A-Z][A-Za-z0-9&.-]*)?|BP)\s+(?:will|owns|needs|should|to)\b/);
  return match?.[1] ?? "";
}

function inferBriefDueDate(value: string): string {
  const text = cleanBriefText(value);
  const dayMatch = text.match(/\b(?:by|before|on|due)\s+((?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|today|tomorrow)\b)/i);
  if (dayMatch?.[1]) {
    return dayMatch[1];
  }
  const dateMatch = text.match(/\b(?:by|before|on|due)\s+([A-Z][a-z]+\.?\s+\d{1,2}(?:,\s+\d{4})?)\b/);
  if (dateMatch?.[1]) {
    return dateMatch[1];
  }
  const checkpointMatch = text.match(/^((?:next\s+)?(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)/i);
  return checkpointMatch?.[1] ?? "";
}

function reviewOpenLoopLooksLikeQuestion(value: string): boolean {
  const normalized = value.trim().toLowerCase();
  return normalized.endsWith("?") || /^(confirm|clarify|ask|check whether|determine|find out)\b/.test(normalized);
}

function enrichBriefCandidatesFromCards(
  candidates: ReviewBriefCandidateGroups,
  cards: SidecarDisplayCard[],
  sourceProjectMap: Map<string, string>,
) {
  for (const card of cards) {
    const kind = reviewBriefKindFromCard(card);
    if (!kind) {
      continue;
    }
    const text = cleanBriefText(card.summary || card.suggestedSay || card.suggestedAsk || card.title);
    if (!text) {
      continue;
    }
    const sourceIds = distinctList(card.sourceSegmentIds ?? []);
    const incoming: ReviewBriefCandidate = {
      id: `${kind}-card-${card.id}`,
      kind,
      text,
      project: inferProjectFromSources(sourceIds, sourceProjectMap),
      owner: card.owner || inferBriefOwner(text),
      dueDate: card.dueDate || inferBriefDueDate(text),
      context: "",
      sourceIds,
      strength: card.owner || card.dueDate ? 86 : 58,
      order: Number.MAX_SAFE_INTEGER,
    };
    const match = candidates[kind].find((candidate) => reviewBriefItemsMatch(candidate, incoming));
    if (match) {
      mergeReviewBriefCandidate(match, incoming);
    }
  }
}

function reviewBriefKindFromCard(card: SidecarDisplayCard): ReviewBriefItemKind | null {
  if (card.category === "action") return "action";
  if (card.category === "decision") return "decision";
  if (card.category === "question" || card.category === "clarification") return "question";
  if (card.category === "risk") return "risk";
  return null;
}

function mergeBriefCandidates(candidates: ReviewBriefCandidate[]): ReviewBriefItem[] {
  const merged: ReviewBriefCandidate[] = [];
  for (const candidate of [...candidates].sort(compareBriefCandidates)) {
    const existing = merged.find((item) => reviewBriefItemsMatch(item, candidate));
    if (existing) {
      mergeReviewBriefCandidate(existing, candidate);
    } else {
      merged.push({ ...candidate, sourceIds: distinctList(candidate.sourceIds) });
    }
  }
  return merged
    .sort((left, right) => left.order - right.order)
    .map(({ strength: _strength, order: _order, ...item }) => ({
      ...item,
      sourceIds: distinctList(item.sourceIds),
    }));
}

function compareBriefCandidates(left: ReviewBriefCandidate, right: ReviewBriefCandidate): number {
  const strengthDiff = right.strength - left.strength;
  if (strengthDiff !== 0) {
    return strengthDiff;
  }
  return left.order - right.order;
}

function mergeReviewBriefCandidate(existing: ReviewBriefCandidate, incoming: ReviewBriefCandidate) {
  const incomingBetter = incoming.strength > existing.strength
    || (incoming.strength === existing.strength && reviewBriefCompleteness(incoming) > reviewBriefCompleteness(existing));
  if (incomingBetter) {
    existing.text = incoming.text;
    existing.project = incoming.project || existing.project;
    existing.context = incoming.context || existing.context;
    existing.strength = incoming.strength;
    existing.order = Math.min(existing.order, incoming.order);
  }
  existing.owner = existing.owner || incoming.owner;
  existing.dueDate = existing.dueDate || incoming.dueDate;
  existing.project = existing.project || incoming.project;
  existing.context = existing.context || incoming.context;
  existing.sourceIds = distinctList([...existing.sourceIds, ...incoming.sourceIds]);
}

function reviewBriefCompleteness(item: ReviewBriefItem): number {
  return (item.owner ? 2 : 0) + (item.dueDate ? 2 : 0) + (item.project ? 1 : 0) + (item.context ? 1 : 0);
}

function reviewBriefItemsMatch(left: ReviewBriefItem, right: ReviewBriefItem): boolean {
  const leftText = normalizeReviewEvidenceText(left.text);
  const rightText = normalizeReviewEvidenceText(right.text);
  if (!leftText || !rightText) {
    return false;
  }
  if (leftText === rightText || leftText.includes(rightText) || rightText.includes(leftText)) {
    return true;
  }
  const leftSources = new Set(left.sourceIds);
  const rightSources = new Set(right.sourceIds);
  const sourceOverlap = leftSources.size > 0 && rightSources.size > 0 && Array.from(leftSources).some((sourceId) => rightSources.has(sourceId));
  const leftTokens = reviewEvidenceTokenSet(leftText);
  const rightTokens = reviewEvidenceTokenSet(rightText);
  const shared = Array.from(leftTokens).filter((token) => rightTokens.has(token)).length;
  const union = new Set([...leftTokens, ...rightTokens]).size || 1;
  if (sourceOverlap && shared >= 2 && shared / union >= 0.28) {
    return true;
  }
  return shared >= 4 && shared / union >= 0.5;
}

function buildReviewBriefReferenceContext(
  summary: SessionMemorySummary,
  canonicalItems: ReviewBriefItem[],
  technicalNotes: ReviewBriefItem[],
): ReviewBriefReferenceItem[] {
  const existingItems = [...canonicalItems, ...technicalNotes];
  const references: ReviewBriefReferenceItem[] = [];
  for (const item of summaryReferenceContextList(summary.reference_context)) {
    const title = cleanBriefText(item.title || "Review context");
    const body = cleanBriefText(item.body);
    if (!title && !body) {
      continue;
    }
    const sourceIds = distinctList(summaryList(item.source_segment_ids));
    const referenceText = [title, body].filter(Boolean).join(": ");
    const duplicate = existingItems.some((existing) => reviewBriefItemsMatch(existing, {
      id: "reference-check",
      kind: "technical",
      text: referenceText,
      sourceIds,
    }));
    if (duplicate && !body) {
      continue;
    }
    const normalized = normalizeReviewEvidenceText(referenceText);
    if (references.some((reference) => normalizeReviewEvidenceText([reference.title, reference.body].filter(Boolean).join(": ")) === normalized)) {
      continue;
    }
    references.push({
      id: `reference-${references.length}`,
      kind: item.kind ?? "reference",
      title,
      body,
      citation: cleanBriefText(item.citation),
      sourceIds,
    });
  }
  return references;
}

function SummaryMiniList({ label, items }: { label: string; items: string[] }) {
  const visibleItems = distinctList(items);
  if (visibleItems.length === 0) {
    return null;
  }
  return (
    <div className="meeting-summary-mini-list">
      <span>{label}</span>
      <ul>
        {visibleItems.map((item) => <li key={item}>{formatReviewDisplayText(item)}</li>)}
      </ul>
    </div>
  );
}

function SummaryWorkstreamCard({ item }: { item: SummaryProjectWorkstream }) {
  const owners = distinctList(summaryList(item.owners));
  const actionHighlights = distinctList([...summaryList(item.actions), ...summaryList(item.decisions)]);
  const sourceCount = summaryList(item.source_segment_ids).length;
  const factGroups = [
    { label: "Next action", items: actionHighlights.slice(0, 2) },
    { label: "Owner / date", items: distinctList([...owners, item.next_checkpoint ?? ""]).slice(0, 2) },
    { label: "Open item", items: summaryList(item.open_questions).slice(0, 1) },
    { label: "Risk", items: summaryList(item.risks).slice(0, 1) },
  ].filter((group) => distinctList(group.items).length > 0);
  return (
    <article>
      <div className="meeting-summary-row-head">
        <div>
          <strong>{formatReviewDisplayText(item.project || "Project / site thread")}</strong>
          {item.client_site && <span>{formatReviewDisplayText(item.client_site)}</span>}
        </div>
        <div className="meeting-summary-row-chips">
          {item.status && <span className="chip">{formatReviewDisplayText(item.status)}</span>}
          {sourceCount > 0 && <span className="chip">{sourceCount} src</span>}
        </div>
      </div>
      {factGroups.length > 0 && <SummaryFactGrid groups={factGroups} />}
    </article>
  );
}

function SummaryTechnicalCard({ item }: { item: SummaryTechnicalFinding }) {
  const sourceCount = summaryList(item.source_segment_ids).length;
  const factGroups = [
    { label: "Basis", items: distinctList([...summaryList(item.assumptions), ...summaryList(item.methods)]).slice(0, 2) },
    { label: "Finding", items: summaryList(item.findings).slice(0, 2) },
    { label: "Recommendation", items: summaryList(item.recommendations).slice(0, 1) },
    { label: "Risk / gap", items: distinctList([...summaryList(item.risks), ...summaryList(item.data_gaps)]).slice(0, 1) },
    { label: "Reference", items: summaryList(item.reference_context).slice(0, 2) },
  ].filter((group) => distinctList(group.items).length > 0);
  return (
    <article>
      <div className="meeting-summary-row-head">
        <div>
          <strong>{formatReviewDisplayText(item.topic || "Technical finding")}</strong>
          {item.question && <span>{formatReviewDisplayText(item.question)}</span>}
        </div>
        <div className="meeting-summary-row-chips">
          {item.confidence && <span className="chip">{formatReviewDisplayText(item.confidence)}</span>}
          {sourceCount > 0 && <span className="chip">{sourceCount} src</span>}
        </div>
      </div>
      {factGroups.length > 0 && <SummaryFactGrid groups={factGroups} />}
    </article>
  );
}

function SummaryFactGrid({ groups }: { groups: { label: string; items: string[] }[] }) {
  return (
    <div className="meeting-summary-fact-grid">
      {groups.map((group) => (
        <SummaryMiniList
          key={group.label}
          label={group.label}
          items={distinctList(group.items)}
        />
      ))}
    </div>
  );
}

function SummaryOverflowRows({ items }: { items: string[] }) {
  const visibleItems = distinctList(items);
  if (visibleItems.length === 0) {
    return null;
  }
  return (
    <div className="meeting-summary-overflow">
      <ul>
        {visibleItems.map((item) => <li key={item}>{formatReviewDisplayText(item)}</li>)}
      </ul>
    </div>
  );
}

function summaryWorkstreamHighlights(item: SummaryProjectWorkstream): string[] {
  return distinctList([
    ...summaryList(item.actions),
    ...summaryList(item.open_questions),
    ...summaryList(item.risks),
    ...summaryList(item.decisions),
  ]);
}

function summaryWorkstreamLine(item: SummaryProjectWorkstream): string {
  const title = item.project || "Project / site thread";
  const status = item.status ? ` (${item.status})` : "";
  const highlight = summaryWorkstreamHighlights(item)[0];
  return highlight ? `${title}${status}: ${highlight}` : `${title}${status}`;
}

function summaryTechnicalLine(item: SummaryTechnicalFinding): string {
  const title = item.topic || "Technical finding";
  const highlight = distinctList([
    ...summaryList(item.findings),
    ...summaryList(item.recommendations),
    ...summaryList(item.data_gaps),
  ])[0];
  return highlight ? `${title}: ${highlight}` : title;
}

function distinctList(items: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const item of items) {
    const trimmed = item.trim();
    const key = trimmed.toLocaleLowerCase();
    if (!trimmed || seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(trimmed);
  }
  return result;
}

function displayReviewTitle(title: string): string {
  const cleaned = title.replace(/^review[\s:_-]+/i, "").trim() || title;
  return formatReviewDisplayText(cleaned);
}

function formatReviewDisplayText(value: string | undefined | null): string {
  return String(value ?? "").replace(/[A-Za-z][A-Za-z0-9]{23,}/g, (token) => {
    if (!/[a-z][A-Z]/.test(token)) {
      return token;
    }
    const humanized = token
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2");
    const words = humanized.split(" ").filter(Boolean);
    if (humanized.length > 56 && words.length > 5) {
      return `${words.slice(0, 4).join(" ")} reference`;
    }
    return humanized;
  });
}

function summaryPortfolioRollup(value: SummaryPortfolioRollup | undefined | null, summary: SessionMemorySummary): SummaryPortfolioRollup {
  const rollup = value ?? {};
  const fallbackActions = summaryList(summary.actions);
  const fallbackOpenLoops = [...summaryList(summary.unresolved_questions), ...summaryList(summary.risks)];
  return {
    bp_next_actions: summaryList(rollup.bp_next_actions).length > 0 ? summaryList(rollup.bp_next_actions) : fallbackActions.filter((item) => /\b(bp|brandon)\b/i.test(item)).concat(fallbackActions).filter((item, index, list) => list.indexOf(item) === index),
    open_loops: summaryList(rollup.open_loops).length > 0 ? summaryList(rollup.open_loops) : fallbackOpenLoops,
    cross_project_dependencies: summaryList(rollup.cross_project_dependencies),
    risk_posture: rollup.risk_posture,
    source_segment_ids: summaryList(rollup.source_segment_ids).length > 0 ? summaryList(rollup.source_segment_ids) : summaryList(summary.source_segment_ids),
  };
}

function hasPortfolioRollup(rollup: SummaryPortfolioRollup): boolean {
  return Boolean(
    summaryList(rollup.bp_next_actions).length
    || summaryList(rollup.open_loops).length
    || summaryList(rollup.cross_project_dependencies).length
    || rollup.risk_posture
  );
}

function summaryReviewMetrics(value: SummaryReviewMetrics | undefined | null, summary: SessionMemorySummary): SummaryReviewMetrics {
  const metrics = value ?? {};
  return {
    workstream_count: numericMetric(metrics.workstream_count, summaryProjectWorkstreamList(summary.project_workstreams).length),
    action_count: numericMetric(metrics.action_count, summaryList(summary.actions).length),
    risk_count: numericMetric(metrics.risk_count, summaryList(summary.risks).length),
    technical_finding_count: numericMetric(metrics.technical_finding_count, summaryTechnicalFindingList(summary.technical_findings).length),
    source_count: numericMetric(metrics.source_count, summaryList(summary.source_segment_ids).length),
    source_coverage: typeof metrics.source_coverage === "number" ? metrics.source_coverage : undefined,
    time_span_coverage: typeof metrics.time_span_coverage === "number" ? metrics.time_span_coverage : undefined,
    context_kinds: Array.isArray(metrics.context_kinds) && metrics.context_kinds.length > 0
      ? metrics.context_kinds
      : summaryReferenceContextList(summary.reference_context).map((item) => item.kind ?? "context").filter(Boolean),
  };
}

function numericMetric(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function reviewMetricChips(metrics: SummaryReviewMetrics): { label: string; value: string }[] {
  const chips = [
    { label: "Threads", value: formatMetricNumber(metrics.workstream_count) },
    { label: "Actions", value: formatMetricNumber(metrics.action_count) },
    { label: "Risks", value: formatMetricNumber(metrics.risk_count) },
    { label: "Technical", value: formatMetricNumber(metrics.technical_finding_count) },
    { label: "Sources", value: formatMetricNumber(metrics.source_count) },
  ].filter((item) => item.value !== "0");
  if (typeof metrics.time_span_coverage === "number") {
    chips.push({ label: "Time coverage", value: formatPercent(metrics.time_span_coverage) });
  }
  if (typeof metrics.source_coverage === "number") {
    chips.push({ label: "Source coverage", value: formatPercent(metrics.source_coverage) });
  }
  const contextKinds = Array.isArray(metrics.context_kinds) ? metrics.context_kinds.filter(Boolean) : [];
  if (contextKinds.length > 0) {
    chips.push({ label: "Context", value: String(contextKinds.length) });
  }
  return chips;
}

type ReviewSummaryEvidenceItem = {
  text: string;
  sourceIds: Set<string>;
};

type ReviewValidationEvidence = {
  card: SidecarDisplayCard;
  duplicateOfSummary: boolean;
  reasons: string[];
  sourceCount: number;
};

function ReviewValidationEvidenceCard({
  item,
  expanded,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onCopy,
}: {
  item: ReviewValidationEvidence;
  expanded: boolean;
  showDebugMetadata: boolean;
  onToggle: () => void;
  onJumpToEvidence: () => void;
  onCopy: () => void;
}) {
  const { card, reasons, sourceCount } = item;
  const hasEvidence = Boolean(card.evidenceQuote?.trim());
  const hasSourceSegments = (card.sourceSegmentIds?.length ?? 0) > 0;
  const hasCopyableText = Boolean(card.evidenceQuote || card.suggestedSay || card.suggestedAsk || card.summary);
  const hasDebugDetails = showDebugMetadata && Boolean(card.rawText || card.debugMetadata);
  const confidenceLabel = reviewConfidenceLabel(card);
  const visibleReasons = reasons.length > 0 ? reasons : [item.duplicateOfSummary ? "Summary check" : "Additional claim"];
  return (
    <article className={`context-card field-bubble compact review-validation-card ${confidenceClass(card)} ${card.category}`} aria-label={`${categoryLabel(card.category)} review validation card`}>
      <div className="context-card-head">
        <span>{categoryLabel(card.category)}</span>
        <time>{formatClock(card.at)}</time>
      </div>
      <h3>{formatReviewDisplayText(card.title)}</h3>
      <div className="context-card-meta review-validation-meta">
        {visibleReasons.slice(0, 3).map((reason) => <span key={reason}>{reason}</span>)}
        {card.qualityLabel && <span className="quality-chip">{card.qualityLabel}</span>}
        {confidenceLabel && <span>{confidenceLabel}</span>}
        <span>{sourceCount} source{sourceCount === 1 ? "" : "s"}</span>
        {card.owner && <span>Owner: {card.owner}</span>}
        {card.dueDate && <span>Due: {card.dueDate}</span>}
      </div>
      {hasEvidence ? (
        <blockquote className="evidence-quote review-validation-quote" aria-label={`Evidence for ${card.title}`}>
          {formatReviewDisplayText(card.evidenceQuote)}
        </blockquote>
      ) : (
        <p className="review-validation-claim">{card.missingInfo ? "Validation needs missing context." : "Validation needs transcript support."}</p>
      )}
      {card.missingInfo && <p className="missing-info"><strong>Missing info:</strong> {formatReviewDisplayText(card.missingInfo)}</p>}
      <div className="card-detail-actions">
        {hasDebugDetails && (
          <button className="link-button" type="button" onClick={onToggle}>
            {expanded ? "Hide debug" : "Debug"}
          </button>
        )}
        {hasSourceSegments && (
          <button className="link-button" type="button" onClick={onJumpToEvidence}>
            Jump to source{card.sourceSegmentIds!.length > 1 ? ` (${card.sourceSegmentIds!.length})` : ""}
          </button>
        )}
        {hasCopyableText && (
          <button className="link-button" type="button" onClick={onCopy}>
            {hasEvidence ? "Copy quote" : "Copy text"}
          </button>
        )}
      </div>
      {expanded && hasDebugDetails && (
        <div className="field-details">
          {card.summary && <p className="detail-note"><strong>Claim:</strong> {formatReviewDisplayText(card.summary)}</p>}
          {card.whyRelevant && <p className="context-card-why">{formatReviewDisplayText(card.whyRelevant)}</p>}
          {hasSourceSegments && (
            <div className="source-segment-list">
              {card.sourceSegmentIds!.slice(0, 8).map((segmentId) => (
                <button key={segmentId} type="button" className="source-segment-chip" onClick={onJumpToEvidence}>
                  {segmentId}
                </button>
              ))}
            </div>
          )}
          {hasDebugDetails && (
            <details className="debug-details" open>
              <summary>Debug metadata</summary>
              <dl>
                {debugEntries(card).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{String(value)}</dd>
                  </div>
                ))}
              </dl>
            </details>
          )}
        </div>
      )}
    </article>
  );
}

function reviewValidationEvidenceCards(summary: SessionMemorySummary | null, cards: SidecarDisplayCard[]): ReviewValidationEvidence[] {
  const summaryItems = reviewSummaryEvidenceItems(summary);
  const summaryText = normalizeReviewEvidenceText(summaryItems.map((item) => item.text).join(" "));
  return cards
    .map((card) => {
      const duplicateOfSummary = reviewCardMatchesSummary(card, summaryItems);
      const sourceCount = distinctList(card.sourceSegmentIds ?? []).length;
      const confidence = card.confidence ?? card.score;
      const reasons: string[] = [];
      if (card.missingInfo) {
        reasons.push("Missing info");
      }
      if (confidence != null && confidence < 0.72) {
        reasons.push("Low confidence");
      }
      if (sourceCount === 0) {
        reasons.push("Unsupported");
      }
      if (card.owner && !summaryText.includes(normalizeReviewEvidenceText(card.owner))) {
        reasons.push("Owner check");
      }
      if (card.dueDate && !summaryText.includes(normalizeReviewEvidenceText(card.dueDate))) {
        reasons.push("Date check");
      }
      if (!duplicateOfSummary) {
        reasons.push("Additional claim");
      }
      return {
        card,
        duplicateOfSummary,
        reasons: distinctList(reasons),
        sourceCount,
      };
    })
    .filter((item) => item.reasons.length > 0 || !item.duplicateOfSummary)
    .sort(compareReviewValidationEvidence);
}

function compareReviewValidationEvidence(left: ReviewValidationEvidence, right: ReviewValidationEvidence): number {
  const score = (item: ReviewValidationEvidence) => {
    let value = 0;
    if (item.reasons.includes("Unsupported")) value += 80;
    if (item.reasons.includes("Missing info")) value += 70;
    if (item.reasons.includes("Low confidence")) value += 50;
    if (item.reasons.includes("Owner check") || item.reasons.includes("Date check")) value += 35;
    if (item.reasons.includes("Transcript quote")) value += 20;
    return value;
  };
  const scoreDiff = score(right) - score(left);
  if (scoreDiff !== 0) {
    return scoreDiff;
  }
  return left.card.at - right.card.at;
}

function reviewCardMatchesSummary(card: SidecarDisplayCard, summaryItems: ReviewSummaryEvidenceItem[]): boolean {
  if (summaryItems.length === 0) {
    return false;
  }
  const cardTexts = [card.title, card.summary, card.suggestedSay, card.suggestedAsk, card.evidenceQuote]
    .map((text) => normalizeReviewEvidenceText(text ?? ""))
    .filter((text) => text.length >= 8);
  const cardSources = new Set(card.sourceSegmentIds ?? []);
  for (const item of summaryItems) {
    const itemText = normalizeReviewEvidenceText(item.text);
    if (itemText.length < 8) {
      continue;
    }
    const sourceOverlap = cardSources.size > 0
      && item.sourceIds.size > 0
      && Array.from(cardSources).some((sourceId) => item.sourceIds.has(sourceId));
    if (sourceOverlap && cardTexts.some((text) => reviewEvidenceTextMatches(text, itemText, 0.42))) {
      return true;
    }
    if (!sourceOverlap && cardTexts.some((text) => reviewEvidenceTextDirectMatch(text, itemText))) {
      return true;
    }
  }
  return false;
}

function reviewEvidenceTextDirectMatch(left: string, right: string): boolean {
  if (left.length < 16 || right.length < 16) {
    return false;
  }
  if (Math.min(reviewEvidenceTokenSet(left).size, reviewEvidenceTokenSet(right).size) < 4) {
    return false;
  }
  return left.includes(right) || right.includes(left);
}

function reviewEvidenceTextMatches(left: string, right: string, threshold: number): boolean {
  if (!left || !right) {
    return false;
  }
  if (left.includes(right) || right.includes(left)) {
    return true;
  }
  const leftTokens = reviewEvidenceTokenSet(left);
  const rightTokens = reviewEvidenceTokenSet(right);
  if (leftTokens.size === 0 || rightTokens.size === 0) {
    return false;
  }
  const shared = Array.from(leftTokens).filter((token) => rightTokens.has(token)).length;
  const union = new Set([...leftTokens, ...rightTokens]).size;
  const minShared = threshold >= 0.7 ? 5 : 2;
  return shared >= minShared && shared / union >= threshold;
}

function reviewEvidenceTokenSet(value: string): Set<string> {
  return new Set(value.split(" ").filter((token) => token.length > 2));
}

function reviewSummaryEvidenceItems(summary: SessionMemorySummary | null): ReviewSummaryEvidenceItem[] {
  if (!summary) {
    return [];
  }
  const items: ReviewSummaryEvidenceItem[] = [];
  const push = (text: string | undefined | null, sourceIds: string[] = []) => {
    const value = String(text ?? "").trim();
    if (!value) {
      return;
    }
    items.push({ text: value, sourceIds: new Set(sourceIds.filter(Boolean)) });
  };
  const summarySourceIds = summaryList(summary.source_segment_ids);
  push(summary.title, summarySourceIds);
  push(summary.summary, summarySourceIds);
  for (const value of [
    ...summaryList(summary.key_points),
    ...summaryList(summary.actions),
    ...summaryList(summary.decisions),
    ...summaryList(summary.unresolved_questions),
    ...summaryList(summary.risks),
    ...summaryList(summary.coverage_notes),
  ]) {
    push(value, summarySourceIds);
  }
  const rollup = summaryPortfolioRollup(summary.portfolio_rollup, summary);
  const rollupSourceIds = summaryList(rollup.source_segment_ids);
  for (const value of [
    ...summaryList(rollup.bp_next_actions),
    ...summaryList(rollup.open_loops),
    ...summaryList(rollup.cross_project_dependencies),
  ]) {
    push(value, rollupSourceIds);
  }
  for (const item of summaryProjectWorkstreamList(summary.project_workstreams)) {
    const sourceIds = summaryList(item.source_segment_ids);
    push(item.project, sourceIds);
    push(item.client_site, sourceIds);
    push(item.status, sourceIds);
    push(item.next_checkpoint, sourceIds);
    for (const value of [
      ...summaryList(item.actions),
      ...summaryList(item.decisions),
      ...summaryList(item.risks),
      ...summaryList(item.open_questions),
      ...summaryList(item.owners),
    ]) {
      push(value, sourceIds);
    }
  }
  for (const item of summaryTechnicalFindingList(summary.technical_findings)) {
    const sourceIds = summaryList(item.source_segment_ids);
    push(item.topic, sourceIds);
    push(item.question, sourceIds);
    push(item.confidence, sourceIds);
    for (const value of [
      ...summaryList(item.assumptions),
      ...summaryList(item.methods),
      ...summaryList(item.findings),
      ...summaryList(item.recommendations),
      ...summaryList(item.risks),
      ...summaryList(item.data_gaps),
      ...summaryList(item.reference_context),
    ]) {
      push(value, sourceIds);
    }
  }
  for (const item of summaryReferenceContextList(summary.reference_context)) {
    const sourceIds = summaryList(item.source_segment_ids);
    push(item.title, sourceIds);
    push(item.body, sourceIds);
    push(item.citation, sourceIds);
  }
  return items;
}

function normalizeReviewEvidenceText(value: string): string {
  return value
    .toLocaleLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function formatMetricNumber(value: number | undefined): string {
  return Number.isFinite(value) ? String(Math.max(0, Number(value))) : "0";
}

function formatPercent(value: number): string {
  const normalized = value <= 1 ? value * 100 : value;
  return `${Math.round(Math.max(0, Math.min(100, normalized)))}%`;
}

function riskPostureLabel(value: string): string {
  return value.replace(/_/g, " ");
}

function reviewConfidenceLabel(card: SidecarDisplayCard): string | null {
  const confidence = card.confidence ?? card.score;
  if (!Number.isFinite(confidence)) {
    return null;
  }
  const normalized = confidence! <= 1 ? confidence! * 100 : confidence!;
  return `${Math.round(Math.max(0, Math.min(100, normalized)))}% confidence`;
}

function ReviewContextCard({
  card,
  expanded,
  showDebugMetadata,
  onToggle,
  onJumpToEvidence,
  onCopy,
}: {
  card: SidecarDisplayCard;
  expanded: boolean;
  showDebugMetadata: boolean;
  onToggle: () => void;
  onJumpToEvidence: () => void;
  onCopy: () => void;
}) {
  const hasEvidence = Boolean(card.evidenceQuote?.trim());
  const hasSourceSegments = (card.sourceSegmentIds?.length ?? 0) > 0;
  const hasDebugDetails = showDebugMetadata && Boolean(card.rawText || card.debugMetadata);
  const suggestion = card.suggestedSay ?? card.suggestedAsk;
  return (
    <article className={`context-card field-bubble compact ${confidenceClass(card)} ${card.category}`} aria-label={`${categoryLabel(card.category)} review card`}>
      <div className="context-card-head">
        <span>{categoryLabel(card.category)}</span>
        <time>{formatClock(card.at)}</time>
      </div>
      <h3>{formatReviewDisplayText(card.title)}</h3>
      <div className="context-card-meta">
        {card.qualityLabel && <span className="quality-chip">{card.qualityLabel}</span>}
        {card.owner && <span>Owner: {card.owner}</span>}
        {card.dueDate && <span>Due: {card.dueDate}</span>}
      </div>
      <p>{formatReviewDisplayText(card.summary)}</p>
      {suggestion && (
        <div className="suggestion-row">
          <strong>{card.suggestedAsk ? "Ask this" : "Say this"}</strong>
          <span>{formatReviewDisplayText(suggestion)}</span>
          <button type="button" className="link-button" onClick={onCopy}>Copy</button>
        </div>
      )}
      {card.whyRelevant && <p className="context-card-why">{formatReviewDisplayText(card.whyRelevant)}</p>}
      {hasEvidence && <blockquote className="evidence-quote">{formatReviewDisplayText(card.evidenceQuote)}</blockquote>}
      {card.missingInfo && <p className="missing-info"><strong>Missing info:</strong> {formatReviewDisplayText(card.missingInfo)}</p>}
      {(hasSourceSegments || hasDebugDetails) && (
        <>
          <div className="card-detail-actions">
            {hasSourceSegments && (
              <button className="link-button" type="button" onClick={onJumpToEvidence}>
                Jump to source{card.sourceSegmentIds!.length > 1 ? ` (${card.sourceSegmentIds!.length})` : ""}
              </button>
            )}
            {hasDebugDetails && (
              <button className="link-button" type="button" onClick={onToggle}>
                {expanded ? "Hide debug" : "Debug"}
              </button>
            )}
          </div>
          {hasSourceSegments && (
            <div className="source-segment-list review-inline-source-list">
              {card.sourceSegmentIds!.slice(0, 6).map((segmentId) => (
                <button key={segmentId} type="button" className="source-segment-chip" onClick={onJumpToEvidence}>
                  {segmentId}
                </button>
              ))}
            </div>
          )}
          {expanded && hasDebugDetails && (
            <div className="field-details">
              <details className="debug-details" open>
                <summary>Debug metadata</summary>
                <dl>
                  {debugEntries(card).map(([key, value]) => (
                    <div key={key}>
                      <dt>{key}</dt>
                      <dd>{String(value)}</dd>
                    </div>
                  ))}
                </dl>
              </details>
            </div>
          )}
        </>
      )}
    </article>
  );
}

function summaryList(value: unknown[] | undefined | null): string[] {
  return Array.isArray(value) ? value.map(cleanBriefText).filter(Boolean) : [];
}

function summaryProjectWorkstreamList(value: SummaryProjectWorkstream[] | undefined | null): SummaryProjectWorkstream[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item) => Boolean(item?.project?.trim()));
}

function summaryTechnicalFindingList(value: SummaryTechnicalFinding[] | undefined | null): SummaryTechnicalFinding[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item) => Boolean(item?.topic?.trim() || item?.question?.trim()));
}

function summaryReferenceContextList(value: SummaryReferenceContext[] | undefined | null): SummaryReferenceContext[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item) => Boolean(item?.title?.trim() || item?.body?.trim()));
}

function referenceContextKindLabel(kind: string | undefined): string {
  switch ((kind ?? "").toLowerCase()) {
    case "energy_lens":
      return "Energy Lens";
    case "ee_reference":
      return "EE Index";
    case "brave_web":
      return "Current Public Web";
    default:
      return "Reference";
  }
}

export function buildReviewBriefMarkdown(summary: SessionMemorySummary | null, cards: SidecarDisplayCard[]): string {
  if (!summary) {
    return buildConsultingBriefMarkdown(cards, "Review Brief");
  }
  const brief = buildReviewBriefModel(summary, cards);
  const sections: string[] = [`# ${brief.title || "Review Brief"}`, "", "## Executive Summary", "", brief.executiveSummary];
  const appendItems = (title: string, items: ReviewBriefItem[]) => {
    if (items.length === 0) {
      return;
    }
    sections.push("", `## ${title}`);
    for (const item of items) {
      const metadata = [
        item.project,
        item.owner ? `Owner: ${item.owner}` : "",
        item.dueDate ? `Due: ${item.dueDate}` : "",
        item.context,
      ].filter(Boolean);
      sections.push(`- ${item.text}${metadata.length > 0 ? ` (${metadata.join("; ")})` : ""}`);
    }
  };
  appendItems("Actions", brief.actions);
  appendItems("Decisions", brief.decisions);
  appendItems("Open Questions", brief.openQuestions);
  appendItems("Risks / Watch Items", brief.risks);
  if (brief.referenceContext.length > 0) {
    sections.push("", "## Reference Context");
    for (const item of brief.referenceContext) {
      const body = item.body ? `: ${item.body}` : "";
      const citation = item.citation ? ` (${item.citation})` : "";
      sections.push(`- ${referenceContextKindLabel(item.kind)} - ${item.title}${body}${citation}`);
    }
  }
  appendItems("Technical Notes", brief.technicalNotes);
  return `${sections.join("\n").trim()}\n`;
}

function reviewOverallProgress(job: ReviewJobResponse | null, busy: string): number {
  if (job?.progress_pct != null) {
    return clampProgress(job.progress_pct);
  }
  if (job?.progress_percent != null) {
    return clampProgress(job.progress_percent);
  }
  if (busy === "uploading") {
    return 3;
  }
  return 0;
}

export function isReviewTerminal(status: string): boolean {
  return status === "approved" || status === "discarded" || status === "error" || status === "canceled";
}

export function isReviewDefaultSelectable(job: ReviewJobResponse): boolean {
  return job.status === "completed_awaiting_validation" || !isReviewTerminal(job.status);
}

export function reviewJobId(job: ReviewJobResponse): string {
  return job.id || job.job_id;
}

export function compareReviewJobs(left: ReviewJobResponse, right: ReviewJobResponse): number {
  const rank = (job: ReviewJobResponse) => {
    if (job.status === "completed_awaiting_validation") return 0;
    if (!isReviewTerminal(job.status)) return 1;
    return 2;
  };
  const rankDiff = rank(left) - rank(right);
  if (rankDiff !== 0) return rankDiff;
  const leftQueue = left.queue_position ?? 9999;
  const rightQueue = right.queue_position ?? 9999;
  if (leftQueue !== rightQueue) return leftQueue - rightQueue;
  return (right.updated_at ?? 0) - (left.updated_at ?? 0);
}

function reviewActiveStep(job: ReviewJobResponse | null): ReviewStep | null {
  if (!job) {
    return null;
  }
  return job.steps.find((step) => step.status === "running") ?? job.steps.find((step) => step.status === "canceled") ?? null;
}

function reviewElapsedLabel(job: ReviewJobResponse | null): string {
  const elapsed = job?.elapsed_seconds;
  if (elapsed == null || !Number.isFinite(elapsed) || elapsed < 1) {
    return "";
  }
  return `${formatPlaybackClock(elapsed)} elapsed`;
}

function reviewStepProgress(step: ReviewStep): number {
  if (step.progress != null) {
    return clampProgress(step.progress);
  }
  if (step.status === "completed") {
    return 100;
  }
  if (step.status === "running") {
    return 40;
  }
  return 0;
}

function clampProgress(value: number): number {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.min(100, Math.round(value)));
}

function reviewAsrLabel(job: ReviewJobResponse | null): string {
  const backend = reviewAsrBackendLabel(job?.asr_backend ?? "");
  if (job?.asr_model) {
    return `${backend} · ${job.asr_model}`;
  }
  return `${backend} · high accuracy`;
}

function reviewSpeakerIdentityLabel(job: ReviewJobResponse | null): string {
  const diagnostics = job?.diagnostics?.speaker_identity;
  if (!diagnostics || typeof diagnostics !== "object") {
    return "";
  }
  const values = diagnostics as Record<string, unknown>;
  const label = typeof values.profile_label === "string" && values.profile_label.trim() ? values.profile_label.trim() : "BP";
  const ready = values.ready === true;
  const bp = numberOrUndefined(values.bp_segment_count) ?? 0;
  const other = numberOrUndefined(values.other_segment_count) ?? 0;
  const unknown = numberOrUndefined(values.unknown_segment_count) ?? 0;
  return `${label} ${ready ? "ready" : "not ready"} · ${bp} BP / ${other + unknown} other`;
}

type ReviewUsefulness = {
  status: string;
  label: string;
  tone: "good" | "warning" | "danger" | "";
  scoreLabel: string;
  flagsLabel: string;
};

function reviewUsefulness(job: ReviewJobResponse | null): ReviewUsefulness | null {
  const diagnostics = reviewDiagnostics(job);
  if (!diagnostics) {
    return null;
  }
  const status = stringOrUndefined(diagnostics.usefulness_status) ?? stringOrUndefined(diagnostics.summary_usefulness_status) ?? "";
  const score = numberOrUndefined(diagnostics.usefulness_score);
  const flags = arrayOfStrings(diagnostics.usefulness_flags ?? diagnostics.summary_usefulness_flags);
  if (!status && score == null && flags.length === 0) {
    return null;
  }
  const label = status === "low_usefulness"
    ? "Needs validation"
    : status === "repaired"
      ? "Summary repaired"
      : status === "passed"
        ? "Summary useful"
        : status === "needs_repair"
          ? "Summary weak"
          : "Usefulness checked";
  const tone = status === "low_usefulness" || status === "needs_repair" ? "danger" : status === "repaired" ? "warning" : "good";
  const scoreLabel = score != null ? `Usefulness ${Math.round(score * 100)}%` : "";
  const flagsLabel = flags.length > 0 ? `${flags.length} usefulness flag${flags.length === 1 ? "" : "s"}` : "";
  return { status, label, tone, scoreLabel, flagsLabel };
}

function reviewDiagnostics(job: ReviewJobResponse | null): Record<string, unknown> | null {
  const jobDiagnostics = job?.diagnostics;
  const summaryDiagnostics = job?.summary?.diagnostics;
  if (summaryDiagnostics && typeof summaryDiagnostics === "object") {
    return { ...(jobDiagnostics ?? {}), ...summaryDiagnostics };
  }
  return jobDiagnostics ?? null;
}

function reviewAsrBackendLabel(value: string): string {
  const normalized = value.trim().toLowerCase();
  if (normalized === "nemo") return "NeMo ASR";
  if (normalized === "faster_whisper") return "Faster-Whisper";
  if (normalized === "fake_review_asr") return "Review ASR";
  return "NeMo ASR";
}

function reviewFeedback(job: ReviewJobResponse | null, busy: string, error: string): { title: string; detail: string } {
  const activeError = error || job?.error || "";
  if (activeError) {
    return { title: "Review blocked", detail: activeError };
  }
  if (busy === "uploading") {
    return { title: "Uploading audio", detail: "Preparing the review job." };
  }
  if (!job) {
    return { title: "Ready for audio", detail: "Upload audio or stop Live to queue high-accuracy review." };
  }
  if (job.status === "completed_awaiting_validation") {
    const usefulness = reviewUsefulness(job);
    if (usefulness?.status === "low_usefulness" || usefulness?.status === "needs_repair") {
      return {
        title: "Needs manual validation",
        detail: `${job.corrected_segment_count} clean transcript segments are ready, but the summary failed the usefulness gate. Validate transcript evidence before approving.`,
      };
    }
    return { title: "Ready to validate", detail: `${job.corrected_segment_count} clean transcript segments and ${job.meeting_cards.length} meeting cards. Approve to save.` };
  }
  if (job.status === "approved") {
    return { title: "Approved", detail: "Saved to Sessions and temporary audio cleanup has run." };
  }
  if (job.status === "discarded") {
    return { title: "Discarded", detail: "No Session was created and temporary audio cleanup has run." };
  }
  if (job.status === "canceled") {
    return { title: "Review canceled", detail: "The current review run was stopped." };
  }
  if (job.status === "conditioning" || job.status === "running_asr") {
    return { title: "Conditioning audio", detail: "Converting the upload into a clean mono transcription file." };
  }
  if (job.status === "reviewing" || job.status === "running_cards") {
    return { title: "Transcript correction running", detail: "Ollama is checking segments for obvious ASR errors." };
  }
  if (job.status === "evaluating" || job.status === "running_summary") {
    return { title: "Meeting evaluation running", detail: "Generating evidence-backed action items, notes, and decisions." };
  }
  return { title: reviewStatusLabel(job.status), detail: "Review is queued." };
}

function reviewStepStatusLabel(status: string): string {
  if (status === "running") return "Running";
  if (status === "completed") return "Done";
  if (status === "error") return "Error";
  if (status === "canceled") return "Canceled";
  return "Pending";
}

function reviewStatusLabel(status: string): string {
  if (status === "completed_awaiting_validation") return "Awaiting validation";
  if (status === "approved") return "Approved";
  if (status === "discarded") return "Discarded";
  if (status === "paused_for_live") return "Paused for Live";
  if (status === "running_asr") return "ASR";
  if (status === "running_cards") return "Reviewing";
  if (status === "running_summary") return "Evaluating";
  if (status === "completed") return "Complete";
  if (status === "error") return "Error";
  if (status === "canceled") return "Canceled";
  if (status === "queued") return "Queued";
  if (status === "conditioning") return "Conditioning";
  if (status === "transcribing") return "ASR";
  if (status === "reviewing") return "Reviewing";
  if (status === "evaluating") return "Evaluating";
  return status;
}

function categoryLabel(category: SidecarDisplayCard["category"]): string {
  if (category === "web") return "Web";
  if (category === "action") return "Follow-up";
  if (category === "decision") return "Decision";
  if (category === "question") return "Ask this";
  if (category === "risk") return "Watch risk";
  if (category === "clarification") return "Clarify";
  if (category === "contribution") return "Say this";
  if (category === "status") return "Status";
  if (category === "memory") return "Memory";
  if (category === "reference") return "Reference";
  return "Note";
}

function confidenceClass(card: SidecarDisplayCard): string {
  const confidence = card.confidence ?? card.score;
  if (confidence === undefined) {
    return "";
  }
  if (confidence >= 0.82) {
    return "high-confidence";
  }
  if (confidence < 0.58) {
    return "low-confidence";
  }
  return "";
}

function debugEntries(card: SidecarDisplayCard): [string, unknown][] {
  const metadata = card.debugMetadata ?? {};
  const entries: [string, unknown][] = [
    ["source type", metadata.sourceType],
    ["source ID", metadata.sourceId],
    ["score", metadata.score ?? card.score],
    ["confidence", metadata.confidence ?? card.confidence],
    ["retrieval reason", metadata.retrievalReason],
    ["backend label", metadata.backendLabel],
    ["card key", metadata.cardKey ?? card.cardKey],
    ["source segments", metadata.sourceSegmentIds],
  ];
  return entries.filter(([, value]) => value !== undefined && value !== null && value !== "");
}

function numberOrUndefined(value: unknown): number | undefined {
  const number = Number(value);
  return Number.isFinite(number) ? number : undefined;
}

function stringOrUndefined(value: unknown): string | undefined {
  const text = String(value ?? "").trim();
  return text ? text : undefined;
}

function arrayOfStrings(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
}

function formatClock(value: number): string {
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function formatDateTime(value: number | null | undefined): string {
  if (!value) {
    return "unknown time";
  }
  return new Date(value * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatSeconds(value: number): string {
  return `${Number(value).toFixed(1)}s`;
}

function formatSecondsFromMs(value: number | undefined): string {
  return formatSeconds((value ?? 0) / 1000);
}

function formatPlaybackClock(value: number): string {
  const totalSeconds = Math.max(0, Math.round(Number.isFinite(value) ? value : 0));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}
