//! Diagnostics: attributing log events to sessions, and classifying them.
//!
//! Every part of the engine, and every dependency through the `log` crate,
//! reports problems as `tracing` events. This module turns those events into
//! something that can fail a test or a smoke run (design 0001, "Sessions"):
//!
//! - A [`Session`] is one run of the engine, or one test, identified by a
//!   [`SessionId`]. Code runs inside the session's span, and the
//!   [`SessionLayer`] records which spans belong to which session, so that each
//!   event can be *attributed* to a session: by the event's own `session`
//!   field if it has one, otherwise by the nearest enclosing session span.
//! - [`classify`] decides, from an event's level, target, message ID and
//!   message, whether it is [`Class::Counted`] (and so fails tests), or one of the
//!   classes that are only logged.
//! - [`DiagnosticCounter`] counts counted events for the application. Tests
//!   use `TestCapture` and `TestSession` instead, in the test-only `capture`
//!   submodule, which fail the test that caused each event.
//!
//! # Logging levels are a contract
//!
//! Failing tests on warnings works only if the levels mean something definite:
//!
//! | Level | Meaning |
//! |---|---|
//! | `ERROR` | An operation failed, or a bug was detected |
//! | `WARN` | Something unexpected happened that someone should act on |
//! | `INFO` | Normal, notable events, including expected limitations |
//! | `DEBUG`, `TRACE` | Detail for diagnosing problems |
//!
//! Code that logs a normal condition at `WARN` is a bug, fixed by changing its
//! level, not by suppressing the message.
//!
//! # Filtering per layer
//!
//! Each layer built here carries its own filter, rather than relying on one
//! global filter. `tracing` disables a call site entirely when no layer wants
//! it, so a global `RUST_LOG=error` would otherwise turn off every `warn!`, and
//! warnings would go uncounted. The layers' constructors ([`SessionLayer::layer`]
//! and [`DiagnosticCounter::layer`]) apply their filters, so they can't be
//! installed without them.

#[cfg(test)]
pub mod capture;

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use tracing::field::{Field, Visit};
use tracing::level_filters::LevelFilter;
use tracing::span::Attributes;
use tracing::{Event, Level, Metadata, Span, Subscriber, info_span};
use tracing_subscriber::Layer;
use tracing_subscriber::filter::{FilterFn, filter_fn};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;

/// The field that attributes an event to a session explicitly, and that marks
/// a span as a session's span. Its value is a [`SessionId::as_u64`].
pub const SESSION_FIELD: &str = "session";

/// The field holding an event's message ID, such as a validation message's
/// VUID, which [`SUPPRESSED`] entries match with [`MessageMatch::Id`].
pub const MESSAGE_ID_FIELD: &str = "message_id";

/// The target of the graphics API's validation messages. Its best-practices
/// messages are [`Class::Advisory`].
pub const VALIDATION_TARGET: &str = "clockwork::validation";

/// The prefix of best-practices validation message IDs.
const BEST_PRACTICES_PREFIX: &str = "BestPractices-";

/// Identifies one session (a run of the engine, or a test) in log events.
/// Unique within the process.
///
/// It's a number from a process-wide counter, not a resource handle: handles
/// don't exist yet when the first messages about them arrive, and their values
/// can be reused once the resource is freed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SessionId(u64);

impl SessionId {
    /// Returns an ID that no other call in this process has returned.
    pub fn unique() -> Self {
        /// The next ID to return. Starts at 1, so that 0 never identifies a
        /// session, which helps to spot a missing or zeroed field.
        static NEXT: AtomicU64 = AtomicU64::new(1);
        // Only uniqueness matters, not ordering with other memory.
        Self(NEXT.fetch_add(1, Ordering::Relaxed))
    }

    /// The ID as an integer, for event fields and for passing through C
    /// callbacks (`pUserData`).
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for SessionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A session and its span. Events emitted inside the span are attributed to
/// it.
///
/// The session owns its span. Entering it returns a guard that must stay on
/// the thread that entered it. Other threads that do work for the session
/// enter a clone of [`span`](Self::span) themselves.
///
/// Create sessions only after the subscriber is installed: a span created
/// with no subscriber is disabled, and attributes nothing.
#[derive(Debug)]
pub struct Session {
    /// The session's ID, recorded in its span's [`SESSION_FIELD`].
    id: SessionId,
    /// The `session` span, at `INFO`. [`SessionLayer`] recognises it by its
    /// [`SESSION_FIELD`].
    span: Span,
}

impl Session {
    /// Creates a session with a new ID.
    pub fn new() -> Self {
        let id = SessionId::unique();
        let span = info_span!("session", session = id.as_u64());
        Self { id, span }
    }

    /// The session's ID.
    pub fn id(&self) -> SessionId {
        self.id
    }

    /// The session's span, for entering on this thread or propagating to
    /// others.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

/// How an event affects tests and the application's warning count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    /// A warning or error that isn't one of the classes below. Fails tests,
    /// and is counted by the application.
    Counted,
    /// A warning or error listed in [`SUPPRESSED`]. Logged at its level, but
    /// neither counted nor failing.
    Suppressed,
    /// A best-practices validation message. Logged, but neither counted nor
    /// failing.
    Advisory,
    /// `INFO` and below. Logged only.
    Ignored,
}

/// A message that is logged, but neither counted nor failing, and why.
#[derive(Debug)]
pub struct Suppression {
    /// The event's target, matched exactly.
    pub target: &'static str,
    /// Which message on that target is suppressed.
    pub message: MessageMatch,
    /// Why the message is harmless, with a link to an upstream issue where
    /// there is one. Never empty.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "only read by the test that checks every entry has one"
        )
    )]
    pub reason: &'static str,
}

impl Suppression {
    /// Returns whether this entry matches an event with this target, message
    /// ID and message.
    fn matches(&self, target: &str, message_id: Option<&str>, message: &str) -> bool {
        self.target == target
            && match self.message {
                MessageMatch::Id(id) => message_id == Some(id),
                MessageMatch::Prefix(prefix) => message.starts_with(prefix),
            }
    }
}

/// How a [`Suppression`] identifies its message on its target.
#[derive(Debug)]
pub enum MessageMatch {
    /// The event's [`MESSAGE_ID_FIELD`] (for validation, the VUID), matched
    /// exactly. Use it whenever the message has an ID.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "no message with an ID is suppressed yet")
    )]
    Id(&'static str),
    /// The start of the event's message, for messages with no ID, such as a
    /// dependency's warnings through the `log` crate, or the Vulkan loader's.
    /// Make it long enough to identify one message, and leave out parts that
    /// vary, such as version numbers.
    Prefix(&'static str),
}

/// The individually listed messages that are suppressed (R4).
///
/// Add an entry only for a message that is harmless and can't be fixed here,
/// such as a false positive in the validation layer. A normal condition logged
/// at `WARN` is fixed by changing its level instead.
pub const SUPPRESSED: &[Suppression] = &[
    // Logged by the Vulkan loader through the debug messenger, so on the
    // validation target. The loader's messages have no VUID.
    Suppression {
        target: VALIDATION_TARGET,
        message: MessageMatch::Prefix("Layer VK_LAYER_OBS_HOOK uses API version "),
        reason: "OBS Studio's implicit capture layer declares an older Vulkan version than the \
                 engine requests, so the loader warns about it on every instance. The layer \
                 still works, and uninstalling or disabling it would stop OBS from capturing \
                 the engine.",
    },
    // Logged by `winit` 0.30 through `tracing` when it connects to an X
    // server, in `XConnection::new`.
    Suppression {
        target: "winit::platform_impl::linux::x11::xdisplay",
        message: MessageMatch::Prefix("error setting XSETTINGS; Xft options won't reload"),
        reason: "winit warns whenever no XSETTINGS manager owns the screen's selection, which \
                 is normal on a bare X server such as Xvfb, which the smoke runs use in CI, \
                 and under minimal window managers. It only means that font settings changed \
                 while the application runs aren't picked up, which the engine doesn't use.",
    },
];

/// Classifies an event by its level, target, message ID and message. Pure.
pub fn classify(level: Level, target: &str, message_id: Option<&str>, message: &str) -> Class {
    classify_with(level, target, message_id, message, SUPPRESSED)
}

/// Classifies an event as [`classify`] does, with `suppressed` in place of
/// [`SUPPRESSED`], so that tests can supply their own list.
fn classify_with(
    level: Level,
    target: &str,
    message_id: Option<&str>,
    message: &str,
    suppressed: &[Suppression],
) -> Class {
    // More verbose levels compare greater in `tracing`.
    if level > Level::WARN {
        return Class::Ignored;
    }
    let is_best_practices = message_id.is_some_and(|id| id.starts_with(BEST_PRACTICES_PREFIX));
    if target == VALIDATION_TARGET && is_best_practices {
        Class::Advisory
    } else if suppressed
        .iter()
        .any(|entry| entry.matches(target, message_id, message))
    {
        Class::Suppressed
    } else {
        Class::Counted
    }
}

/// Returns whether `metadata` is a session's span: a span with a
/// [`SESSION_FIELD`]. A fmt layer can use it to leave session spans out of
/// its output.
pub fn is_session_span(metadata: &Metadata<'_>) -> bool {
    metadata.is_span() && metadata.fields().field(SESSION_FIELD).is_some()
}

/// A per-layer filter that passes only what [`attribute`] needs: session
/// spans, and events at `WARN` and above, the only events that can be
/// [`Class::Counted`].
///
/// A layer whose filter rejects a span doesn't see it when walking an event's
/// scope, so a layer that attributes events must let session spans through.
/// Other spans are rejected, and skipped by the walk.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "only the test capture layer attributes events; the application has one session"
    )
)]
fn attribution_filter() -> FilterFn<fn(&Metadata<'_>) -> bool> {
    let is_attributable = |metadata: &Metadata<'_>| {
        is_session_span(metadata) || (metadata.is_event() && *metadata.level() <= Level::WARN)
    };
    // Session spans are at `INFO`, and nothing more verbose is needed.
    filter_fn(is_attributable as fn(&Metadata<'_>) -> bool).with_max_level_hint(LevelFilter::INFO)
}

/// Layer that attributes events to sessions. Must be in every subscriber that
/// uses [`DiagnosticCounter`] or `TestCapture`.
///
/// When a span with a [`SESSION_FIELD`] is created, it records the field's
/// value as a [`SessionId`] in the span's extensions, where [`attribute`]
/// finds it. It never panics, since spans are created on any thread.
#[derive(Debug)]
pub struct SessionLayer;

impl SessionLayer {
    /// The layer, with its per-layer filter, which passes session spans only.
    pub fn layer<S>() -> impl Layer<S>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        // Session spans are at `INFO`, and nothing more verbose is needed.
        let filter = filter_fn(is_session_span as fn(&Metadata<'_>) -> bool)
            .with_max_level_hint(LevelFilter::INFO);
        Self.with_filter(filter)
    }
}

impl<S> Layer<S> for SessionLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &tracing::span::Id, ctx: Context<'_, S>) {
        let mut fields = EventFields::default();
        attrs.record(&mut fields);
        if let (Some(session), Some(span)) = (fields.session, ctx.span(id)) {
            // `replace` rather than `insert`, which panics if there's already
            // a value.
            span.extensions_mut().replace(session);
        }
    }
}

/// The fields of an event or span that diagnostics use.
#[derive(Debug, Default)]
struct EventFields {
    /// The [`SESSION_FIELD`], if present and a valid ID.
    session: Option<SessionId>,
    /// The [`MESSAGE_ID_FIELD`], if present.
    message_id: Option<String>,
    /// The event's message, or empty if it has none.
    message: String,
}

impl EventFields {
    /// Collects the fields of `event`.
    fn of(event: &Event<'_>) -> Self {
        let mut fields = Self::default();
        event.record(&mut fields);
        fields
    }
}

impl Visit for EventFields {
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == SESSION_FIELD {
            self.session = Some(SessionId(value));
        }
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        // An untyped integer literal (`session = 3`) is recorded as `i64`.
        if let Ok(value) = u64::try_from(value) {
            self.record_u64(field, value);
        }
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        // Recorded here, rather than through `record_debug`, so that strings
        // aren't quoted.
        match field.name() {
            MESSAGE_ID_FIELD => self.message_id = Some(value.to_owned()),
            "message" => self.message = value.to_owned(),
            _ => {}
        }
    }

    fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
        // The message is usually `fmt::Arguments`, whose `Debug` output is
        // the formatted text.
        match field.name() {
            MESSAGE_ID_FIELD => self.message_id = Some(format!("{value:?}")),
            "message" => self.message = format!("{value:?}"),
            _ => {}
        }
    }
}

/// Attributes `event` to a session: `explicit` (the event's own
/// [`SESSION_FIELD`]) if present, otherwise the nearest enclosing span that
/// [`SessionLayer`] recorded a session for, otherwise none.
///
/// The walk sees only the spans that the calling layer's filter passes, so the
/// caller must be filtered by [`attribution_filter`] or another filter that
/// passes session spans.
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "only the test capture layer attributes events; the application has one session"
    )
)]
fn attribute<S>(
    explicit: Option<SessionId>,
    event: &Event<'_>,
    ctx: &Context<'_, S>,
) -> Option<SessionId>
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    explicit.or_else(|| {
        ctx.event_scope(event)?
            .find_map(|span| span.extensions().get::<SessionId>().copied())
    })
}

/// Layer that counts [`Class::Counted`] events, for the application's summary
/// and `--fail-on-warnings`.
///
/// The application has a single session, so the counter doesn't attribute
/// events: unattributed ones (from the `log` crate, for example) count too.
/// Clones share their counts, so the application keeps one clone and installs
/// another with [`layer`](Self::layer).
#[derive(Debug, Clone, Default)]
pub struct DiagnosticCounter {
    /// The counts, shared between clones.
    counts: Arc<Counts>,
}

/// The counts behind a [`DiagnosticCounter`].
#[derive(Debug, Default)]
struct Counts {
    /// The number of counted errors.
    errors: AtomicU32,
    /// The number of counted warnings.
    warnings: AtomicU32,
}

impl DiagnosticCounter {
    /// A clone of this counter as a layer, with its per-layer filter, which
    /// passes events at `WARN` and above whatever `RUST_LOG` says.
    pub fn layer<S>(&self) -> impl Layer<S> + use<S>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        self.clone().with_filter(LevelFilter::WARN)
    }

    /// The number of counted errors so far.
    pub fn errors(&self) -> u32 {
        self.counts.errors.load(Ordering::Relaxed)
    }

    /// The number of counted warnings so far.
    pub fn warnings(&self) -> u32 {
        self.counts.warnings.load(Ordering::Relaxed)
    }
}

impl<S> Layer<S> for DiagnosticCounter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let fields = EventFields::of(event);
        let level = *metadata.level();
        let class = classify(
            level,
            metadata.target(),
            fields.message_id.as_deref(),
            &fields.message,
        );
        if class != Class::Counted {
            return;
        }
        let count = if level == Level::ERROR {
            &self.counts.errors
        } else {
            &self.counts.warnings
        };
        // Only the totals matter, not ordering with other memory.
        count.fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of attribution (R2), classification (R4) and counting (part
    //! of R15). Attribution and counting are checked under local subscribers,
    //! the one place tests may install their own: they test the layers
    //! themselves, rather than code that logs through them.

    use std::sync::Mutex;
    use std::thread;

    use tracing::{Dispatch, dispatcher, error, info, info_span, warn};
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::{EnvFilter, fmt, registry};

    use super::*;

    /// A layer that records the attribution of every warning or error it sees.
    #[derive(Clone, Default)]
    struct AttributionProbe {
        /// Each event's attribution, in the order they arrived.
        seen: Arc<Mutex<Vec<Option<SessionId>>>>,
    }

    impl<S> Layer<S> for AttributionProbe
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
            let fields = EventFields::of(event);
            let session = attribute(fields.session, event, &ctx);
            self.seen.lock().unwrap().push(session);
        }
    }

    /// Runs `f` under a subscriber like the application's, and returns the
    /// attribution of each warning or error it logged. `f` receives the
    /// subscriber, to install on any threads it starts.
    fn attributions(f: impl FnOnce(&Dispatch)) -> Vec<Option<SessionId>> {
        let probe = AttributionProbe::default();
        // The fmt layer passes every span, so that spans without a session
        // exist and the walk has to skip them.
        let dispatch = Dispatch::new(
            registry()
                .with(SessionLayer::layer())
                .with(
                    fmt::layer()
                        .with_test_writer()
                        .with_filter(LevelFilter::TRACE),
                )
                .with(probe.clone().with_filter(attribution_filter())),
        );
        dispatcher::with_default(&dispatch, || f(&dispatch));
        probe.seen.lock().unwrap().clone()
    }

    /// An event inside a session's span is attributed to that session.
    #[test]
    fn events_in_a_session_span_are_attributed_to_it() {
        let mut id = None;
        let seen = attributions(|_| {
            let session = Session::new();
            id = Some(session.id());
            session.span().in_scope(|| warn!("in the session"));
        });
        assert_eq!(seen, [id]);
    }

    /// An event in a span without a session, inside a session's span, is
    /// attributed to the enclosing session.
    #[test]
    fn events_in_a_nested_span_are_attributed_to_the_enclosing_session() {
        let mut id = None;
        let seen = attributions(|_| {
            let session = Session::new();
            id = Some(session.id());
            session.span().in_scope(|| {
                info_span!("inner").in_scope(|| warn!("in a nested span"));
            });
        });
        assert_eq!(seen, [id]);
    }

    /// When session spans are nested, the nearest one wins.
    #[test]
    fn the_nearest_session_span_wins() {
        let mut inner_id = None;
        let seen = attributions(|_| {
            let outer = Session::new();
            let inner = Session::new();
            inner_id = Some(inner.id());
            outer
                .span()
                .in_scope(|| inner.span().in_scope(|| warn!("in the inner session")));
        });
        assert_eq!(seen, [inner_id]);
    }

    /// An explicit `session` field, as callbacks from C code use, overrides
    /// the enclosing session span.
    #[test]
    fn an_explicit_session_field_overrides_the_span() {
        let mut explicit_id = None;
        let seen = attributions(|_| {
            let current = Session::new();
            let explicit = Session::new();
            explicit_id = Some(explicit.id());
            current
                .span()
                .in_scope(|| warn!(session = explicit.id().as_u64(), "from a callback"));
        });
        assert_eq!(seen, [explicit_id]);
    }

    /// An event on another thread that entered the session's span is
    /// attributed to the session, as the job system's will be.
    #[test]
    fn events_on_threads_that_enter_the_span_are_attributed() {
        let mut id = None;
        let seen = attributions(|dispatch| {
            let session = Session::new();
            id = Some(session.id());
            let span = session.span().clone();
            let dispatch = dispatch.clone();
            thread::spawn(move || {
                dispatcher::with_default(&dispatch, || {
                    let _entered = span.entered();
                    warn!("on a worker thread");
                });
            })
            .join()
            .unwrap();
        });
        assert_eq!(seen, [id]);
    }

    /// Threads don't inherit the span of the thread that started them, so a
    /// thread that doesn't enter it logs unattributed events. This is why
    /// threads must enter the session span.
    #[test]
    fn events_on_threads_that_dont_enter_the_span_are_unattributed() {
        let seen = attributions(|dispatch| {
            let session = Session::new();
            let dispatch = dispatch.clone();
            session.span().in_scope(|| {
                thread::spawn(move || {
                    dispatcher::with_default(&dispatch, || warn!("on a bare thread"));
                })
                .join()
                .unwrap();
            });
        });
        assert_eq!(seen, [None]);
    }

    /// An event outside any session is unattributed.
    #[test]
    fn events_outside_a_session_are_unattributed() {
        let seen = attributions(|_| warn!("outside any session"));
        assert_eq!(seen, [None]);
    }

    /// No two sessions share an ID.
    #[test]
    fn session_ids_are_unique() {
        let first = SessionId::unique();
        let second = SessionId::unique();
        assert_ne!(first, second);
    }

    /// A suppression list for the classification tests, with one entry of
    /// each kind.
    const TEST_SUPPRESSED: &[Suppression] = &[
        Suppression {
            target: "clockwork::test",
            message: MessageMatch::Id("VUID-harmless"),
            reason: "a test entry",
        },
        Suppression {
            target: "clockwork::test",
            message: MessageMatch::Prefix("A harmless message"),
            reason: "a test entry",
        },
    ];

    /// Warnings and errors with no special handling are counted.
    #[test]
    fn warnings_and_errors_are_counted() {
        for level in [Level::WARN, Level::ERROR] {
            assert_eq!(
                classify_with(level, "clockwork::test", None, "a message", TEST_SUPPRESSED),
                Class::Counted,
            );
            assert_eq!(
                classify_with(
                    level,
                    "clockwork::test",
                    Some("VUID-other"),
                    "a message",
                    TEST_SUPPRESSED,
                ),
                Class::Counted,
            );
        }
    }

    /// `INFO` and below are only logged.
    #[test]
    fn info_and_below_are_ignored() {
        for level in [Level::INFO, Level::DEBUG, Level::TRACE] {
            assert_eq!(
                classify_with(level, "clockwork::test", None, "a message", TEST_SUPPRESSED),
                Class::Ignored,
            );
        }
    }

    /// A message listed by ID is suppressed, but only on its own target.
    #[test]
    fn messages_listed_by_id_are_suppressed_on_their_target_only() {
        assert_eq!(
            classify_with(
                Level::ERROR,
                "clockwork::test",
                Some("VUID-harmless"),
                "a message",
                TEST_SUPPRESSED,
            ),
            Class::Suppressed,
        );
        assert_eq!(
            classify_with(
                Level::ERROR,
                "clockwork::other",
                Some("VUID-harmless"),
                "a message",
                TEST_SUPPRESSED,
            ),
            Class::Counted,
        );
    }

    /// A message listed by prefix is suppressed whether or not it has an ID,
    /// but only on its own target.
    #[test]
    fn messages_listed_by_prefix_are_suppressed_on_their_target_only() {
        for message_id in [None, Some("VUID-other")] {
            assert_eq!(
                classify_with(
                    Level::WARN,
                    "clockwork::test",
                    message_id,
                    "A harmless message, with details",
                    TEST_SUPPRESSED,
                ),
                Class::Suppressed,
            );
        }
        assert_eq!(
            classify_with(
                Level::WARN,
                "clockwork::other",
                None,
                "A harmless message, with details",
                TEST_SUPPRESSED,
            ),
            Class::Counted,
        );
    }

    /// A prefix matches only at the start of the message.
    #[test]
    fn prefixes_match_only_at_the_start() {
        assert_eq!(
            classify_with(
                Level::WARN,
                "clockwork::test",
                None,
                "Not A harmless message",
                TEST_SUPPRESSED,
            ),
            Class::Counted,
        );
    }

    /// Best-practices validation messages are advisory, but the same ID on
    /// another target is counted.
    #[test]
    fn best_practices_validation_messages_are_advisory() {
        let id = Some("BestPractices-vkCreateDevice-physical-device-features-not-retrieved");
        assert_eq!(
            classify_with(
                Level::WARN,
                VALIDATION_TARGET,
                id,
                "a message",
                TEST_SUPPRESSED
            ),
            Class::Advisory,
        );
        assert_eq!(
            classify_with(
                Level::WARN,
                "clockwork::test",
                id,
                "a message",
                TEST_SUPPRESSED
            ),
            Class::Counted,
        );
    }

    /// winit's warning when there's no XSETTINGS manager, as on the Xvfb the
    /// smoke runs use in CI, is suppressed, exactly as winit 0.30 logs it.
    #[test]
    fn winits_missing_xsettings_warning_is_suppressed() {
        assert_eq!(
            classify(
                Level::WARN,
                "winit::platform_impl::linux::x11::xdisplay",
                None,
                "error setting XSETTINGS; Xft options won't reload automatically",
            ),
            Class::Suppressed,
        );
    }

    /// Every suppression says what it suppresses and why (R4).
    #[test]
    fn every_suppression_has_a_target_message_and_reason() {
        for entry in SUPPRESSED {
            assert!(!entry.target.is_empty(), "{entry:?} has no target");
            let (MessageMatch::Id(text) | MessageMatch::Prefix(text)) = entry.message;
            assert!(!text.is_empty(), "{entry:?} matches every message");
            assert!(!entry.reason.trim().is_empty(), "{entry:?} has no reason");
        }
    }

    /// The OBS capture layer's loader warning, as logged on the Windows
    /// development machine, is suppressed for any pair of versions.
    #[test]
    fn the_obs_layer_version_warning_is_suppressed() {
        for message in [
            "Layer VK_LAYER_OBS_HOOK uses API version 1.3 which is older than the application \
             specified API version of 1.4. May cause issues.",
            "Layer VK_LAYER_OBS_HOOK uses API version 1.4 which is older than the application \
             specified API version of 1.5. May cause issues.",
        ] {
            assert_eq!(
                classify(
                    Level::WARN,
                    VALIDATION_TARGET,
                    Some("Loader Message"),
                    message
                ),
                Class::Suppressed,
            );
        }
    }

    /// The counter counts warnings and errors even when the fmt layer's
    /// `RUST_LOG` filter is `error`, which would disable every `warn!` if it
    /// were a global filter (R15).
    #[test]
    fn the_counter_counts_whatever_rust_log_says() {
        let counter = DiagnosticCounter::default();
        let dispatch = Dispatch::new(
            registry()
                .with(SessionLayer::layer())
                .with(
                    fmt::layer()
                        .with_test_writer()
                        .with_filter(EnvFilter::new("error")),
                )
                .with(counter.layer()),
        );
        dispatcher::with_default(&dispatch, || {
            warn!("a warning");
            warn!("another warning");
            error!("an error");
            info!("not counted");
        });
        assert_eq!(counter.errors(), 1);
        assert_eq!(counter.warnings(), 2);
    }
}
