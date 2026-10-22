//! Failing tests on warnings and errors: `TestCapture` and [`TestSession`].
//!
//! `TestCapture` is the capture layer of the test subscriber, which
//! `install_test_subscriber` installs once per test process. It holds a
//! registry from [`SessionId`] to `Collector`, and routes each
//! `Class::Counted` event to the collector of the session it's attributed
//! to. Events it can't route are printed with an `unattributed` banner.
//!
//! A [`TestSession`] registers a collector and enters its session's span on
//! the test's thread. When dropped, it fails the test if its collector saw an
//! unexpected warning or error, or if an expected one never arrived.
//!
//! The capture layer runs on whichever thread emitted the event, sometimes
//! inside a C callback, so it never panics: failures are reported when the
//! `TestSession` is dropped, on the test's own thread.

use std::collections::HashMap;
use std::fmt;
use std::io::{self, Write};
use std::panic::{self, AssertUnwindSafe, Location};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, Once, PoisonError};

use tracing::level_filters::LevelFilter;
use tracing::span::EnteredSpan;
use tracing::{Event, Level, Span, Subscriber};
use tracing_subscriber::layer::{Context, SubscriberExt};
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer, registry};

use super::{
    Class, EventFields, Session, SessionId, SessionLayer, attribute, attribution_filter, classify,
    target_of,
};

/// The most unexpected events a collector records. Beyond this it only counts
/// them, so that a broken loop can't grow memory without bound.
const RECORD_CAP: usize = 64;

/// A counted event, as recorded for a test.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapturedEvent {
    /// The event's level: `WARN` or `ERROR`.
    pub level: Level,
    /// The event's target.
    pub target: String,
    /// The event's message ID (for validation, the VUID), if it has one.
    pub message_id: Option<String>,
    /// The event's message.
    pub message: String,
}

impl fmt::Display for CapturedEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.level, self.target)?;
        if let Some(id) = &self.message_id {
            write!(f, " [{id}]")?;
        }
        write!(f, ": {}", self.message)
    }
}

/// A predicate an expected event must satisfy, besides its level and target.
type Matcher = Box<dyn Fn(&CapturedEvent) -> bool + Send + Sync>;

/// One event a test expects, declared with [`TestSession::expect`].
struct Expectation {
    /// The level the event must have.
    level: Level,
    /// The target the event must have, matched exactly.
    target: String,
    /// What else the event must satisfy.
    matcher: Matcher,
    /// Where the expectation was declared, for the failure message.
    location: &'static Location<'static>,
}

impl Expectation {
    /// Returns whether `event` meets this expectation. A matcher that panics
    /// doesn't match: the capture layer must never panic.
    fn matches(&self, event: &CapturedEvent) -> bool {
        self.level == event.level
            && self.target == event.target
            && panic::catch_unwind(AssertUnwindSafe(|| (self.matcher)(event))).unwrap_or(false)
    }
}

impl fmt::Display for Expectation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}, declared at {}",
            self.level, self.target, self.location
        )
    }
}

/// The counted events attributed to one session, and what it expects.
///
/// Shared between the registry and the session's [`TestSession`], which
/// removes it from the registry when dropped, so it never outlives its
/// session.
#[derive(Default)]
struct Collector {
    /// The collector's state. Locked only briefly, from any thread.
    state: Mutex<CollectorState>,
}

/// The state behind a [`Collector`]'s lock.
#[derive(Default)]
struct CollectorState {
    /// Expectations not yet met, in the order they were declared.
    expected: Vec<Expectation>,
    /// The first [`RECORD_CAP`] unexpected events.
    unexpected: Vec<CapturedEvent>,
    /// The number of unexpected events, including those not recorded.
    unexpected_count: usize,
}

impl Collector {
    /// Locks the state, recovering it if a thread panicked while holding the
    /// lock, since the capture layer must never panic.
    fn lock(&self) -> MutexGuard<'_, CollectorState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Adds an expectation, which the next matching event will meet.
    fn expect(&self, expectation: Expectation) {
        self.lock().expected.push(expectation);
    }

    /// Records a counted event: it meets the first unmet expectation that
    /// matches it, or else is unexpected.
    ///
    /// Matchers run while the lock is held, so they mustn't log.
    fn record(&self, event: CapturedEvent) {
        let mut state = self.lock();
        if let Some(index) = state.expected.iter().position(|e| e.matches(&event)) {
            state.expected.remove(index);
        } else {
            state.unexpected_count += 1;
            if state.unexpected.len() < RECORD_CAP {
                state.unexpected.push(event);
            }
        }
    }

    /// Describes every unexpected event, and every expectation not met, or
    /// returns `None` if there are none.
    fn report(&self, id: SessionId) -> Option<String> {
        use std::fmt::Write as _;

        let state = self.lock();
        if state.unexpected_count == 0 && state.expected.is_empty() {
            return None;
        }
        // Writing to a `String` can't fail.
        let mut report = format!("test session {id} failed:");
        if state.unexpected_count > 0 {
            let _ = write!(
                report,
                "\n{} unexpected warning or error event(s)",
                state.unexpected_count
            );
            let unrecorded = state.unexpected_count - state.unexpected.len();
            if unrecorded > 0 {
                let _ = write!(report, " (the first {RECORD_CAP} are listed)");
            }
            report.push(':');
            for event in &state.unexpected {
                let _ = write!(report, "\n  {event}");
            }
            if unrecorded > 0 {
                let _ = write!(report, "\n  ...and {unrecorded} more");
            }
        }
        if !state.expected.is_empty() {
            let _ = write!(
                report,
                "\n{} expected event(s) never arrived:",
                state.expected.len()
            );
            for expectation in &state.expected {
                let _ = write!(report, "\n  {expectation}");
            }
        }
        Some(report)
    }
}

/// The collectors of every live [`TestSession`], by session.
static COLLECTORS: LazyLock<Mutex<HashMap<SessionId, Arc<Collector>>>> =
    LazyLock::new(Mutex::default);

/// Locks [`COLLECTORS`], recovering it if a thread panicked while holding the
/// lock, since the capture layer must never panic.
fn collectors() -> MutexGuard<'static, HashMap<SessionId, Arc<Collector>>> {
    COLLECTORS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The capture layer of the test subscriber. Routes each counted event to
/// the collector of the session it's attributed to, or prints it with an
/// `unattributed` banner. Never panics.
#[derive(Debug)]
struct TestCapture;

impl TestCapture {
    /// The layer, with its per-layer filter, which passes session spans and
    /// events at `WARN` and above whatever `RUST_LOG` says.
    fn layer<S>() -> impl Layer<S>
    where
        S: Subscriber + for<'a> LookupSpan<'a>,
    {
        Self.with_filter(attribution_filter())
    }
}

/// Installs the test subscriber as the process's global default, once per
/// test process. [`TestSession::new`] calls it, so tests don't need to.
///
/// The subscriber has three layers, each with its own filter:
/// 1. [`SessionLayer`], which attributes events to sessions
/// 2. a fmt layer using the test writer, so that logs appear in the output of
///    failing tests only, filtered by `RUST_LOG` (`INFO` and above by default)
/// 3. [`TestCapture`], which takes warnings and errors whatever `RUST_LOG`
///    says, and fails the tests they're attributed to
///
/// It also routes the `log` crate's records into `tracing`.
///
/// # Panics
///
/// Panics if another global subscriber, or another `log` logger, was
/// installed first. Tests must not install their own subscribers, which would
/// hide events from the capture layer.
fn install_test_subscriber() {
    /// Guards the installation.
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let fmt_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();
        registry()
            .with(SessionLayer::layer())
            .with(
                tracing_subscriber::fmt::layer()
                    .with_test_writer()
                    .with_filter(fmt_filter),
            )
            .with(TestCapture::layer())
            .try_init()
            .expect("another global subscriber was installed before the test subscriber");
    });
}

impl<S> Layer<S> for TestCapture
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_event(&self, event: &Event<'_>, ctx: Context<'_, S>) {
        let metadata = event.metadata();
        let fields = EventFields::of(event);
        let level = *metadata.level();
        let target = target_of(event);
        let class = classify(
            level,
            &target,
            fields.message_id.as_deref(),
            &fields.message,
        );
        if class != Class::Counted {
            return;
        }
        let session = attribute(fields.session, event, &ctx);
        let captured = CapturedEvent {
            level,
            target,
            message_id: fields.message_id,
            message: fields.message,
        };
        // Clone the collector out, so that the registry is locked only
        // briefly.
        let collector = session.and_then(|id| collectors().get(&id).cloned());
        match collector {
            Some(collector) => collector.record(captured),
            None => report_unattributed(session, &captured),
        }
    }
}

/// Prints a counted event that can't fail any test, with a banner, so that
/// it's visible.
///
/// It's written to the process's stderr directly, rather than with
/// `eprintln!`, which the test harness captures and shows only if the
/// current test fails.
fn report_unattributed(session: Option<SessionId>, event: &CapturedEvent) {
    let context = match session {
        None => "outside any session".to_owned(),
        Some(id) => format!("in session {id}, which has no live TestSession"),
    };
    // Nothing can be done if stderr is unwritable, and the capture layer
    // must never panic.
    let _ = writeln!(
        io::stderr(),
        "=== clockwork: unattributed {} {context} ===\n  {event}",
        event.level
    );
}

/// A test's session. Fails the test, when dropped, if an unexpected warning
/// or error was attributed to it, or an expected one never arrived.
///
/// It enters its session's span on the test's thread, and holds the guard,
/// so it isn't `Send`: it belongs to that thread. Threads that the test
/// starts must enter a clone of [`span`](Self::span) to be attributed to it.
///
/// ```ignore
/// let session = TestSession::new();
/// session.expect(Level::WARN, "clockwork::import", |event| event.message.contains("case"));
/// // ...code that should log that warning, and nothing else at `WARN` or above...
/// ```
pub struct TestSession {
    /// The guard that keeps the session's span entered on this thread.
    /// Declared first, so that it's exited before anything else is dropped.
    _entered: EnteredSpan,
    /// The session, which owns the span.
    session: Session,
    /// Where events attributed to the session are collected, shared with
    /// [`COLLECTORS`].
    collector: Arc<Collector>,
}

impl TestSession {
    /// Installs the test subscriber if needed, then creates and enters a
    /// session.
    ///
    /// # Panics
    ///
    /// Panics if another global subscriber was installed first.
    #[expect(
        clippy::new_without_default,
        reason = "a session installs the subscriber and enters its span, which a default value \
                  shouldn't hide"
    )]
    pub fn new() -> Self {
        install_test_subscriber();
        let session = Session::new();
        let collector = Arc::new(Collector::default());
        collectors().insert(session.id(), Arc::clone(&collector));
        let entered = session.span().clone().entered();
        Self {
            _entered: entered,
            session,
            collector,
        }
    }

    /// The session's ID.
    pub fn id(&self) -> SessionId {
        self.session.id()
    }

    /// The session's span, for entering on threads the test starts.
    pub fn span(&self) -> &Span {
        self.session.span()
    }

    /// Declares that one counted event at `level`, on `target`, and
    /// satisfying `matcher`, is expected. Declare it before the code that
    /// triggers it. Each expectation is met by one event, so declare an event
    /// that's expected twice twice.
    ///
    /// `matcher` runs on the thread that logged the event, so it mustn't log,
    /// and a panic in it counts as not matching.
    #[track_caller]
    pub fn expect(
        &self,
        level: Level,
        target: &str,
        matcher: impl Fn(&CapturedEvent) -> bool + Send + Sync + 'static,
    ) {
        self.collector.expect(Expectation {
            level,
            target: target.to_owned(),
            matcher: Box::new(matcher),
            location: Location::caller(),
        });
    }
}

impl Drop for TestSession {
    /// Unregisters the collector, then fails the test if anything was
    /// unexpected or missing. If the test is already panicking, prints the
    /// report instead, since a second panic would abort the test process.
    fn drop(&mut self) {
        collectors().remove(&self.session.id());
        if let Some(report) = self.collector.report(self.session.id()) {
            if std::thread::panicking() {
                eprintln!("{report}");
            } else {
                panic!("{report}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of `TestSession` and its collector (R3).

    use std::thread;

    use tracing::{error, info, warn};

    use super::*;

    /// This module's target, which its events are logged on.
    const TARGET: &str = module_path!();

    /// An unexpected warning fails the test.
    #[test]
    #[should_panic(expected = "1 unexpected warning or error event(s):\n  WARN")]
    fn unexpected_warnings_fail_the_test() {
        let _session = TestSession::new();
        warn!("an unexpected warning");
    }

    /// An unexpected error fails the test.
    #[test]
    #[should_panic(expected = "1 unexpected warning or error event(s):\n  ERROR")]
    fn unexpected_errors_fail_the_test() {
        let _session = TestSession::new();
        error!("an unexpected error");
    }

    /// An unexpected warning from a thread that entered the test's span fails
    /// the test, on the test's own thread.
    #[test]
    #[should_panic(expected = "from a worker")]
    fn unexpected_warnings_on_other_threads_fail_the_test() {
        let session = TestSession::new();
        let span = session.span().clone();
        thread::spawn(move || {
            let _entered = span.entered();
            warn!("from a worker");
        })
        .join()
        .unwrap();
    }

    /// An event with an explicit `session` field, from a thread with no
    /// session span, reaches the test's session, as the Vulkan debug
    /// callback's events will.
    #[test]
    fn explicit_session_fields_reach_the_test_session() {
        let session = TestSession::new();
        session.expect(Level::ERROR, TARGET, |event| event.message == "from C");
        let id = session.id().as_u64();
        thread::spawn(move || error!(session = id, "from C"))
            .join()
            .unwrap();
    }

    /// Events below `WARN` don't fail the test.
    #[test]
    fn info_events_pass() {
        let _session = TestSession::new();
        info!("normal and notable");
    }

    /// A declared warning that arrives passes.
    #[test]
    fn expected_warnings_pass() {
        let session = TestSession::new();
        session.expect(Level::WARN, TARGET, |event| event.message == "expected");
        warn!("expected");
    }

    /// A declared warning that never arrives fails the test.
    #[test]
    #[should_panic(expected = "1 expected event(s) never arrived:\n  WARN")]
    fn missing_expected_warnings_fail_the_test() {
        let session = TestSession::new();
        session.expect(Level::WARN, TARGET, |event| event.message == "never logged");
    }

    /// An expectation is met by one event only, so a second matching event
    /// is unexpected.
    #[test]
    #[should_panic(expected = "1 unexpected warning or error event(s)")]
    fn each_expectation_is_met_once() {
        let session = TestSession::new();
        session.expect(Level::WARN, TARGET, |_| true);
        warn!("expected once");
        warn!("expected once");
    }

    /// An expected event at the wrong level doesn't meet the expectation.
    #[test]
    #[should_panic(expected = "never arrived")]
    fn expectations_match_the_level() {
        let session = TestSession::new();
        session.expect(Level::ERROR, TARGET, |_| true);
        warn!("a warning, not an error");
    }

    /// When the test is already panicking, the session prints its report
    /// rather than panicking again, which would abort the test process. The
    /// test's own panic is the one reported.
    #[test]
    #[should_panic(expected = "the test's own panic")]
    fn sessions_dont_panic_while_the_test_is_panicking() {
        let _session = TestSession::new();
        warn!("unexpected, but reported only on stderr");
        panic!("the test's own panic");
    }

    /// A warning or error for the collector tests.
    fn event(message: &str) -> CapturedEvent {
        CapturedEvent {
            level: Level::WARN,
            target: TARGET.to_owned(),
            message_id: None,
            message: message.to_owned(),
        }
    }

    /// The collector records up to `RECORD_CAP` unexpected events, but counts
    /// them all, so that a broken loop can't grow memory without bound.
    #[test]
    fn collectors_cap_recorded_events_but_count_them_all() {
        let collector = Collector::default();
        for i in 0..100 {
            collector.record(event(&format!("event {i}")));
        }
        let state = collector.lock();
        assert_eq!(state.unexpected.len(), RECORD_CAP);
        assert_eq!(state.unexpected_count, 100);
        drop(state);
        let report = collector.report(SessionId::unique()).unwrap();
        assert!(report.contains("100 unexpected"), "{report}");
        assert!(report.contains("...and 36 more"), "{report}");
    }

    /// A matcher that panics doesn't match, and the collector carries on,
    /// since it runs on whichever thread logged the event.
    #[test]
    fn panicking_matchers_dont_match() {
        let collector = Collector::default();
        collector.expect(Expectation {
            level: Level::WARN,
            target: TARGET.to_owned(),
            matcher: Box::new(|_| panic!("a broken matcher")),
            location: Location::caller(),
        });
        collector.record(event("not matched"));
        let state = collector.lock();
        assert_eq!(state.unexpected_count, 1);
        assert_eq!(state.expected.len(), 1);
    }
}
