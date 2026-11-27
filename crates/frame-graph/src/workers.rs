//! Running the frame graph's jobs on the caller's threads (design 0003,
//! "Threading").
//!
//! The graph runs jobs only to record passes and translate them, and, once
//! techniques load, to compile programs: each time a number of jobs, then
//! a wait for them all. So the pool it's given is a trait with one method,
//! which the caller implements over whatever threads it has. The
//! application implements it over a `rayon` pool, and the engine's job
//! system, which `0005-ecs` chooses, will implement it too, so recording
//! shares threads with everything else. Only the application depends on
//! `rayon`.

/// Runs jobs on the caller's threads.
///
/// It's object-safe, and a call has no lifetime beyond itself, so jobs
/// borrow the frame's data freely.
///
/// An implementation over scoped threads, entering the caller's span in
/// each job so what it logs is attributed where the caller's is:
///
/// ```
/// use clockwork_frame_graph::WorkerPool;
///
/// /// A thread for each job, for the example's sake.
/// struct Threads;
///
/// impl WorkerPool for Threads {
///     fn for_each(&self, count: usize, job: &(dyn Fn(usize) + Sync)) {
///         let span = tracing::Span::current();
///         std::thread::scope(|scope| {
///             for index in 0..count {
///                 let span = span.clone();
///                 scope.spawn(move || span.in_scope(|| job(index)));
///             }
///         });
///     }
/// }
///
/// let squares: Vec<_> = (0..4).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect();
/// Threads.for_each(4, &|index| {
///     squares[index].store(index * index, std::sync::atomic::Ordering::Relaxed);
/// });
/// let squares: Vec<_> = squares.into_iter().map(|square| square.into_inner()).collect();
/// assert_eq!(squares, [0, 1, 4, 9]);
/// ```
pub trait WorkerPool: Sync {
    /// Calls `job` with each index in `0..count`, in any order and on any
    /// threads, and returns once every call has returned. Each call runs in
    /// the tracing span that was current where `for_each` was called.
    ///
    /// A panic in a job is propagated to the caller once no job is still
    /// running. Jobs not yet started when one panics may or may not run:
    /// `Serial` stops at the panic, and a pool of threads usually runs the
    /// rest first.
    fn for_each(&self, count: usize, job: &(dyn Fn(usize) + Sync));
}

/// Runs each job in turn on the caller's thread.
///
/// Tests are deterministic with it, a panic has a plain backtrace, and
/// what jobs log stays in the caller's span with no help.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Serial;

impl WorkerPool for Serial {
    fn for_each(&self, count: usize, job: &(dyn Fn(usize) + Sync)) {
        for index in 0..count {
            job(index);
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of `Serial`.

    use std::sync::Mutex;

    use super::*;

    /// `Serial` calls the job with each index once, in order, and with
    /// none for no jobs.
    #[test]
    fn serial_runs_each_job_in_turn() {
        let called = Mutex::new(Vec::new());
        let pool: &dyn WorkerPool = &Serial;
        pool.for_each(4, &|index| called.lock().unwrap().push(index));
        assert_eq!(*called.lock().unwrap(), [0, 1, 2, 3]);
        pool.for_each(0, &|_| panic!("there are no jobs"));
    }

    /// Pools are shared with the threads that call them, and `Serial` can
    /// be sent to one.
    #[test]
    fn pools_are_shared_between_threads() {
        /// Compiles only if `T` is `Send` and `Sync`.
        fn assert_send_sync<T: Send + Sync + ?Sized>() {}
        assert_send_sync::<Serial>();
        /// Compiles only if `T` is `Sync`.
        fn assert_sync<T: Sync + ?Sized>() {}
        assert_sync::<dyn WorkerPool>();
    }
}
