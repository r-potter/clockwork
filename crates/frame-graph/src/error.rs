//! The errors compiling reports (design 0003, "Error handling").
//!
//! Compiling reports every mistake it finds at once, each naming the passes
//! and resources involved, so a technique's author fixes them in one go
//! rather than one per run.

use std::error::Error;
use std::fmt;

/// One mistake in a graph's description, naming the passes and resources
/// involved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileError {
    /// What's wrong, as a sentence without a full stop.
    message: String,
}

impl CompileError {
    /// An error that says `message`.
    pub(crate) fn new(message: String) -> Self {
        Self { message }
    }

    /// What's wrong.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CompileError {
    /// Writes the message.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// Every mistake compiling found in a graph's description. Never empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompileErrors {
    /// The graph's label.
    graph: String,
    /// The mistakes, in the order they were found.
    errors: Vec<CompileError>,
}

impl CompileErrors {
    /// The mistakes, in the order they were found.
    pub fn errors(&self) -> &[CompileError] {
        &self.errors
    }
}

impl fmt::Display for CompileErrors {
    /// Writes a line naming the graph, then each error on a line of its
    /// own.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.errors.len();
        let noun = if count == 1 { "error" } else { "errors" };
        write!(f, "frame graph \"{}\" has {count} {noun}:", self.graph)?;
        for error in &self.errors {
            write!(f, "\n  {error}")?;
        }
        Ok(())
    }
}

impl Error for CompileErrors {}

/// Collects errors as compiling finds them.
#[derive(Debug, Default)]
pub(crate) struct Errors(Vec<CompileError>);

impl Errors {
    /// Records the error `message`.
    pub(crate) fn push(&mut self, message: String) {
        self.0.push(CompileError::new(message));
    }

    /// `value` if no error was recorded, and otherwise every error, for the
    /// graph `graph`.
    ///
    /// # Errors
    ///
    /// Returns the errors recorded, if there are any.
    pub(crate) fn finish<T>(self, graph: &str, value: T) -> Result<T, CompileErrors> {
        if self.0.is_empty() {
            Ok(value)
        } else {
            Err(CompileErrors {
                graph: graph.to_owned(),
                errors: self.0,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    //! T0 tests of how errors are reported.

    use super::*;

    /// With no errors recorded, finishing gives the value.
    #[test]
    fn no_errors_gives_the_value() {
        assert_eq!(Errors::default().finish("empty", 7), Ok(7));
    }

    /// Errors keep their order, and display one per line after a line
    /// naming the graph.
    #[test]
    fn errors_are_listed_one_per_line_after_the_graph() {
        let mut errors = Errors::default();
        errors.push("the first".to_owned());
        errors.push("the second".to_owned());
        let errors = errors.finish("bloom", ()).unwrap_err();

        assert_eq!(errors.errors().len(), 2);
        assert_eq!(errors.errors()[1].message(), "the second");
        assert_eq!(
            errors.to_string(),
            "frame graph \"bloom\" has 2 errors:\n  the first\n  the second"
        );
    }

    /// A single error is counted in the singular.
    #[test]
    fn one_error_is_singular() {
        let mut errors = Errors::default();
        errors.push("the only one".to_owned());
        assert_eq!(
            errors.finish("bloom", ()).unwrap_err().to_string(),
            "frame graph \"bloom\" has 1 error:\n  the only one"
        );
    }
}
