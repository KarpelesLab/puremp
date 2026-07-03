//! Error and result types shared across the crate.

use core::fmt;

/// Errors produced by fallible `puremp` operations.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub enum Error {
    /// A string could not be parsed as a number of the requested kind.
    Parse,
    /// A division (or modular reduction, or rational construction) had a zero
    /// divisor.
    DivisionByZero,
    /// An operation was requested that this build does not yet implement.
    ///
    /// Used by scaffolding entry points so callers get a clean error rather
    /// than a panic while a layer is still under construction.
    Unimplemented,
    /// A value did not fit the target type of a conversion.
    Overflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            Error::Parse => "invalid numeric literal",
            Error::DivisionByZero => "division by zero",
            Error::Unimplemented => "operation not yet implemented",
            Error::Overflow => "value out of range for the target type",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience alias for results carrying an [`Error`].
pub type Result<T> = core::result::Result<T, Error>;
