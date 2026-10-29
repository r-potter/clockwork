//! Helpers for passing Rust data to C APIs, such as Vulkan.

use std::ffi::CString;

/// `text` as a C string, truncated at its first null character, which
/// would otherwise end the string early on the C side.
pub(crate) fn truncate_to_cstring(text: &str) -> CString {
    let end = text.find('\0').unwrap_or(text.len());
    CString::new(&text[..end]).expect("the text stops before any null character")
}

#[cfg(test)]
mod tests {
    //! T0 tests of the C string helpers.

    use super::*;

    /// Text without a null character is kept whole, and text with one is
    /// cut short there, rather than failing.
    #[test]
    fn text_is_truncated_at_a_null_character() {
        assert_eq!(
            truncate_to_cstring("triangle vertices").as_bytes(),
            b"triangle vertices"
        );
        assert_eq!(truncate_to_cstring("").as_bytes(), b"");
        assert_eq!(truncate_to_cstring("cut\0short").as_bytes(), b"cut");
    }
}
