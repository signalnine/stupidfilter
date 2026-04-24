// Integration tests: run the stupidfilter binary against stdin and assert
// its output matches the reference C++ binary (whose numbers are baked in
// as constants below -- obtained by running an instrumented build on the
// same inputs, see data/c_rbf + the DEBUG-printf patch applied temporarily
// during debugging).
//
// The regressions covered here are:
//
//  1. main.rs used to call `input.trim_end()` before feature extraction.
//     That silently removed the `\n` that `echo` / line-buffered stdin
//     appends, which then caused flex rule 6 (initial_cap) to miss the
//     final word whenever it was `[A-Z][a-z]+`.
//
//  2. extract_features used to set `num_total = bytes.len()`, but flex
//     rule 1 is `.` which matches every byte EXCEPT `\n`. That shifted
//     every ratio feature on inputs containing `\n`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stupidfilter"))
}

fn model_base() -> PathBuf {
    // Integration tests run with cwd = crate root (rust/).
    PathBuf::from("..").join("data").join("c_rbf")
}

fn score(stdin_bytes: &[u8]) -> String {
    let mut child = Command::new(binary_path())
        .arg(model_base())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn stupidfilter");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin_bytes)
        .unwrap();
    let out = child.wait_with_output().expect("wait");
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

#[test]
fn last_word_capitalized_with_trailing_newline_matches_cpp() {
    // echo "UGH this is Dumb" -> C++ gives 1.000000 because rule 6 fires on
    // "Dumb\n". Before the fix, Rust trimmed the \n and scored 0.000000.
    assert_eq!(score(b"UGH this is Dumb\n"), "1.000000");
}

#[test]
fn trailing_capital_word_with_short_context_matches_cpp() {
    // Another stdin-from-echo case where the final word is initial-capped.
    assert_eq!(score(b"a b c d Dog\n"), "1.000000");
}

#[test]
fn input_without_trailing_newline_still_matches_cpp() {
    // Without trailing whitespace, rule 6 cannot fire on the last word.
    // Both binaries should agree: 0.000000 (instrumented C++ reference).
    assert_eq!(score(b"UGH this is Dumb"), "0.000000");
}
