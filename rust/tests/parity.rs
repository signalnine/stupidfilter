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

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn binary_path() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_stupidfilter"))
}

fn model_base() -> PathBuf {
    // Integration tests run with cwd = crate root (rust/).
    PathBuf::from("..").join("data").join("c_rbf")
}

fn run_with_model(base: &Path, stdin_bytes: &[u8]) -> Output {
    let mut child = Command::new(binary_path())
        .arg(base)
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
    child.wait_with_output().expect("wait")
}

fn score(stdin_bytes: &[u8]) -> String {
    let out = run_with_model(&model_base(), stdin_bytes);
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

fn write_modified_model(tag: &str, transform: impl FnOnce(String) -> String) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(tag);
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mod_text = fs::read_to_string(model_base().with_extension("mod")).unwrap();
    fs::write(dir.join("m.mod"), transform(mod_text)).unwrap();
    fs::copy(model_base().with_extension("sf"), dir.join("m.sf")).unwrap();
    dir.join("m")
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

#[test]
fn model_missing_gamma_errors_like_cpp() {
    // bd issue stupidfilter-o76: a model file with gamma renamed (e.g.,
    // 'zamma') used to load with gamma silently defaulting to 0. The RBF
    // kernel then evaluates to 1 for every support vector and predictions
    // become meaningless. The C++ binary errors -- the Rust binary must too.
    let base = write_modified_model("missing_gamma", |s| s.replace("gamma ", "zamma "));
    let out = run_with_model(&base, b"Hello world\n");
    assert!(
        !out.status.success(),
        "expected non-zero exit when gamma is missing; got stdout={:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn model_missing_rho_errors() {
    // bd issue stupidfilter-o76: rho missing is also a fatal condition; it
    // used to default to 0, shifting the decision boundary.
    let base = write_modified_model("missing_rho", |s| s.replace("rho ", "rrho "));
    let out = run_with_model(&base, b"Hello world\n");
    assert!(
        !out.status.success(),
        "expected non-zero exit when rho is missing; got stdout={:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn inverted_label_inverts_output() {
    // bd issue stupidfilter-wgl: the default model has 'label 1 0', so
    // sum > rho => 1, otherwise 0. With 'label 0 1' the meaning of the
    // output should flip. The reference C++ binary produces 0.000000 on
    // this input with the inverted label; the Rust binary used to ignore
    // the label line entirely and produce 1.000000.
    let base = write_modified_model("inverted_label", |s| s.replace("label 1 0", "label 0 1"));
    let out = run_with_model(&base, b"Hello world\n");
    assert!(out.status.success(), "binary exited with {:?}", out.status);
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        "0.000000"
    );
}
