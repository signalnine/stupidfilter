// Feature extraction module
// Extracts 8 text features for SVM classification

/// The 8 features used by the classifier
#[derive(Debug, Clone)]
pub struct Features {
    pub num_lowers: f64,    // Ratio of lowercase letters
    pub num_caps: f64,      // Ratio of uppercase letters
    pub num_punct: f64,     // Ratio of punctuation
    pub repeat_emphasis: f64, // Count of repeated !!/??
    pub initial_cap: f64,   // Ratio of words starting with capital
    pub intercap: f64,      // Ratio of camelCase words
    pub word_length: f64,   // word_count / total_chars
    pub misspell: f64,      // Count of l33t speak patterns
}

impl Features {
    /// Convert to array in the order expected by the model
    pub fn to_array(&self) -> [f64; 8] {
        [
            self.num_lowers,
            self.num_caps,
            self.num_punct,
            self.repeat_emphasis,
            self.initial_cap,
            self.intercap,
            self.word_length,
            self.misspell,
        ]
    }
}

/// Extract features from text
///
/// Operates on raw bytes to match the flex scanner's byte-by-byte semantics.
/// The flex rules use ASCII ranges ([a-z], [A-Z], [a-zA-Z0-9]); multi-byte
/// UTF-8 sequences never satisfy these classes, so each continuation byte
/// contributes only to num_total and is otherwise ignored.
pub fn extract_features(text: &str) -> Features {
    let bytes = text.as_bytes();
    let num_total = bytes.len() as f64;

    if num_total == 0.0 {
        return Features {
            num_lowers: 0.0,
            num_caps: 0.0,
            num_punct: 0.0,
            repeat_emphasis: 0.0,
            initial_cap: 0.0,
            intercap: 0.0,
            word_length: 0.0,
            misspell: 0.0,
        };
    }

    // Count character types -- ASCII-only, matching flex [a-z]/[A-Z]/punct.
    let num_lowers = bytes.iter().filter(|&&b| b.is_ascii_lowercase()).count() as f64;
    let num_caps = bytes.iter().filter(|&&b| b.is_ascii_uppercase()).count() as f64;
    let num_punct = bytes.iter().filter(|&&b| is_punct_byte(b)).count() as f64;

    // word_count follows the flex rule [a-zA-Z0-9]+ with REJECT: for each
    // run of consecutive alphanumeric chars of length L, every starting
    // position within the run fires (L - offset) matches, summing to
    // L*(L+1)/2 -- the count of all non-empty substrings.
    let mut word_count = 0.0;
    let mut initial_cap_count = 0.0;
    let mut intercap_count = 0.0;

    let n = bytes.len();

    let mut i = 0;
    while i < n {
        if bytes[i].is_ascii_alphanumeric() {
            let start = i;
            while i < n && bytes[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let end = i;
            let run_len = end - start;
            word_count += (run_len * (run_len + 1) / 2) as f64;
        } else {
            i += 1;
        }
    }

    // initial_cap fires once per [A-Z][a-z]+ substring that is immediately
    // followed by whitespace (space, tab, or newline). Leading context is
    // unconstrained. The actual DFA rule is reverse-engineered from the
    // instrumented C++ binary; the reconstructed fclassify.flex is an
    // approximation.
    let mut k = 0;
    while k < n {
        if bytes[k].is_ascii_uppercase() {
            let mut m = k + 1;
            while m < n && bytes[m].is_ascii_lowercase() {
                m += 1;
            }
            if m > k + 1 && m < n && is_ws_byte(bytes[m]) {
                initial_cap_count += 1.0;
            }
        }
        k += 1;
    }

    // intercap (unchanged here) still uses the legacy [a-zA-Z][A-Z]-pair scan.
    // That rule also diverges from the compiled C++ DFA on inputs with trailing
    // whitespace, but is left as-is in this fix and tracked separately.
    let mut j = 0;
    while j < n {
        if bytes[j].is_ascii_alphabetic() {
            let start = j;
            while j < n && bytes[j].is_ascii_alphabetic() {
                j += 1;
            }
            let end = j;
            let run_len = end - start;

            if run_len >= 2 {
                for k in start..(end - 1) {
                    if bytes[k + 1].is_ascii_uppercase() {
                        intercap_count += 1.0;
                    }
                }
            }
        } else {
            j += 1;
        }
    }

    // Count repeated emphasis (!! or ??)
    let repeat_emphasis = count_repeat_emphasis(text);

    // Count misspellings / l33t speak
    let misspell = count_misspellings(text);

    // Calculate ratios
    let num_lowers_ratio = num_lowers / num_total;
    let num_caps_ratio = num_caps / num_total;
    let num_punct_ratio = num_punct / num_total;

    let initial_cap_ratio = if word_count > 0.0 {
        initial_cap_count / word_count
    } else {
        0.0
    };

    let intercap_ratio = if word_count > 0.0 {
        intercap_count / word_count
    } else {
        0.0
    };

    let word_length = word_count / num_total;

    Features {
        num_lowers: num_lowers_ratio,
        num_caps: num_caps_ratio,
        num_punct: num_punct_ratio,
        repeat_emphasis,
        initial_cap: initial_cap_ratio,
        intercap: intercap_ratio,
        word_length,
        misspell,
    }
}

/// Whitespace bytes as treated by the flex scanner's trailing-context rules
/// (space, tab, newline only).
fn is_ws_byte(b: u8) -> bool {
    b == b' ' || b == b'\t' || b == b'\n'
}

/// Check if byte is in the flex punct character class
/// [!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]
fn is_punct_byte(b: u8) -> bool {
    matches!(
        b,
        b'!' | b'"'
            | b'#'
            | b'$'
            | b'%'
            | b'&'
            | b'\''
            | b'('
            | b')'
            | b'*'
            | b'+'
            | b','
            | b'-'
            | b'.'
            | b'/'
            | b':'
            | b';'
            | b'<'
            | b'='
            | b'>'
            | b'?'
            | b'@'
            | b'['
            | b'\\'
            | b']'
            | b'^'
            | b'_'
            | b'`'
            | b'{'
            | b'|'
            | b'}'
            | b'~'
    )
}

/// Count misspellings. Reverse-engineered from the C++ binary's behavior: the
/// flex rule has two alternatives, both counted additively.
///
/// Rule A: runs of two or more literal ' ' characters. A run of length N
/// fires (N - 1) matches (REJECT enumerates lengths 2..=N). Tabs and newlines
/// are not counted -- only ' '. classify.sh normalizes runs of spaces away,
/// so this alternative rarely fires in the standard pipeline.
///
/// Rule B: a small set of chat/l33t abbreviations (u, r, n, g, y, ur, gr8,
/// all lowercase) surrounded by literal ' ' on both sides. Leading space is
/// consumed; trailing space is trailing context. End-of-input, tabs, and
/// punctuation do NOT satisfy either boundary.
pub(crate) fn count_misspellings(text: &str) -> f64 {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut count = 0.0;

    let mut i = 0;
    while i < n {
        if bytes[i] == b' ' {
            let start = i;
            while i < n && bytes[i] == b' ' {
                i += 1;
            }
            let run_len = i - start;
            if run_len >= 2 {
                count += (run_len - 1) as f64;
            }
        } else {
            i += 1;
        }
    }

    const ABBREVS: &[&[u8]] = &[b"gr8", b"ur", b"u", b"r", b"n", b"g", b"y"];
    for i in 0..n {
        if bytes[i] != b' ' {
            continue;
        }
        for abbrev in ABBREVS {
            let end = i + 1 + abbrev.len();
            if end < n && &bytes[i + 1..end] == *abbrev && bytes[end] == b' ' {
                count += 1.0;
                break;
            }
        }
    }

    count
}

/// Count repeated emphasis runs. Mirrors the flex rule
/// `[!]{2,}|[?]{2,} ++repeat_emphasis; REJECT;`: for each run of N consecutive
/// identical emphasis characters (all '!' or all '?'), REJECT enumerates
/// lengths N, N-1, ..., 2 at the run's start and then the scanner advances
/// past the longest match, yielding (N - 1) fires per run. A single '!' or
/// '?' (N = 1) does not fire. '!' and '?' are separate runs: "!?" is two
/// length-1 runs and contributes 0.
pub(crate) fn count_repeat_emphasis(text: &str) -> f64 {
    let bytes = text.as_bytes();
    let n = bytes.len();
    let mut count = 0.0;

    let mut i = 0;
    while i < n {
        let c = bytes[i];
        if c == b'!' || c == b'?' {
            let start = i;
            while i < n && bytes[i] == c {
                i += 1;
            }
            let run_len = i - start;
            if run_len >= 2 {
                count += (run_len - 1) as f64;
            }
        } else {
            i += 1;
        }
    }

    count
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Recover the raw initial_cap counter (pre-division) used by the flex
    /// scanner, to compare against instrumented C++ reference values.
    fn initial_cap_count(input: &str) -> f64 {
        let features = extract_features(input);
        let num_total = input.as_bytes().len() as f64;
        let word_count = features.word_length * num_total;
        (features.initial_cap * word_count).round()
    }

    // Reference values come from instrumenting stupidfilter.cpp with a stderr
    // print of the raw initial_cap counter just before svm_predict. The actual
    // rule 6 (as compiled into the DFA) fires once per [A-Z][a-z]+ substring
    // that is immediately followed by [ \t\n]. Leading context is unconstrained.

    #[test]
    fn initial_cap_zero_without_trailing_whitespace() {
        assert_eq!(initial_cap_count("Hello"), 0.0);
        assert_eq!(initial_cap_count("Aa"), 0.0);
        assert_eq!(initial_cap_count(".Hello"), 0.0);
        assert_eq!(initial_cap_count(" Hello"), 0.0);
    }

    #[test]
    fn initial_cap_fires_with_trailing_space_tab_newline() {
        assert_eq!(initial_cap_count("Hello "), 1.0);
        assert_eq!(initial_cap_count("Hello\t"), 1.0);
        assert_eq!(initial_cap_count("Hello\n"), 1.0);
        assert_eq!(initial_cap_count("Aa "), 1.0);
    }

    #[test]
    fn initial_cap_requires_lowercase_after_capital() {
        assert_eq!(initial_cap_count("A "), 0.0);
        assert_eq!(initial_cap_count("AB "), 0.0);
        assert_eq!(initial_cap_count("ABCDEF "), 0.0);
    }

    #[test]
    fn initial_cap_leading_context_is_unconstrained() {
        assert_eq!(initial_cap_count(".Hello "), 1.0);
        assert_eq!(initial_cap_count(",Zebra "), 1.0);
        assert_eq!(initial_cap_count("Hi!World "), 1.0);
        assert_eq!(initial_cap_count("1Ab "), 1.0);
        assert_eq!(initial_cap_count("aAb "), 1.0);
        // Non-ASCII byte: é is 0xC3 0xA9, neither ASCII letter nor whitespace.
        assert_eq!(initial_cap_count("éHello "), 1.0);
    }

    #[test]
    fn initial_cap_multiple_words() {
        assert_eq!(initial_cap_count("Foo Bar"), 1.0);
        assert_eq!(initial_cap_count("Foo Bar "), 2.0);
        assert_eq!(initial_cap_count("Foo Bar Baz"), 2.0);
        assert_eq!(initial_cap_count("Hello World Foo"), 2.0);
        assert_eq!(initial_cap_count("Aaaa Bbbb "), 2.0);
    }

    #[test]
    fn initial_cap_inner_match_in_cap_run() {
        // "ABc " -> the [A-Z][a-z]+ match is "Bc" (inner position), followed
        // by space. Fires 1.
        assert_eq!(initial_cap_count("ABc "), 1.0);
        // Same logic: "AAAbc " -> match "Abc" at position 2, followed by space.
        assert_eq!(initial_cap_count("AAAbc "), 1.0);
    }

    #[test]
    fn initial_cap_only_match_with_trailing_whitespace() {
        // "AbAb " -> first "Ab" is followed by 'A' (not whitespace); only the
        // second "Ab" qualifies.
        assert_eq!(initial_cap_count("AbAb "), 1.0);
        // "Hello.World " -> "Hello" is followed by '.', only "World" qualifies.
        assert_eq!(initial_cap_count("Hello.World "), 1.0);
        assert_eq!(initial_cap_count("Hello.World"), 0.0);
        // "AbCd " -> "Ab" followed by C, skip. "Cd" followed by space, fire.
        assert_eq!(initial_cap_count("AbCd "), 1.0);
    }

    #[test]
    fn test_extract_features_normal() {
        let features = extract_features("Hello world");
        assert!(features.num_lowers > 0.0);
        assert!(features.num_caps > 0.0);
    }

    #[test]
    fn test_extract_features_emphasis() {
        // "OMG!!! WOW???": two runs of length 3 -> (3-1) + (3-1) = 4.
        let features = extract_features("OMG!!! WOW???");
        assert_eq!(features.repeat_emphasis, 4.0);
    }

    // Reference values from running the instrumented C++ binary on the same
    // inputs. The flex rule fires (N - 1) times per run of N identical '!' or
    // '?' characters; runs of different characters do NOT combine.

    #[test]
    fn repeat_emphasis_zero_for_single_char() {
        assert_eq!(count_repeat_emphasis("!"), 0.0);
        assert_eq!(count_repeat_emphasis("?"), 0.0);
    }

    #[test]
    fn repeat_emphasis_counts_n_minus_one() {
        assert_eq!(count_repeat_emphasis("!!"), 1.0);
        assert_eq!(count_repeat_emphasis("!!!"), 2.0);
        assert_eq!(count_repeat_emphasis("!!!!"), 3.0);
        assert_eq!(count_repeat_emphasis("!!!!!!!!!!"), 9.0);
    }

    #[test]
    fn repeat_emphasis_counts_question_runs() {
        assert_eq!(count_repeat_emphasis("??"), 1.0);
        assert_eq!(count_repeat_emphasis("????"), 3.0);
    }

    #[test]
    fn repeat_emphasis_sums_across_runs() {
        // "wow!!!! test????": two runs of 4 -> 3 + 3 = 6.
        assert_eq!(count_repeat_emphasis("wow!!!! test????"), 6.0);
    }

    #[test]
    fn repeat_emphasis_does_not_pool_bangs_and_questions() {
        // Adjacent '!' and '?' are two separate length-1 runs -> 0.
        assert_eq!(count_repeat_emphasis("!?"), 0.0);
        // "!!??" is two length-2 runs -> 1 + 1 = 2.
        assert_eq!(count_repeat_emphasis("!!??"), 2.0);
        // Separator between runs.
        assert_eq!(count_repeat_emphasis("!!a!!"), 1.0 + 1.0);
    }

    #[test]
    fn test_extract_features_leet_has_alphanumeric_runs() {
        // Alphanumeric runs include digits, so wlen > 0.
        let features = extract_features("u r 2 cool 4 school");
        assert!(features.word_length > 0.0);
    }

    // Helper: convert the intercap ratio back to the raw count the flex
    // scanner would have accumulated, to compare against C++ reference values.
    fn intercap_count(input: &str) -> f64 {
        let features = extract_features(input);
        let word_count = features.word_length * input.chars().count() as f64;
        (features.intercap * word_count).round()
    }

    #[test]
    fn intercap_matches_flex_for_all_caps() {
        // Flex fires rule 7 for every [a-zA-Z][A-Z] pair.
        // "ABC" has pairs (A,B) and (B,C) -> 2.
        assert_eq!(intercap_count("ABC"), 2.0);
    }

    #[test]
    fn intercap_matches_flex_for_mixed_case_run() {
        // "aBCd" has pairs (a,B) and (B,C) -> 2.
        assert_eq!(intercap_count("aBCd"), 2.0);
    }

    #[test]
    fn intercap_matches_flex_for_letter_then_upper_run() {
        // "aXYz" has pairs (a,X) and (X,Y) -> 2.
        assert_eq!(intercap_count("aXYz"), 2.0);
    }

    #[test]
    fn intercap_counts_every_match_not_just_first() {
        // "aBcDeF" has three transitions into uppercase: (a,B), (c,D), (e,F).
        assert_eq!(intercap_count("aBcDeF"), 3.0);
    }

    #[test]
    fn intercap_matches_flex_across_words() {
        // "OMG ur SO DUMB!!!": OM, MG, SO, DU, UM, MB -> 6.
        assert_eq!(intercap_count("OMG ur SO DUMB!!!"), 6.0);
    }

    #[test]
    fn intercap_counts_camel_transition() {
        // "HelloWorld": only (o,W) -> 1.
        assert_eq!(intercap_count("HelloWorld"), 1.0);
    }

    #[test]
    fn intercap_zero_for_all_lower_start() {
        // "Hello": only (H,e) is a letter pair touching a capital, but
        // the capital is first, not second -> 0.
        assert_eq!(intercap_count("Hello"), 0.0);
    }

    #[test]
    fn intercap_does_not_cross_whitespace() {
        // "Hello World": pairs across the space are not letter-letter -> 0.
        assert_eq!(intercap_count("Hello World"), 0.0);
    }

    // Reference values come from running the C++ binary with feature-dump
    // instrumentation on the same inputs. The scanner's actual misspell rule
    // matches runs of two or more ' ' characters (not tabs, not newlines).

    #[test]
    fn misspell_zero_for_single_space() {
        assert_eq!(count_misspellings(" "), 0.0);
    }

    #[test]
    fn misspell_counts_n_minus_one_for_space_run() {
        assert_eq!(count_misspellings("  "), 1.0);
        assert_eq!(count_misspellings("   "), 2.0);
        assert_eq!(count_misspellings("    "), 3.0);
    }

    #[test]
    fn misspell_sums_across_space_runs() {
        assert_eq!(count_misspellings("a  b   c"), 1.0 + 2.0);
    }

    #[test]
    fn misspell_ignores_tabs() {
        // Tabs are not matched by the scanner's misspell rule.
        assert_eq!(count_misspellings("\t\t\t"), 0.0);
    }

    #[test]
    fn misspell_zero_for_bare_leet_words() {
        // Without surrounding spaces, none of the abbreviation patterns fire.
        assert_eq!(count_misspellings("gr8"), 0.0);
        assert_eq!(count_misspellings("4ever"), 0.0);
        assert_eq!(count_misspellings("l8r"), 0.0);
        assert_eq!(count_misspellings("u4me"), 0.0);
    }

    #[test]
    fn misspell_fires_for_surrounded_abbrevs() {
        assert_eq!(count_misspellings("aa u bb"), 1.0);
        assert_eq!(count_misspellings("aa r bb"), 1.0);
        assert_eq!(count_misspellings("aa n bb"), 1.0);
        assert_eq!(count_misspellings("aa g bb"), 1.0);
        assert_eq!(count_misspellings("aa y bb"), 1.0);
        assert_eq!(count_misspellings("aa ur bb"), 1.0);
        assert_eq!(count_misspellings("aa gr8 bb"), 1.0);
    }

    #[test]
    fn misspell_abbrev_needs_space_on_both_sides() {
        // No leading space / no trailing space / end-of-input = no match.
        assert_eq!(count_misspellings("u"), 0.0);
        assert_eq!(count_misspellings("u bb"), 0.0);
        assert_eq!(count_misspellings("aa u"), 0.0);
        assert_eq!(count_misspellings("aa u\tbb"), 0.0);
        assert_eq!(count_misspellings("aa!u bb"), 0.0);
        assert_eq!(count_misspellings("aa u!bb"), 0.0);
    }

    #[test]
    fn misspell_abbrev_is_case_sensitive() {
        assert_eq!(count_misspellings("aa U bb"), 0.0);
        assert_eq!(count_misspellings("aa UR bb"), 0.0);
        assert_eq!(count_misspellings("aa GR8 bb"), 0.0);
    }

    #[test]
    fn misspell_abbrev_other_letters_do_not_fire() {
        // Only u, r, n, g, y are special among single letters.
        assert_eq!(count_misspellings("aa a bb"), 0.0);
        assert_eq!(count_misspellings("aa b bb"), 0.0);
        assert_eq!(count_misspellings("aa x bb"), 0.0);
        assert_eq!(count_misspellings("aa z bb"), 0.0);
    }

    #[test]
    fn misspell_abbrev_fires_once_per_surrounded_token() {
        assert_eq!(count_misspellings("a u r b"), 2.0);
        assert_eq!(count_misspellings("a ur u b"), 2.0);
        assert_eq!(count_misspellings("a u r n g y b"), 5.0);
    }

    #[test]
    fn misspell_combines_space_runs_and_abbrevs() {
        // "aa u bb  cc": one abbrev + one two-space run = 2.
        assert_eq!(count_misspellings("aa u bb  cc"), 2.0);
        // "aa   u   bb": two three-space runs (2 matches each) + one 'u' = 5.
        assert_eq!(count_misspellings("aa   u   bb"), 5.0);
    }

    // word_count follows the alphanumeric rule [a-zA-Z0-9]+ with REJECT.

    #[test]
    fn word_count_includes_digits_in_runs() {
        // "gr8" is a single alphanumeric run of length 3 -> 3*4/2 = 6.
        let features = extract_features("gr8");
        assert_eq!(features.word_length * 3.0, 6.0);
    }

    #[test]
    fn word_count_crosses_letter_digit_boundary() {
        // "u4me" is one run of length 4 -> 4*5/2 = 10.
        let features = extract_features("u4me");
        assert_eq!(features.word_length * 4.0, 10.0);
    }

    #[test]
    fn word_count_splits_on_non_alphanumeric() {
        // "ab 12" -> two runs of length 2 each -> 3 + 3 = 6.
        let features = extract_features("ab 12");
        assert_eq!(features.word_length * 5.0, 6.0);
    }

    // The flex rules only match ASCII [a-z]/[A-Z]. Non-ASCII Unicode letters
    // (like 'é' or 'É') must not count toward num_lowers / num_caps. num_total
    // counts bytes (matching flex '.' which matches any non-newline byte).

    #[test]
    fn num_lowers_excludes_non_ascii_letters() {
        // "cafe" + 'é' (UTF-8 0xC3 0xA9) = 5 bytes, 3 ASCII lowercase.
        let features = extract_features("café");
        assert!(
            (features.num_lowers - 3.0 / 5.0).abs() < 1e-9,
            "expected 3/5 = 0.6, got {}",
            features.num_lowers
        );
    }

    #[test]
    fn num_caps_excludes_non_ascii_letters() {
        // "JOS" + 'É' (UTF-8 0xC3 0x89) = 5 bytes, 3 ASCII uppercase.
        let features = extract_features("JOSÉ");
        assert!(
            (features.num_caps - 3.0 / 5.0).abs() < 1e-9,
            "expected 3/5 = 0.6, got {}",
            features.num_caps
        );
    }

    #[test]
    fn non_ascii_letters_neither_lower_nor_upper() {
        // "é" alone: 2 bytes, zero ASCII lowercase/uppercase.
        let features = extract_features("é");
        assert_eq!(features.num_lowers, 0.0);
        assert_eq!(features.num_caps, 0.0);
    }

    #[test]
    fn num_total_counts_bytes_not_codepoints() {
        // "é" = 2 bytes, so word_length = word_count / 2.
        // word_count is a run of [a-zA-Z0-9]+ which matches no bytes of 'é',
        // so word_count = 0 and word_length = 0.
        let features = extract_features("é");
        // Non-ASCII chars don't form word runs, so word_length == 0.
        assert_eq!(features.word_length, 0.0);
    }

    #[test]
    fn initial_cap_uses_ascii_boundary() {
        // "éHello" has no trailing whitespace, so rule 6 does not fire.
        // "éHello " (with trailing space) fires once.
        assert_eq!(initial_cap_count("éHello"), 0.0);
        assert_eq!(initial_cap_count("éHello "), 1.0);
    }

    #[test]
    fn intercap_uses_ascii_letter_pairs() {
        // "éHello": even though 'é' is a Unicode letter, flex only sees
        // ASCII letters. "Hello" alone has no inter-capitalization.
        let features = extract_features("éHello");
        assert_eq!(features.intercap, 0.0);
    }
}
