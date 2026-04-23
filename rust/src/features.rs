// Feature extraction module
// Extracts 8 text features for SVM classification

use regex::Regex;

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
pub fn extract_features(text: &str) -> Features {
    let chars: Vec<char> = text.chars().collect();
    let num_total = chars.len() as f64;

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

    // Count character types
    let num_lowers = chars.iter().filter(|c| c.is_lowercase()).count() as f64;
    let num_caps = chars.iter().filter(|c| c.is_uppercase()).count() as f64;
    let num_punct = chars.iter().filter(|c| is_punct(**c)).count() as f64;

    // word_count follows the flex rule [a-zA-Z0-9]+ with REJECT: for each
    // run of consecutive alphanumeric chars of length L, every starting
    // position within the run fires (L - offset) matches, summing to
    // L*(L+1)/2 -- the count of all non-empty substrings.
    let mut word_count = 0.0;
    let mut initial_cap_count = 0.0;
    let mut intercap_count = 0.0;

    let n = chars.len();

    let mut i = 0;
    while i < n {
        if chars[i].is_ascii_alphanumeric() {
            let start = i;
            while i < n && chars[i].is_ascii_alphanumeric() {
                i += 1;
            }
            let end = i;
            let run_len = end - start;
            word_count += (run_len * (run_len + 1) / 2) as f64;
        } else {
            i += 1;
        }
    }

    // initial_cap / intercap still use pure-letter runs to match the existing
    // flex rules (which look at [a-zA-Z] only, not digits).
    let mut j = 0;
    while j < n {
        if chars[j].is_alphabetic() {
            let start = j;
            while j < n && chars[j].is_alphabetic() {
                j += 1;
            }
            let end = j;
            let run_len = end - start;

            let is_at_word_boundary = start == 0 || !chars[start - 1].is_alphabetic();
            if is_at_word_boundary && run_len >= 1 {
                let first = chars[start];
                let second = if run_len >= 2 { Some(chars[start + 1]) } else { None };
                if first.is_uppercase() {
                    if let Some(s) = second {
                        if s.is_lowercase() {
                            initial_cap_count += 1.0;
                        }
                    } else {
                        initial_cap_count += 1.0;
                    }
                }
            }

            if run_len >= 2 {
                for k in start..(end - 1) {
                    if chars[k + 1].is_uppercase() {
                        intercap_count += 1.0;
                    }
                }
            }
        } else {
            j += 1;
        }
    }

    // Count repeated emphasis (!! or ??)
    let emphasis_re = Regex::new(r"[!]{2,}|[?]{2,}").unwrap();
    let repeat_emphasis = emphasis_re.find_iter(text).count() as f64;

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

/// Check if character is punctuation
fn is_punct(c: char) -> bool {
    matches!(
        c,
        '!' | '"'
            | '#'
            | '$'
            | '%'
            | '&'
            | '\''
            | '('
            | ')'
            | '*'
            | '+'
            | ','
            | '-'
            | '.'
            | '/'
            | ':'
            | ';'
            | '<'
            | '='
            | '>'
            | '?'
            | '@'
            | '['
            | '\\'
            | ']'
            | '^'
            | '_'
            | '`'
            | '{'
            | '|'
            | '}'
            | '~'
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_features_normal() {
        let features = extract_features("Hello world");
        assert!(features.num_lowers > 0.0);
        assert!(features.num_caps > 0.0);
    }

    #[test]
    fn test_extract_features_emphasis() {
        let features = extract_features("OMG!!! WOW???");
        assert!(features.repeat_emphasis >= 2.0);
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
}
