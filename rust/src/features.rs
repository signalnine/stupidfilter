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

    // Count words
    let words: Vec<&str> = text.split_whitespace().collect();
    let word_count = words.len() as f64;

    // Count words starting with capital letter
    let initial_cap_count = words
        .iter()
        .filter(|w| {
            w.chars()
                .next()
                .map(|c| c.is_uppercase())
                .unwrap_or(false)
        })
        .count() as f64;

    // Count intercap (camelCase) words - lowercase followed by uppercase
    let intercap_re = Regex::new(r"[a-z]+[A-Z][a-z]*").unwrap();
    let intercap_count = intercap_re.find_iter(text).count() as f64;

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

/// Count misspellings and l33t speak patterns
fn count_misspellings(text: &str) -> f64 {
    let text_lower = text.to_lowercase();
    let mut count = 0.0;

    // Pattern: standalone "u" or "ur" as word (you/your)
    let u_re = Regex::new(r"(?i)\b[Uu][Rr]?\b").unwrap();
    count += u_re.find_iter(&text_lower).count() as f64;

    // Pattern: numbers mixed with letters (l33t speak)
    // e.g., gr8, l8r, 4ever, 2day, b4, etc.
    let leet_re = Regex::new(r"\b\w*[0-9]+\w*[a-zA-Z]+\w*\b|\b\w*[a-zA-Z]+\w*[0-9]+\w*\b").unwrap();
    count += leet_re.find_iter(text).count() as f64;

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
    fn test_extract_features_leet() {
        let features = extract_features("u r 2 cool 4 school");
        assert!(features.misspell > 0.0);
    }
}
