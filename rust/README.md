# stupidfilter-rs

Rust port of the StupidFilter text classifier.

## Building

```bash
# Install Rust if needed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build release version
cargo build --release

# Binary will be at target/release/stupidfilter
```

## Usage

```bash
# Uses the same model files as the C++ version
echo "Hello world" | ./target/release/stupidfilter ../data/c_rbf
# Output: 1.000000 (non-stupid)

echo "OMG ur SO DUMB!!!" | ./target/release/stupidfilter ../data/c_rbf
# Output: 0.000000 (stupid)
```

## Architecture

- `src/main.rs` - CLI entry point
- `src/features.rs` - Text feature extraction (8 features)
- `src/svm.rs` - SVM model loading and RBF kernel prediction

## Features Extracted

1. **num_lowers** - Ratio of lowercase letters
2. **num_caps** - Ratio of uppercase letters
3. **num_punct** - Ratio of punctuation
4. **repeat_emphasis** - Count of repeated `!!` or `??`
5. **initial_cap** - Ratio of words starting with capital
6. **intercap** - Ratio of camelCase words
7. **word_length** - Words per character
8. **misspell** - Count of l33t speak patterns

## Model Format

Uses standard libsvm text format (`.mod`) and simple scale factors (`.sf`).
Compatible with the pre-trained `data/c_rbf` model.
