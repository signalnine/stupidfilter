stupidfilter
============

A text classifier that detects low-quality writing. Uses an SVM model trained on stylistic features: capitalization patterns, punctuation density, word length, and spelling indicators.

## Usage

```bash
echo "your text here" | bin/stupidfilter data/c_rbf
```

Output: `0.0` = low-quality, `1.0` = acceptable. Values between indicate confidence.

Strip HTML and normalize whitespace before classification. See `classify.sh` for an example.

## Building

### C++ (original)

Dependencies:

```bash
# Debian/Ubuntu
sudo apt install g++ flex libboost-serialization-dev

# Fedora/RHEL
sudo dnf install gcc-c++ flex boost-devel
```

Build:

```bash
make
```

This produces `bin/stupidfilter`. The build uses system Boost headers (not the bundled 2008 versions in `thirdparty/boost.old`).

To rebuild the lexer from source:

```bash
flex -o stupidfilter.cpp fclassify.flex
```

### Rust (port)

Requires Rust 1.70+:

```bash
cd rust
cargo build --release
```

This produces `rust/target/release/stupidfilter`. Run it the same way:

```bash
echo "test text" | ./target/release/stupidfilter ../data/c_rbf
```

The Rust port produces identical classifications and runs 1.7–2.1× faster than C++.

## Project History

Originally released in 2008 by Rarefied Technologies under GPL v2. Updated in 2026 to build on modern systems (GCC 14, 64-bit Linux) and ported to Rust.
