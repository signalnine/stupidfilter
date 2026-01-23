# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StupidFilter is a C++ text classification tool using Support Vector Machines (SVM) to detect "stupid" vs "non-stupid" text. It reads from stdin and outputs a classification score (0.0 = stupid, 1.0 = non-stupid).

## Build Commands

```bash
# Install dependencies (Ubuntu/Debian)
sudo apt-get install build-essential flex libboost-serialization-dev

# Build
make

# Clean build artifacts
make clean

# Install to /usr/bin
make install
```

## Running the Application

```bash
# Direct usage
bin/stupidfilter data/c_rbf
# Then type text, press Ctrl+D for EOF

# Pipeline usage (recommended - normalizes whitespace)
echo "your text here" | sed -r 's/ +/ /g' | bin/stupidfilter data/c_rbf
```

## Architecture

**Classification Pipeline:**
```
Text Input → Flex Tokenizer → Feature Extraction → Feature Scaling → SVM Prediction → Score
```

**Key Components:**

- **stupidfilter.cpp**: Main entry point containing flex-generated lexical scanner and feature extraction logic. Extracts 8 text metrics: lowercase/capital letter counts, punctuation count, word count/length, initial capitalization, intercap words, repeated emphasis, misspelling indicators.

- **SVMUtil.h/.cpp**: Wrapper around libsvm providing model loading (`Load`), feature scaling (`ScaleNode`), cross-validation, and parameter search. Uses Boost serialization for model persistence.

- **parametersearch.h/.cpp**: Grid search optimization for SVM C and gamma parameters with multi-level refinement.

- **data/c_rbf.mod + c_rbf.sf**: Pre-trained RBF kernel SVM model and associated scale factors.

- **thirdparty/libsvm/**: Bundled libsvm library for SVM training/prediction.

## Build Notes

- Requires system Boost serialization library (`libboost-serialization-dev`)
- The `BOOSTLIB` path in Makefile may need adjustment: `/usr/lib/x86_64-linux-gnu` on Debian/Ubuntu, `/usr/lib64` on RHEL/Fedora
- Compilation uses g++ with C++11 standard
- Object files output to `bin/`

## Licensing

- Source code: GPL v2
- Data files (data/): CC-BY-NC-SA 3.0
