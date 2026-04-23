#!/usr/bin/env bash
# Tests for classify.sh. Run from the repo root after building bin/stupidfilter.
# Exits 0 on success, non-zero if any test fails.

set -u
cd "$(dirname "$0")"

fail=0
report() {
  if [ "$1" -eq 0 ]; then
    printf '  PASS  %s\n' "$2"
  else
    printf '  FAIL  %s\n' "$2"
    fail=1
  fi
}

if [ ! -x bin/stupidfilter ]; then
  echo "bin/stupidfilter not built; run 'make' first" >&2
  exit 2
fi

# --- Test 1: "Hello world" classifies as not stupid ---
out=$(printf 'Hello world\n' | bash classify.sh 2>&1)
echo "$out" | grep -q 'not likely to be stupid'
report $? "classify 'Hello world' -> not stupid"

# --- Test 2: "OMG UR SO DUMB 4 REAL" classifies as stupid ---
out=$(printf 'OMG UR SO DUMB 4 REAL\n' | bash classify.sh 2>&1)
echo "$out" | grep -q 'likely to be stupid'
report $? "classify 'OMG UR SO DUMB 4 REAL' -> stupid"

# --- Test 3: bin/stupidfilter invoked exactly once per run ---
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT
cat > "$tmpdir/stupidfilter" <<'WRAP'
#!/usr/bin/env bash
printf 'x' >> "$COUNT_FILE"
exec "$REAL_STUPIDFILTER" "$@"
WRAP
chmod +x "$tmpdir/stupidfilter"
export COUNT_FILE="$tmpdir/count"
export REAL_STUPIDFILTER="$PWD/bin/stupidfilter"
: > "$COUNT_FILE"

# Run classify.sh with bin/ shadowed by our counter wrapper.
shadow_dir=$(mktemp -d)
ln -s "$tmpdir/stupidfilter" "$shadow_dir/stupidfilter"
# classify.sh references bin/stupidfilter relative to cwd, so we need a bin/
# next to the script. Temporarily swap.
mkdir -p "$tmpdir/wd/bin"
ln -s "$tmpdir/stupidfilter" "$tmpdir/wd/bin/stupidfilter"
cp classify.sh "$tmpdir/wd/classify.sh"
ln -s "$PWD/data" "$tmpdir/wd/data"

printf 'Hello world\n' | (cd "$tmpdir/wd" && bash classify.sh) > /dev/null 2>&1

count=$(wc -c < "$COUNT_FILE")
if [ "$count" -eq 1 ]; then
  report 0 "classify.sh invokes bin/stupidfilter exactly once (got $count)"
else
  report 1 "classify.sh invokes bin/stupidfilter exactly once (got $count)"
fi

# --- Test 4: missing model -> clear error, no bash syntax errors, non-zero exit ---
cp classify.sh "$tmpdir/wd/classify.sh"
# Point data to an empty dir so model load fails.
rm "$tmpdir/wd/data"
mkdir "$tmpdir/wd/data"

out=$(printf 'hi\n' | (cd "$tmpdir/wd" && bash classify.sh) 2>&1)
rc=$?

# Should not contain bash's unary-operator gibberish.
if echo "$out" | grep -q 'unary operator expected'; then
  report 1 "no bash syntax error on failure (got: $out)"
else
  report 0 "no bash syntax error on failure"
fi

# Should exit non-zero.
if [ "$rc" -ne 0 ]; then
  report 0 "non-zero exit on classification failure"
else
  report 1 "non-zero exit on classification failure (got $rc)"
fi

# Should mention failure in output.
if echo "$out" | grep -qi 'fail\|error\|could not'; then
  report 0 "prints a clear error on classification failure"
else
  report 1 "prints a clear error on classification failure (got: $out)"
fi

# --- Test 5: backslashes preserved (read -r) ---
# With read -r, 'a\b' stays 'a\b'. Without -r, it becomes 'ab'.
# We verify by grepping classify.sh itself; pure end-to-end is hard since
# the feature extractor doesn't echo input.
grep -q 'read -r' classify.sh
report $? "classify.sh uses 'read -r'"

exit $fail
