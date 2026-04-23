#!/usr/bin/env bash

echo "Enter text to be classified, hit return to run classification."
read -r text

if ! score=$(echo "$text" | sed -r 's/ +/ /g' | bin/stupidfilter data/c_rbf); then
  echo "classification failed: could not run bin/stupidfilter" >&2
  exit 1
fi

case "$score" in
  1.000000) echo "Text is not likely to be stupid." ;;
  0.000000) echo "Text is likely to be stupid." ;;
  *)
    echo "classification failed: unexpected output from bin/stupidfilter: '$score'" >&2
    exit 1
    ;;
esac
