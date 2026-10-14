#!/usr/bin/env bash
# Prints a Markdown summary of a test_comprehensive log: which distance code
# ran, the totals, every failed test, and any sanitizer report. Used by the
# nightly workflow to build the report shown on the run's page.
#
# Usage: summarize_tests.sh <log file> <title>
set -u
log="$1"
title="$2"
echo "### $title"
echo
if [ ! -s "$log" ]; then
    echo "No test output: the build failed or the tests never started. See the job's log."
    exit 0
fi
kernel=$(grep -m1 -o 'kernel: [a-z0-9]*' "$log" || true)
[ -n "$kernel" ] && echo "Distance code: \`${kernel#kernel: }\`" && echo
totals=$(grep -E '^[0-9]+ passed, [0-9]+ failed' "$log" | tail -1 || true)
if [ -n "$totals" ]; then
    echo "**$totals**"
else
    echo "**The run stopped before printing its totals** (a crash, a sanitizer report or a timeout). See the log artifact."
fi
failed=$(sed -n '/^Failed tests:/,$p' "$log" | grep -E '^  [a-z0-9_]+\.' || true)
if [ -n "$failed" ]; then
    echo
    echo "Failed tests:"
    echo "$failed" | sed -e 's/^  /- `/' -e 's/$/`/'
fi
pattern='ERROR: (AddressSanitizer|LeakSanitizer)|runtime error:|WARNING: ThreadSanitizer'
if grep -qE "$pattern" "$log"; then
    echo
    echo "A sanitizer reported a problem (first lines):"
    echo
    echo '```'
    grep -m5 -E "$pattern" "$log"
    echo '```'
fi