#!/usr/bin/env bash
# Decides which kinds of files changed between two commits, so CI can skip the
# builds and tests when only documentation changed. Prints two lines for
# $GITHUB_OUTPUT:
#   code=true|false   anything other than documentation changed
#   docs=true|false   documentation changed
#
# Usage: classify_changes.sh <base commit> <head commit>
#
# Documentation is docs/, any .md file, and LICENSE. Everything else is code,
# including the CI workflows themselves, so a CI change is always tested by CI.
# The comparison starts at the merge base, so only the branch's own changes
# count. Renames list both names, so moving a code file into docs/ is a code
# change. When in doubt (no base, as on the first push of a branch, or a base
# that is not available), everything runs.
set -euo pipefail
base="${1:-}"
head="${2:-HEAD}"
if [[ -z "$base" || "$base" =~ ^0+$ ]] || ! git cat-file -e "${base}^{commit}" 2>/dev/null; then
    echo "code=true"
    echo "docs=true"
    exit 0
fi
code=false
docs=false
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    case "$file" in
        docs/* | *.md | LICENSE) docs=true ;;
        *) code=true ;;
    esac
done < <(git diff --name-only --no-renames "${base}...${head}")
echo "code=$code"
echo "docs=$docs"