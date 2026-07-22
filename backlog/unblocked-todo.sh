#!/usr/bin/env bash
# List "To Do" tasks whose dependencies (if any) are all Done.
set -euo pipefail
cd "$(dirname "$0")"

frontmatter() {
  awk '/^---$/{c++; next} c==1' "$1"
}

declare -A status_of
for f in tasks/*.md completed/*.md archive/tasks/*.md; do
  [ -f "$f" ] || continue
  fm=$(frontmatter "$f")
  id=$(printf '%s\n' "$fm" | yq -r '.id')
  st=$(printf '%s\n' "$fm" | yq -r '.status')
  status_of["$id"]="$st"
done

for f in tasks/*.md; do
  [ -f "$f" ] || continue
  fm=$(frontmatter "$f")
  st=$(printf '%s\n' "$fm" | yq -r '.status')
  [ "$st" = "To Do" ] || continue

  id=$(printf '%s\n' "$fm" | yq -r '.id')
  title=$(printf '%s\n' "$fm" | yq -r '.title')
  mapfile -t deps < <(printf '%s\n' "$fm" | yq -o=json '.dependencies // []' | jq -r '.[]')

  blocked=false
  for d in "${deps[@]:-}"; do
    [ -z "$d" ] && continue
    if [ "${status_of[$d]:-MISSING}" != "Done" ]; then
      blocked=true
      break
    fi
  done

  if [ "$blocked" = false ]; then
    echo "$id - $title"
  fi
done
