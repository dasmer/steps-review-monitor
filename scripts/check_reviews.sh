#!/usr/bin/env bash
set -euo pipefail

APP_ID="1602546738"
COUNTRY="us"
STATE_FILE=".data/review_count.txt"

mkdir -p .data

# Get current count from Apple Lookup API
json="$(curl -s "https://itunes.apple.com/lookup?id=${APP_ID}&country=${COUNTRY}")"
current_count="$(printf '%s' "$json" | jq -r '.results[0].userRatingCount')"

if [[ -z "${current_count}" || "${current_count}" == "null" ]]; then
  echo "Failed to read userRatingCount from API." >&2
  exit 1
fi

# Read previous count (default to 0 if missing)
if [[ -f "${STATE_FILE}" ]]; then
  prev_count="$(cat "${STATE_FILE}")"
else
  prev_count="0"
fi

echo "Current: ${current_count}, Previous: ${prev_count}"

changed="false"
diff="0"

if (( current_count > prev_count )); then
  changed="true"
  diff=$(( current_count - prev_count ))
  echo "${current_count}" > "${STATE_FILE}"
fi

# Export for later steps
{
  echo "changed=${changed}"
  echo "current_count=${current_count}"
  echo "prev_count=${prev_count}"
  echo "diff=${diff}"
} >> "$GITHUB_OUTPUT"
