#!/bin/bash
# Sync nowcast data from the Pi (or mounted volume) to local backtest_data/
#
# Usage:
#   ./scripts/sync_backtest_data.sh                    # from mounted volume
#   ./scripts/sync_backtest_data.sh pi:/media/nowcast_data/  # from Pi via SSH
#
# Safe by design:
#   - Read-only source (rsync, no delete)
#   - Dry-run first, real run only after confirmation
#   - Verifies source exists before starting
#   - Destination is gitignored (backtest_data/)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$REPO_ROOT/backtest_data"

# Default source: mounted Pi volume
SOURCE="${1:-/Volumes/media/nowcast_data/}"

# Verify we're in the right repo
if [[ ! -f "$REPO_ROOT/CLAUDE.md" ]]; then
    echo "ERROR: Not in the rain-incoming repo root ($REPO_ROOT)" >&2
    exit 1
fi

# Verify source exists
if [[ "$SOURCE" != *:* ]]; then
    # Local path - check it exists
    if [[ ! -d "$SOURCE" ]]; then
        echo "ERROR: Source not found: $SOURCE" >&2
        echo "Is the Pi volume mounted? Try: ls /Volumes/media/" >&2
        exit 1
    fi
fi

# Ensure destination exists
mkdir -p "$DEST"

# Ensure gitignored
if ! grep -q "backtest_data" "$REPO_ROOT/.gitignore" 2>/dev/null; then
    echo "backtest_data/" >> "$REPO_ROOT/.gitignore"
    echo "Added backtest_data/ to .gitignore"
fi

echo "Source:      $SOURCE"
echo "Destination: $DEST"
echo ""

# Dry run first
echo "=== DRY RUN ==="
rsync -av --dry-run "$SOURCE" "$DEST/" | tail -5
echo ""

TOTAL=$(rsync -av --dry-run "$SOURCE" "$DEST/" 2>/dev/null | grep "total size" || true)
echo "$TOTAL"
echo ""

read -p "Proceed with sync? [y/N] " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Real sync
echo "=== SYNCING ==="
rsync -av --progress "$SOURCE" "$DEST/"

echo ""
echo "Done. Data at: $DEST"
du -sh "$DEST"
