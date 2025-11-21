#!/bin/bash

# Auto-commit script - commits changes after 1 hour
REPO_DIR="/workspaces/DS1001-LABS-Projects"
LOG_FILE="$REPO_DIR/.git-auto-commit.log"

cd "$REPO_DIR"

# Check if there are any changes
if [[ -n $(git status --porcelain) ]]; then
    echo "[$(date)] Changes detected, committing..." >> "$LOG_FILE"
    git add -A
    git commit -m "Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "[$(date)] Commit completed" >> "$LOG_FILE"
else
    echo "[$(date)] No changes to commit" >> "$LOG_FILE"
fi
