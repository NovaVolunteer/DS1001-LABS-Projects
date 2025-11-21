#!/bin/bash

# Background auto-commit daemon
# Monitors changes and commits after 1 hour of the last change

REPO_DIR="/workspaces/DS1001-LABS-Projects"
LOG_FILE="$REPO_DIR/.git-auto-commit.log"
PID_FILE="$REPO_DIR/.git-auto-commit.pid"
COMMIT_DELAY=3600  # 1 hour in seconds

cd "$REPO_DIR"

# Store PID
echo $$ > "$PID_FILE"

echo "[$(date)] Auto-commit daemon started (PID: $$)" >> "$LOG_FILE"

last_change_time=""

while true; do
    # Check for changes
    if [[ -n $(git status --porcelain) ]]; then
        current_time=$(date +%s)
        
        # If this is the first change detected
        if [[ -z "$last_change_time" ]]; then
            last_change_time=$current_time
            echo "[$(date)] Changes detected, will commit in 1 hour" >> "$LOG_FILE"
        else
            # Check if 1 hour has passed since last change
            time_diff=$((current_time - last_change_time))
            
            if [[ $time_diff -ge $COMMIT_DELAY ]]; then
                echo "[$(date)] Committing changes after 1 hour..." >> "$LOG_FILE"
                git add -A
                git commit -m "Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "[$(date)] Commit completed" >> "$LOG_FILE"
                last_change_time=""
            fi
        fi
    else
        # No changes, reset timer
        if [[ -n "$last_change_time" ]]; then
            echo "[$(date)] No changes remaining" >> "$LOG_FILE"
        fi
        last_change_time=""
    fi
    
    # Check every 60 seconds
    sleep 60
done
