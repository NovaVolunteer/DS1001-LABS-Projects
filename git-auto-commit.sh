#!/bin/bash

# Control script for auto-commit daemon

REPO_DIR="/workspaces/DS1001-LABS-Projects"
DAEMON_SCRIPT="$REPO_DIR/.git-auto-commit-daemon.sh"
PID_FILE="$REPO_DIR/.git-auto-commit.pid"
LOG_FILE="$REPO_DIR/.git-auto-commit.log"

case "$1" in
    start)
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Auto-commit daemon is already running (PID: $PID)"
                exit 1
            fi
        fi
        
        echo "Starting auto-commit daemon..."
        nohup "$DAEMON_SCRIPT" >> "$LOG_FILE" 2>&1 &
        sleep 1
        
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            echo "Auto-commit daemon started (PID: $PID)"
            echo "Changes will be committed automatically 1 hour after detection"
            echo "View log: tail -f $LOG_FILE"
        else
            echo "Failed to start daemon"
            exit 1
        fi
        ;;
        
    stop)
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Stopping auto-commit daemon (PID: $PID)..."
                kill "$PID"
                rm -f "$PID_FILE"
                echo "Daemon stopped"
            else
                echo "Daemon not running (stale PID file)"
                rm -f "$PID_FILE"
            fi
        else
            echo "Daemon not running (no PID file)"
        fi
        ;;
        
    status)
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            if ps -p "$PID" > /dev/null 2>&1; then
                echo "Auto-commit daemon is running (PID: $PID)"
                exit 0
            else
                echo "Daemon not running (stale PID file)"
                exit 1
            fi
        else
            echo "Daemon not running"
            exit 1
        fi
        ;;
        
    restart)
        $0 stop
        sleep 2
        $0 start
        ;;
        
    log)
        if [[ -f "$LOG_FILE" ]]; then
            tail -f "$LOG_FILE"
        else
            echo "No log file found"
        fi
        ;;
        
    *)
        echo "Usage: $0 {start|stop|status|restart|log}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the auto-commit daemon"
        echo "  stop    - Stop the auto-commit daemon"
        echo "  status  - Check daemon status"
        echo "  restart - Restart the daemon"
        echo "  log     - View the log file"
        exit 1
        ;;
esac
