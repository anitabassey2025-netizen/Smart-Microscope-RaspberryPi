#!/usr/bin/env bash
set -e

# Create logs directory if missing
mkdir -p logs

# Timestamp
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/${TS}_test_tests.log"

echo "=== RUN: $1 ===" | tee "$LOGFILE"
echo "Timestamp: $(date)" | tee -a "$LOGFILE"
echo "PWD: $(pwd)" | tee -a "$LOGFILE"
echo "Python: $(python3 --version)" | tee -a "$LOGFILE"
echo "Venv: $VIRTUAL_ENV" | tee -a "$LOGFILE"

echo -e "\n--- SYSTEM BEFORE ---" | tee -a "$LOGFILE"
uptime | tee -a "$LOGFILE"
free -h | tee -a "$LOGFILE"

echo -e "\n--- RUNNING (/usr/bin/time -v gives peak RAM + CPU) ---" | tee -a "$LOGFILE"

# THIS is the important part:
# Forward every argument EXACTLY as passed:
if command -v /usr/bin/time >/dev/null 2>&1; then
    /usr/bin/time -v python "$@" 2>&1 | tee -a "$LOGFILE"
else
    python "$@" 2>&1 | tee -a "$LOGFILE"
fi

