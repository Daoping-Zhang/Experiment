#!/bin/bash
# start_worker_mac.command — double-click to launch a MiniMPI student worker.
#
# 1) It asks for the teacher's IP:Port (default shown in the prompt).
# 2) Runs: python3 worker.py --server <ip>:<port>
# 3) Keeps the window open so you can see errors.
#
# First time on macOS: make it executable once (or right-click -> Open):
#   chmod +x scripts/start_worker_mac.command

cd "$(dirname "$0")/.."                     # classroom_minimpi
export PYTHONUTF8=1

DEFAULT="192.168.1.100:9000"                # change to your teacher's IP
printf 'Teacher IP:Port [%s]: ' "$DEFAULT"
read -r ADDR
[ -z "$ADDR" ] && ADDR="$DEFAULT"
case "$ADDR" in
  *:*) ;;
  *)   ADDR="$ADDR:9000" ;;                 # allow "192.168.1.100"
esac

echo
echo "Connecting to teacher $ADDR ..."
python3 -u worker.py --server "$ADDR"
RC=$?
echo
echo "Worker exited with code $RC."
read -r -p "Press Enter to close..."
