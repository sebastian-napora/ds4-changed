#!/usr/bin/env bash
#
# High-level logged workflow for repository work plus optional model tracing.
#
# This script creates one parent session under summary-logs/ and can cover:
# assessment, improvements notes, validation, verification, setup, creation,
# editing, reading, content checks, git history, tools/MCP checks, terminal
# commands, per-test logs, and optional DS4 model startup with expert tracing.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${REPO_WORK_LOG_ROOT:-"$SCRIPT_DIR/summary-logs"}"
SESSION_NAME="${REPO_WORK_SESSION_NAME:-repo-work-$(date +%Y%m%d-%H%M%S)}"
SESSION_DIR="$LOG_ROOT/$SESSION_NAME"

RUN_MODEL=0
EXPERT_TRACE="summary"
SMOKE_PROMPT="Briefly answer: repository logging smoke test."
RUN_BUILD=0
RUN_TESTS=0
RUN_MCP=0
BASELINE=1
NOTES=()
PHASE_NAMES=()
PHASE_COMMANDS=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Creates a complete logged repository-work session under summary-logs/.

Core options:
  --session-name NAME       Use a custom session directory name.
  --note TEXT               Add a note to the session. May be repeated.
  --with-build              Run make inside the repo assessment.
  --with-tests              Run tests one by one via run-tests-logged.sh.
  --with-mcp                Try local MCP CLI inspection when available.
  --no-baseline             Skip initial repo-session.sh assessment.

Model/expert tracing:
  --start-model             Start start-low-gpu-summary.sh in this session.
  --expert-summary          Save router expert summary logs. Default.
  --expert-detail           Save sampled per-row expert IDs/weights/probs.
  --no-expert-trace         Start model without DS4_ROUTER_TRACE.
  --smoke-prompt TEXT       Prompt sent after server readiness to create
                            expert routing logs. Default: short smoke prompt.
  --no-smoke-prompt         Do not send an automatic prompt.

Logged phase commands:
  --setup-cmd COMMAND       Log setup command.
  --create-cmd COMMAND      Log creation command.
  --edit-cmd COMMAND        Log editing command.
  --read-cmd COMMAND        Log reading command.
  --check-cmd COMMAND       Log checking command.
  --verify-cmd COMMAND      Log verification command.
  --validate-cmd COMMAND    Log validation command.
  --tool-cmd COMMAND        Log tool command.
  --cmd COMMAND             Log general command.

Environment:
  REPO_WORK_LOG_ROOT        Log root. Default: ./summary-logs
  REPO_WORK_SESSION_NAME    Session directory name. Default: repo-work-timestamp
  DS4_ROUTER_TRACE_LAYER    Optional layer filter for --expert-detail.
  DS4_ROUTER_TRACE_LIMIT    Optional row limit for --expert-detail.
  DS4_ROUTER_TRACE_POS      Optional position filter for --expert-detail.

Examples:
  $0 --with-tests --with-mcp
  $0 --start-model --with-tests --expert-summary
  $0 --start-model --expert-detail --smoke-prompt "Say hello in one sentence."
  $0 --setup-cmd "./setup-mtp.sh --setup-only" --verify-cmd "git status --short"
EOF
}

add_phase() {
    PHASE_NAMES+=("$1")
    PHASE_COMMANDS+=("$2")
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --session-name)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --session-name" >&2
                exit 1
            fi
            SESSION_NAME="$1"
            SESSION_DIR="$LOG_ROOT/$SESSION_NAME"
            ;;
        --note)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --note" >&2
                exit 1
            fi
            NOTES+=("$1")
            ;;
        --with-build)
            RUN_BUILD=1
            ;;
        --with-tests)
            RUN_TESTS=1
            ;;
        --with-mcp)
            RUN_MCP=1
            ;;
        --no-baseline)
            BASELINE=0
            ;;
        --start-model)
            RUN_MODEL=1
            ;;
        --expert-summary)
            EXPERT_TRACE="summary"
            ;;
        --expert-detail)
            EXPERT_TRACE="1"
            ;;
        --no-expert-trace)
            EXPERT_TRACE=""
            ;;
        --smoke-prompt)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --smoke-prompt" >&2; exit 1; }
            SMOKE_PROMPT="$1"
            ;;
        --no-smoke-prompt)
            SMOKE_PROMPT=""
            ;;
        --setup-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --setup-cmd" >&2; exit 1; }
            add_phase "setup" "$1"
            ;;
        --create-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --create-cmd" >&2; exit 1; }
            add_phase "creation" "$1"
            ;;
        --edit-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --edit-cmd" >&2; exit 1; }
            add_phase "editing" "$1"
            ;;
        --read-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --read-cmd" >&2; exit 1; }
            add_phase "reading" "$1"
            ;;
        --check-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --check-cmd" >&2; exit 1; }
            add_phase "checking" "$1"
            ;;
        --verify-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --verify-cmd" >&2; exit 1; }
            add_phase "verification" "$1"
            ;;
        --validate-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --validate-cmd" >&2; exit 1; }
            add_phase "validation" "$1"
            ;;
        --tool-cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --tool-cmd" >&2; exit 1; }
            add_phase "tool" "$1"
            ;;
        --cmd)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --cmd" >&2; exit 1; }
            add_phase "command" "$1"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
    shift
done

mkdir -p "$SESSION_DIR/phases"

run_shell_log() {
    local name="$1"
    local cmd="$2"
    local log="$3"
    local status

    {
        echo "# $name"
        echo "\$ $cmd"
        echo "started_at=$(date -Iseconds)"
        echo
    } > "$log"

    set +e
    bash -lc "$cmd" >> "$log" 2>&1
    status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$log"

    return "$status"
}

write_readme() {
    {
        echo "# Repository Agent Work Session"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Started: $(date -Iseconds)"
        echo "- Repository: $SCRIPT_DIR"
        echo "- Start model: $RUN_MODEL"
        echo "- Expert trace: ${EXPERT_TRACE:-disabled}"
        echo "- Smoke prompt: ${SMOKE_PROMPT:-disabled}"
        echo "- Build: $RUN_BUILD"
        echo "- Tests: $RUN_TESTS"
        echo "- MCP: $RUN_MCP"
        echo
        echo "## What This Covers"
        echo
        echo "- improvements: improvements.md and repo/improvements.md"
        echo "- assessment: repo/assessment.md"
        echo "- validation: repo/validation.md and phase logs"
        echo "- verification: git/test/build logs and phase logs"
        echo "- setup/creation/editing/reading/checking/tools/commands: phases/*.log"
        echo "- content checks: repo/content-scan.log"
        echo "- git history: repo/git-history.log"
        echo "- MCP: repo/mcp.log when --with-mcp is used"
        echo "- model expert usage: model/run.log when --start-model is used"
        echo
        echo "## Notes"
        echo
        if [ "${#NOTES[@]}" -eq 0 ]; then
            echo "No notes provided."
        else
            local note
            for note in "${NOTES[@]}"; do
                echo "- $note"
            done
        fi
    } > "$SESSION_DIR/README.md"
}

write_improvements() {
    {
        echo "# Improvements"
        echo
        echo "- Add candidate improvements here as they are discovered."
        echo "- Link each item to the relevant log under this session."
        echo "- Validate changes with phase logs, build logs, or test logs."
    } > "$SESSION_DIR/improvements.md"
}

run_baseline() {
    local args=("--session-name" "repo")
    local note

    for note in "${NOTES[@]}"; do
        args+=("--note" "$note")
    done

    if [ "$RUN_BUILD" -ne 0 ]; then
        args+=("--with-build")
    fi
    if [ "$RUN_TESTS" -ne 0 ] && [ "$RUN_MODEL" -eq 0 ]; then
        args+=("--with-tests")
    fi
    if [ "$RUN_MCP" -ne 0 ]; then
        args+=("--with-mcp")
    fi

    {
        echo "\$ ./repo-session.sh ${args[*]}"
        echo "started_at=$(date -Iseconds)"
        echo
    } > "$SESSION_DIR/repo-session.log"

    echo "Running repository baseline. Log: $SESSION_DIR/repo-session.log"
    if [ "$RUN_TESTS" -ne 0 ] && [ "$RUN_MODEL" -ne 0 ]; then
        echo "Deferring tests until after the model exits to avoid the ds4 lock."
        echo "Deferring tests until after the model exits to avoid the ds4 lock." >> "$SESSION_DIR/repo-session.log"
    fi

    set +e
    REPO_SESSION_LOG_ROOT="$SESSION_DIR" "$SCRIPT_DIR/repo-session.sh" "${args[@]}" 2>&1 | tee -a "$SESSION_DIR/repo-session.log"
    local status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$SESSION_DIR/repo-session.log"

    return "$status"
}

run_phases() {
    local status=0
    local i

    for i in "${!PHASE_COMMANDS[@]}"; do
        local n=$((i + 1))
        local phase="${PHASE_NAMES[$i]}"
        local cmd="${PHASE_COMMANDS[$i]}"
        local log
        log="$SESSION_DIR/phases/$(printf "%02d" "$n")-$phase.log"
        echo "Running phase '$phase'. Log: $log"
        run_shell_log "$phase" "$cmd" "$log" || status=1
    done

    return "$status"
}

run_model() {
    local log="$SESSION_DIR/model-start.log"
    local status

    {
        echo "\$ ./start-low-gpu-summary.sh"
        echo "started_at=$(date -Iseconds)"
        echo "expert_trace=${EXPERT_TRACE:-disabled}"
        echo "smoke_prompt=${SMOKE_PROMPT:-disabled}"
        echo
    } > "$log"

    echo "Starting model. Live output is also saved to: $log"
    echo "Session folder: $SESSION_DIR/model"
    if [ -n "$SMOKE_PROMPT" ]; then
        echo "A smoke prompt will be sent after the server is ready so expert routing appears in model/run.log."
    else
        echo "No smoke prompt configured; expert routing appears only after you send a request."
    fi
    echo "Press Ctrl+C to stop the model. Final repo snapshot/tests run after shutdown."

    set +e
    DS4_SUMMARY_LOG_ROOT="$SESSION_DIR" \
    DS4_SUMMARY_SESSION_NAME="model" \
    DS4_SUMMARY_FINAL_REPO_SESSION=1 \
    DS4_SUMMARY_FINAL_TESTS="$RUN_TESTS" \
    DS4_SUMMARY_FINAL_MCP="$RUN_MCP" \
    DS4_SUMMARY_SMOKE_PROMPT="$SMOKE_PROMPT" \
    DS4_ROUTER_TRACE="$EXPERT_TRACE" \
    "$SCRIPT_DIR/start-low-gpu-summary.sh" 2>&1 | tee -a "$log"
    status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$log"

    return "$status"
}

write_summary() {
    local final_status="$1"
    {
        echo "# Repository Work Summary"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Updated: $(date -Iseconds)"
        echo "- Status: $final_status"
        echo "- Directory: $SESSION_DIR"
        echo
        echo "## Main Logs"
        echo
        find "$SESSION_DIR" -maxdepth 2 -type f | sort | sed "s#^$SESSION_DIR/#- #"
    } > "$SESSION_DIR/summary.md"
}

write_readme
write_improvements

overall_status=0

if [ "$BASELINE" -ne 0 ]; then
    run_baseline || overall_status=1
fi

if [ "${#PHASE_COMMANDS[@]}" -gt 0 ]; then
    run_phases || overall_status=1
fi

if [ "$RUN_MODEL" -ne 0 ]; then
    run_model || overall_status=1
fi

if [ "$overall_status" -eq 0 ]; then
    write_summary "completed"
else
    write_summary "completed-with-errors"
fi

echo "Repository work session written to: $SESSION_DIR"
exit "$overall_status"
