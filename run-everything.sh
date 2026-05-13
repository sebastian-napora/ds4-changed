#!/usr/bin/env bash
#
# One-command DS4 repository validation workflow.
#
# Default flow:
#   1. Create one parent summary-logs/everything-* session.
#   2. Capture repository/tool/MCP baseline logs.
#   3. Run tests one by one with separated logs.
#   4. Start the low-GPU DS4 server with expert tracing.
#   5. Ask the model to inspect repository/test cases one by one.
#   6. Save real model responses, router/expert traces, and final summary.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${DS4_EVERYTHING_LOG_ROOT:-"$SCRIPT_DIR/summary-logs"}"
SESSION_NAME="${DS4_EVERYTHING_SESSION_NAME:-everything-$(date +%Y%m%d-%H%M%S)}"
SESSION_DIR="$LOG_ROOT/$SESSION_NAME"

RUN_TESTS=1
RUN_MCP=1
RUN_BUILD=0
START_MODEL=1
KEEP_MODEL_RUNNING=0
EXPERT_TRACE="summary"
THINK="${DS4_MODEL_CASE_THINK:-true}"
MAX_TOKENS="${DS4_MODEL_CASE_MAX_TOKENS:-1024}"
TEMPERATURE="${DS4_MODEL_CASE_TEMPERATURE:-0}"
NO_DEFAULT_CASES=0
CUSTOM_PROMPTS=()
CUSTOM_CASE_FILES=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs the complete logged DS4 repository workflow in one command.

Default:
  - repository baseline snapshot
  - MCP/tool checks
  - tests one by one
  - low-GPU model startup
  - model review of repo/test cases
  - expert routing logs
  - final summary

Options:
  --session-name NAME       Use summary-logs/NAME.
  --no-tests               Skip run-tests-logged.sh.
  --no-mcp                 Skip MCP inspection in repo sessions.
  --with-build             Include make in repo snapshots.
  --no-start-model         Do not start model; use an already-running server.
  --keep-model-running     Leave model running after model review completes.
  --expert-summary         Enable DS4_ROUTER_TRACE=summary. Default.
  --expert-detail          Enable DS4_ROUTER_TRACE=1.
  --no-expert-trace        Start model without router trace.
  --think                  Send "think": true to model review. Default.
  --no-think               Send "think": false to model review.
  --max-tokens N           Max model response tokens. Default: 1024.
  --temperature N          Model review temperature. Default: 0.
  --prompt TEXT            Add custom model review prompt. May be repeated.
  --case-file FILE         Add custom model review case file. May be repeated.
  --no-default-cases       Only run test/custom model review cases.
  --help                   Show this help.

Environment:
  DS4_HOST                 DS4 host. Default comes from child scripts.
  DS4_PORT                 DS4 port. Default comes from child scripts.
  DS4_MODEL                Model path used by start-low-gpu.sh.
  DS4_EVERYTHING_LOG_ROOT  Log root. Default: ./summary-logs

Examples:
  $0
  $0 --expert-detail --max-tokens 2048
  $0 --no-start-model --prompt "Review only the current repository logs."
  $0 --keep-model-running --with-build
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --session-name)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --session-name" >&2; exit 1; }
            SESSION_NAME="$1"
            SESSION_DIR="$LOG_ROOT/$SESSION_NAME"
            ;;
        --no-tests)
            RUN_TESTS=0
            ;;
        --no-mcp)
            RUN_MCP=0
            ;;
        --with-build)
            RUN_BUILD=1
            ;;
        --no-start-model)
            START_MODEL=0
            ;;
        --keep-model-running)
            KEEP_MODEL_RUNNING=1
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
        --think)
            THINK="true"
            ;;
        --no-think)
            THINK="false"
            ;;
        --max-tokens)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --max-tokens" >&2; exit 1; }
            MAX_TOKENS="$1"
            ;;
        --temperature)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --temperature" >&2; exit 1; }
            TEMPERATURE="$1"
            ;;
        --prompt)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --prompt" >&2; exit 1; }
            CUSTOM_PROMPTS+=("$1")
            ;;
        --case-file)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --case-file" >&2; exit 1; }
            CUSTOM_CASE_FILES+=("$1")
            ;;
        --no-default-cases)
            NO_DEFAULT_CASES=1
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

mkdir -p "$SESSION_DIR"

TEST_DIR="$SESSION_DIR/tests"
BASELINE_DIR_NAME="repo-baseline"
FINAL_DIR_NAME="repo-final"
MODEL_REVIEW_NAME="model-review"
RUN_LOG="$SESSION_DIR/run.log"

log_line() {
    local msg="$1"
    printf "[%s] %s\n" "$(date -Iseconds)" "$msg" | tee -a "$RUN_LOG"
}

run_step() {
    local name="$1"
    shift
    local log="$SESSION_DIR/$name.log"
    local status

    {
        echo "# $name"
        echo "\$ $*"
        echo "started_at=$(date -Iseconds)"
        echo
    } > "$log"

    log_line "Starting $name"
    set +e
    "$@" >> "$log" 2>&1
    status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$log"

    if [ "$status" -eq 0 ]; then
        log_line "Finished $name"
    else
        log_line "Finished $name with exit_status=$status"
    fi

    return "$status"
}

write_readme() {
    {
        echo "# DS4 Everything Session"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Started: $(date -Iseconds)"
        echo "- Repository: $SCRIPT_DIR"
        echo "- Tests: $RUN_TESTS"
        echo "- MCP: $RUN_MCP"
        echo "- Build in repo snapshots: $RUN_BUILD"
        echo "- Start model: $START_MODEL"
        echo "- Keep model running: $KEEP_MODEL_RUNNING"
        echo "- Expert trace: ${EXPERT_TRACE:-disabled}"
        echo "- Think: $THINK"
        echo "- Max tokens: $MAX_TOKENS"
        echo "- Temperature: $TEMPERATURE"
        echo
        echo "## Main Outputs"
        echo
        echo "- run.log"
        echo "- repo-baseline/"
        echo "- tests/"
        echo "- model-review/"
        echo "- repo-final/"
        echo "- summary.md"
        echo
        echo "Model responses are in model-review/model-responses/."
        echo "Expert/router traces are in model-review/model-runtime/run.log when the model is started by this script."
    } > "$SESSION_DIR/README.md"
}

build_repo_session_args() {
    REPO_SESSION_ARGS=()
    if [ "$RUN_BUILD" -ne 0 ]; then
        REPO_SESSION_ARGS+=("--with-build")
    fi
    if [ "$RUN_MCP" -ne 0 ]; then
        REPO_SESSION_ARGS+=("--with-mcp")
    fi
}

run_baseline_repo_session() {
    build_repo_session_args
    REPO_SESSION_ARGS+=("--session-name" "$BASELINE_DIR_NAME")
    REPO_SESSION_ARGS+=("--note" "Baseline snapshot for all-in-one DS4 workflow $SESSION_NAME.")
    REPO_SESSION_ARGS+=("--cmd" "git status --short")
    REPO_SESSION_ARGS+=("--cmd" "git diff --stat")

    run_step "01-repo-baseline" env REPO_SESSION_LOG_ROOT="$SESSION_DIR" "$SCRIPT_DIR/repo-session.sh" "${REPO_SESSION_ARGS[@]}"
}

run_tests() {
    if [ "$RUN_TESTS" -eq 0 ]; then
        log_line "Skipping tests by request"
        mkdir -p "$TEST_DIR"
        echo "Tests skipped by run-everything.sh." > "$TEST_DIR/README.md"
        return 0
    fi

    run_step "02-tests" "$SCRIPT_DIR/run-tests-logged.sh" --log-dir "$TEST_DIR"
}

run_model_review() {
    local args=("--session-name" "$MODEL_REVIEW_NAME")

    if [ "$START_MODEL" -ne 0 ]; then
        args+=("--start-model")
    fi
    if [ "$KEEP_MODEL_RUNNING" -ne 0 ]; then
        args+=("--keep-model-running")
    fi
    case "$EXPERT_TRACE" in
        summary)
            args+=("--expert-summary")
            ;;
        1)
            args+=("--expert-detail")
            ;;
        "")
            args+=("--no-expert-trace")
            ;;
    esac
    if [ "$THINK" = "true" ]; then
        args+=("--think")
    else
        args+=("--no-think")
    fi
    args+=("--max-tokens" "$MAX_TOKENS")
    args+=("--temperature" "$TEMPERATURE")

    if [ "$RUN_TESTS" -ne 0 ] && [ -d "$TEST_DIR" ]; then
        args+=("--test-log-dir" "$TEST_DIR")
    fi
    if [ "$NO_DEFAULT_CASES" -ne 0 ]; then
        args+=("--no-default-cases")
    fi

    local prompt
    for prompt in "${CUSTOM_PROMPTS[@]}"; do
        args+=("--prompt" "$prompt")
    done

    local case_file
    for case_file in "${CUSTOM_CASE_FILES[@]}"; do
        args+=("--case-file" "$case_file")
    done

    run_step "03-model-review" env DS4_MODEL_CASE_LOG_ROOT="$SESSION_DIR" "$SCRIPT_DIR/repo-model-review.sh" "${args[@]}"
}

run_final_repo_session() {
    build_repo_session_args
    REPO_SESSION_ARGS+=("--session-name" "$FINAL_DIR_NAME")
    REPO_SESSION_ARGS+=("--note" "Final snapshot after all-in-one DS4 workflow $SESSION_NAME.")
    REPO_SESSION_ARGS+=("--note" "Model review summary: $SESSION_DIR/$MODEL_REVIEW_NAME/summary.md")
    REPO_SESSION_ARGS+=("--cmd" "find \"$SESSION_DIR\" -maxdepth 3 -type f | sort")

    run_step "04-repo-final" env REPO_SESSION_LOG_ROOT="$SESSION_DIR" "$SCRIPT_DIR/repo-session.sh" "${REPO_SESSION_ARGS[@]}"
}

write_summary() {
    local final_status="$1"

    status_for() {
        local log="$SESSION_DIR/$1.log"
        if [ -f "$log" ]; then
            sed -n 's/^exit_status=//p' "$log" | tail -n 1
        else
            echo "missing"
        fi
    }

    {
        echo "# DS4 Everything Summary"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Updated: $(date -Iseconds)"
        echo "- Status: $final_status"
        echo "- Directory: $SESSION_DIR"
        echo
        echo "## Step Status"
        echo
        echo "- repo baseline: $(status_for 01-repo-baseline)"
        echo "- tests: $(status_for 02-tests)"
        echo "- model review: $(status_for 03-model-review)"
        echo "- repo final: $(status_for 04-repo-final)"
        echo
        echo "## Where To Look"
        echo
        echo "- All console timeline: run.log"
        echo "- Test summary: tests/summary.md"
        echo "- Model review summary: model-review/summary.md"
        echo "- Model responses: model-review/model-responses/"
        echo "- Expert/router trace: model-review/model-runtime/run.log"
        echo "- Runtime summary: model-review/model-runtime/summary.md"
        echo "- Final repo snapshot: repo-final/"
        echo
        echo "## Files"
        echo
        find "$SESSION_DIR" -maxdepth 3 -type f | sort | sed "s#^$SESSION_DIR/#- #"
    } > "$SESSION_DIR/summary.md"
}

write_readme

log_line "Everything session: $SESSION_DIR"
log_line "This will run repo logs, tests, model review cases, and expert tracing."

overall_status=0

run_baseline_repo_session || overall_status=1
run_tests || overall_status=1
run_model_review || overall_status=1
run_final_repo_session || overall_status=1

if [ "$overall_status" -eq 0 ]; then
    write_summary "completed"
else
    write_summary "completed-with-errors"
fi

log_line "Everything session written to: $SESSION_DIR"
log_line "Open summary: $SESSION_DIR/summary.md"
exit "$overall_status"
