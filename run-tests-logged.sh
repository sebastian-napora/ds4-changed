#!/usr/bin/env bash
#
# Build and run ds4 tests one by one, saving each test to a separate log.
#
# Default behavior:
#   1. make ds4_test
#   2. ./ds4_test --list
#   3. ./ds4_test <each-listed-test-flag>
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${DS4_TEST_LOG_ROOT:-"$SCRIPT_DIR/summary-logs"}"
SESSION_NAME="${DS4_TEST_SESSION_NAME:-tests-$(date +%Y%m%d-%H%M%S)}"
SESSION_DIR="$LOG_ROOT/$SESSION_NAME"
BUILD=1
LIST_ONLY=0
TESTS=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs ds4 tests one by one and writes separate logs for every step.

Options:
  --log-dir DIR       Write logs directly to DIR instead of a timestamped session.
  --session-name NAME Use summary-logs/NAME as the session directory.
  --test FLAG         Run one test flag, for example --test --server.
                     May be repeated. If omitted, tests come from ./ds4_test --list.
  --no-build          Do not run make ds4_test before tests.
  --list-only         Discover tests and write the plan, but do not execute them.
  --help              Show this help.

Environment:
  DS4_TEST_LOG_ROOT      Log root. Default: ./summary-logs
  DS4_TEST_SESSION_NAME  Session directory name. Default: tests-timestamp
  DS4_TEST_MODEL         Passed through to ds4_test when set.

Examples:
  $0
  $0 --test --server --test --logprob-vectors
  $0 --log-dir summary-logs/manual-tests
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --log-dir)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --log-dir" >&2
                exit 1
            fi
            SESSION_DIR="$1"
            ;;
        --session-name)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --session-name" >&2
                exit 1
            fi
            SESSION_NAME="$1"
            SESSION_DIR="$LOG_ROOT/$SESSION_NAME"
            ;;
        --test)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --test" >&2
                exit 1
            fi
            TESTS+=("$1")
            ;;
        --no-build)
            BUILD=0
            ;;
        --list-only)
            LIST_ONLY=1
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

log_command() {
    local log="$1"
    shift
    local status

    {
        echo "\$ $*"
        echo "started_at=$(date -Iseconds)"
        echo
    } > "$log"

    set +e
    "$@" >> "$log" 2>&1
    status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$log"

    return "$status"
}

safe_name() {
    printf "%s" "$1" | sed 's/^--//; s/[^A-Za-z0-9_.-]/_/g'
}

write_readme() {
    {
        echo "# Logged Test Session"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Started: $(date -Iseconds)"
        echo "- Directory: $SESSION_DIR"
        echo "- Build enabled: $BUILD"
        echo "- List only: $LIST_ONLY"
        echo
        echo "## Files"
        echo
        echo "- summary.md"
        echo "- test-plan.txt"
        echo "- 00-build-ds4-test.log"
        echo "- 01-list-tests.log"
        echo "- NN-<test-name>.log"
    } > "$SESSION_DIR/README.md"
}

discover_tests() {
    local list_log="$SESSION_DIR/01-list-tests.log"
    local status

    if [ "${#TESTS[@]}" -gt 0 ]; then
        printf "%s\n" "${TESTS[@]}" > "$SESSION_DIR/test-plan.txt"
        {
            echo "Tests provided by --test:"
            cat "$SESSION_DIR/test-plan.txt"
            echo "exit_status=0"
        } > "$list_log"
        return 0
    fi

    set +e
    log_command "$list_log" ./ds4_test --list
    status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        sed -n '/^--/p' "$list_log" > "$SESSION_DIR/test-plan.txt"
    else
        : > "$SESSION_DIR/test-plan.txt"
    fi

    return "$status"
}

write_summary() {
    local final_status="$1"
    {
        echo "# Test Summary"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Updated: $(date -Iseconds)"
        echo "- Status: $final_status"
        echo "- Directory: $SESSION_DIR"
        echo
        echo "## Plan"
        echo
        if [ -s "$SESSION_DIR/test-plan.txt" ]; then
            sed 's/^/- /' "$SESSION_DIR/test-plan.txt"
        else
            echo "- No tests discovered."
        fi
        echo
        echo "## Results"
        echo
        for log in "$SESSION_DIR"/*.log; do
            [ -e "$log" ] || continue
            local name
            local status
            name="$(basename "$log")"
            status="$(sed -n 's/^exit_status=//p' "$log" | tail -n 1)"
            [ -n "$status" ] || status="unknown"
            echo "- $name: $status"
        done
    } > "$SESSION_DIR/summary.md"
}

write_readme

overall_status=0

if [ "$BUILD" -ne 0 ]; then
    log_command "$SESSION_DIR/00-build-ds4-test.log" make ds4_test || overall_status=1
fi

if [ ! -x ./ds4_test ] && { [ "$LIST_ONLY" -eq 0 ] || [ "${#TESTS[@]}" -eq 0 ]; }; then
    {
        echo "Missing executable ./ds4_test."
        echo "Run with build enabled, or build it manually with: make ds4_test"
        echo "exit_status=1"
    } > "$SESSION_DIR/01-list-tests.log"
    : > "$SESSION_DIR/test-plan.txt"
    write_summary "failed"
    echo "Logged test session written to: $SESSION_DIR"
    exit 1
fi

discover_tests || overall_status=1

if [ "$LIST_ONLY" -eq 0 ]; then
    index=2
    while IFS= read -r test_flag; do
        [ -n "$test_flag" ] || continue
        log_name="$(printf "%02d-%s.log" "$index" "$(safe_name "$test_flag")")"
        log_command "$SESSION_DIR/$log_name" ./ds4_test "$test_flag" || overall_status=1
        index=$((index + 1))
    done < "$SESSION_DIR/test-plan.txt"
fi

if [ "$overall_status" -eq 0 ]; then
    write_summary "passed"
else
    write_summary "failed"
fi

echo "Logged test session written to: $SESSION_DIR"
exit "$overall_status"
