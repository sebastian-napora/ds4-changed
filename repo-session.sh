#!/usr/bin/env bash
#
# Create a timestamped repository work session under summary-logs/.
#
# The default session records repository state, git history, content scans,
# shell-script syntax checks, and a validation plan. Optional commands can be
# added with --cmd and will be logged as separate command files.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${REPO_SESSION_LOG_ROOT:-"$SCRIPT_DIR/summary-logs"}"
SESSION_NAME="${REPO_SESSION_NAME:-repo-session-$(date +%Y%m%d-%H%M%S)}"
SESSION_DIR="$LOG_ROOT/$SESSION_NAME"

RUN_TESTS=0
RUN_BUILD=0
RUN_MCP=0
COMMANDS=()
NOTES=()

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Creates summary-logs/<timestamp>/ with logs for repository assessment,
validation, verification, git history, content checks, and optional commands.

Options:
  --with-build          Run make and log the result.
  --with-tests          Run make test and log the result.
  --with-mcp            Try to log MCP CLI state if a local CLI is available.
  --cmd COMMAND         Run COMMAND through bash -lc and log it separately.
                        May be repeated.
  --note TEXT           Add a note to notes.md. May be repeated.
  --session-name NAME   Use a custom session directory name.
  --help                Show this help.

Environment:
  REPO_SESSION_LOG_ROOT  Log root. Default: ./summary-logs
  REPO_SESSION_NAME      Session directory name. Default: repo-session-timestamp

Examples:
  $0
  $0 --with-build --with-tests
  $0 --cmd "git diff --stat" --cmd "make test"
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --with-build)
            RUN_BUILD=1
            ;;
        --with-tests)
            RUN_TESTS=1
            ;;
        --with-mcp)
            RUN_MCP=1
            ;;
        --cmd)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --cmd" >&2
                exit 1
            fi
            COMMANDS+=("$1")
            ;;
        --note)
            shift
            if [ "$#" -eq 0 ]; then
                echo "Missing value after --note" >&2
                exit 1
            fi
            NOTES+=("$1")
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

mkdir -p "$SESSION_DIR/commands"

run_logged() {
    local name="$1"
    shift
    local log="$SESSION_DIR/$name.log"
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

run_shell_logged() {
    local name="$1"
    local cmd="$2"
    local log="$SESSION_DIR/commands/$name.log"
    local status

    {
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

repo_files() {
    if command -v rg >/dev/null 2>&1; then
        rg --files --glob '!venv/**' --glob '!gguf/**' --glob '!summary-logs/**'
    else
        find . \
            -path './.git' -prune -o \
            -path './venv' -prune -o \
            -path './gguf' -prune -o \
            -path './summary-logs' -prune -o \
            -type f -print | sed 's#^\./##'
    fi
}

repo_shell_files() {
    if command -v rg >/dev/null 2>&1; then
        rg --files -g '*.sh' --glob '!venv/**' --glob '!summary-logs/**'
    else
        find . \
            -path './.git' -prune -o \
            -path './venv' -prune -o \
            -path './gguf' -prune -o \
            -path './summary-logs' -prune -o \
            -type f -name '*.sh' -print | sed 's#^\./##'
    fi
}

file_inventory() {
    local log="$SESSION_DIR/file-inventory.log"
    local status

    {
        echo "\$ repo file inventory"
        echo "started_at=$(date -Iseconds)"
        echo
    } > "$log"

    set +e
    repo_files | sort >> "$log" 2>&1
    status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$log"

    return "$status"
}

write_manifest() {
    {
        echo "# Repository Session"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Started: $(date -Iseconds)"
        echo "- Repository: $SCRIPT_DIR"
        echo "- Build enabled: $RUN_BUILD"
        echo "- Tests enabled: $RUN_TESTS"
        echo "- MCP check enabled: $RUN_MCP"
        echo
        echo "## Logs"
        echo
        echo "- assessment.md"
        echo "- validation.md"
        echo "- improvements.md"
        echo "- git-status.log"
        echo "- git-history.log"
        echo "- git-diff.log"
        echo "- content-scan.log"
        echo "- file-inventory.log"
        echo "- shell-syntax.log"
        echo "- tool-versions.log"
        echo "- commands/*.log"
    } > "$SESSION_DIR/README.md"
}

write_notes() {
    {
        echo "# Notes"
        echo
        if [ "${#NOTES[@]}" -eq 0 ]; then
            echo "No session notes were provided."
        else
            local note
            for note in "${NOTES[@]}"; do
                echo "- $note"
            done
        fi
    } > "$SESSION_DIR/notes.md"
}

scan_content() {
    {
        echo "# Content Scan"
        echo
        echo "## TODO/FIXME/HACK/BUG/NOTE"
        echo
        if command -v rg >/dev/null 2>&1; then
            rg -n "TODO|FIXME|HACK|BUG|NOTE" \
                --glob '!venv/**' \
                --glob '!gguf/**' \
                --glob '!summary-logs/**' \
                --glob '!*.gguf' . || true
        else
            repo_files | grep -v '\.gguf$' | while IFS= read -r file; do
                grep -InE "TODO|FIXME|HACK|BUG|NOTE" "$file" /dev/null 2>/dev/null || true
            done
        fi
        echo
        echo "## Large or generated-looking files"
        echo
        find . \
            -path './.git' -prune -o \
            -path './venv' -prune -o \
            -path './gguf' -prune -o \
            -path './summary-logs' -prune -o \
            -type f -size +10M -print
    } > "$SESSION_DIR/content-scan.log"
}

shell_syntax_check() {
    local status=0
    local file
    {
        echo "# Shell Syntax Check"
        echo
    } > "$SESSION_DIR/shell-syntax.log"

    while IFS= read -r file; do
        {
            echo "===== bash -n $file ====="
            bash -n "$file"
            echo "exit_status=$?"
            echo
        } >> "$SESSION_DIR/shell-syntax.log" 2>&1 || status=1
    done < <(repo_shell_files | sort)

    return "$status"
}

tool_versions() {
    {
        echo "# Tool Versions"
        echo
        echo "date: $(date -Iseconds)"
        echo "uname: $(uname -a 2>/dev/null || true)"
        echo
        for tool in git rg awk sed bash make cc clang nvidia-smi curl codex; do
            if command -v "$tool" >/dev/null 2>&1; then
                echo "## $tool"
                command -v "$tool"
                case "$tool" in
                    nvidia-smi)
                        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || true
                        ;;
                    codex)
                        codex --version 2>/dev/null || true
                        ;;
                    *)
                        "$tool" --version 2>&1 | sed -n '1,3p' || true
                        ;;
                esac
                echo
            else
                echo "$tool: unavailable"
            fi
        done
    } > "$SESSION_DIR/tool-versions.log"
}

mcp_check() {
    {
        echo "# MCP Check"
        echo
        if command -v codex >/dev/null 2>&1; then
            echo "\$ codex mcp list"
            codex mcp list 2>&1 || true
        else
            echo "No local codex CLI found. MCP tools available inside the assistant runtime are not directly callable from this shell script."
        fi
    } > "$SESSION_DIR/mcp.log"
}

write_assessment() {
    local dirty="no"
    local shell_status="unknown"
    local build_status="not-run"
    local test_status="not-run"

    if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
        dirty="yes"
    fi
    if grep -Eq "exit_status=[1-9][0-9]*" "$SESSION_DIR/shell-syntax.log" 2>/dev/null; then
        shell_status="failed"
    elif grep -q "exit_status=0" "$SESSION_DIR/shell-syntax.log" 2>/dev/null; then
        shell_status="completed"
    fi
    if [ -s "$SESSION_DIR/build.log" ]; then
        build_status="$(tail -n 1 "$SESSION_DIR/build.log" | sed 's/^exit_status=//')"
    fi
    if [ -s "$SESSION_DIR/test.log" ]; then
        test_status="$(tail -n 1 "$SESSION_DIR/test.log" | sed 's/^exit_status=//')"
    fi

    {
        echo "# Assessment"
        echo
        echo "- Dirty working tree: $dirty"
        echo "- Shell syntax check: $shell_status"
        echo "- Build status: $build_status"
        echo "- Test status: $test_status"
        echo "- Session directory: $SESSION_DIR"
        echo
        echo "Review the logs in this directory before making edits. If --cmd was used, each command has a separate log in commands/."
    } > "$SESSION_DIR/assessment.md"
}

write_validation_plan() {
    {
        echo "# Validation"
        echo
        echo "- Shell syntax: logged in shell-syntax.log."
        echo "- Build: run this script with --with-build, or inspect build.log if already present."
        echo "- Tests: run this script with --with-tests, or inspect test.log if already present."
        echo "- Runtime model validation: use start-low-gpu-summary.sh and inspect its summary.md plus run.log."
        echo "- Git verification: inspect git-status.log, git-history.log, and git-diff.log."
    } > "$SESSION_DIR/validation.md"
}

write_improvements_template() {
    {
        echo "# Improvements"
        echo
        echo "- Candidate improvements should be copied here with links to supporting logs."
        echo "- Keep proposed edits small, testable, and tied to a validation command."
        echo "- For inference changes, prefer short smoke tests before long model runs."
    } > "$SESSION_DIR/improvements.md"
}

write_manifest
write_notes

run_logged git-status git status --short --branch || true
run_logged git-history git log --oneline --decorate --graph -n 40 || true
run_logged git-diff git diff --stat || true
run_logged git-diff-names git diff --name-status || true
file_inventory || true
tool_versions
scan_content
shell_syntax_check || true

if [ "$RUN_MCP" -eq 1 ]; then
    mcp_check
fi

if [ "$RUN_BUILD" -eq 1 ]; then
    run_logged build make || true
fi

if [ "$RUN_TESTS" -eq 1 ]; then
    run_logged test make test || true
fi

if [ "${#COMMANDS[@]}" -gt 0 ]; then
    index=1
    for cmd in "${COMMANDS[@]}"; do
        run_shell_logged "command-$(printf '%02d' "$index")" "$cmd" || true
        index=$((index + 1))
    done
fi

write_assessment
write_validation_plan
write_improvements_template

echo "Repository session written to: $SESSION_DIR"
