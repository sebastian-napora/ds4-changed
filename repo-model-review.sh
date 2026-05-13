#!/usr/bin/env bash
#
# Run local DS4 model review cases one by one and save real model output.
#
# This is different from repo-session.sh and run-tests-logged.sh:
# those scripts gather repository/test evidence. This script sends that
# evidence to the running local model so inference actually happens.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_ROOT="${DS4_MODEL_CASE_LOG_ROOT:-"$SCRIPT_DIR/summary-logs"}"
SESSION_NAME="${DS4_MODEL_CASE_SESSION_NAME:-model-cases-$(date +%Y%m%d-%H%M%S)}"
SESSION_DIR="$LOG_ROOT/$SESSION_NAME"

HOST="${DS4_HOST:-127.0.0.1}"
PORT="${DS4_PORT:-11112}"
MODEL_NAME="${DS4_MODEL_CASE_MODEL:-deepseek-chat}"
API_KEY="${DS4_MODEL_CASE_API_KEY:-dsv4-local}"
MAX_TOKENS="${DS4_MODEL_CASE_MAX_TOKENS:-1024}"
TEMPERATURE="${DS4_MODEL_CASE_TEMPERATURE:-0}"
THINK="${DS4_MODEL_CASE_THINK:-true}"
REQUEST_TIMEOUT="${DS4_MODEL_CASE_TIMEOUT:-900}"
READY_TIMEOUT="${DS4_MODEL_CASE_READY_TIMEOUT:-300}"

START_MODEL=0
KEEP_MODEL_RUNNING=0
RUN_TESTS=0
NO_DEFAULT_CASES=0
EXPERT_TRACE="summary"
TEST_LOG_DIR=""
CUSTOM_CASE_FILES=()
CUSTOM_PROMPTS=()

MODEL_WRAPPER_PID=""
CASES_DIR="$SESSION_DIR/cases"
CONTEXT_DIR="$SESSION_DIR/repo-context"
RESPONSES_DIR="$SESSION_DIR/model-responses"
TESTS_DIR="$SESSION_DIR/tests"

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Runs repository/model review cases against a local DS4 server and saves every
prompt plus every real model response under summary-logs/.

Core options:
  --start-model             Start ./start-low-gpu-summary.sh, wait for DS4,
                            run model cases, then stop it.
  --keep-model-running      With --start-model, leave the model running.
  --with-tests              Run ./run-tests-logged.sh first and ask the model
                            to inspect each test log one by one.
  --test-log-dir DIR        Use an existing run-tests-logged.sh directory.
  --case-file FILE          Add a custom prompt/case file. May be repeated.
  --prompt TEXT             Add a custom inline prompt. May be repeated.
  --no-default-cases        Only run custom/test-log cases.
  --session-name NAME       Use a custom session directory name.

Model behavior:
  --expert-summary          Enable DS4_ROUTER_TRACE=summary when starting model.
                            Default.
  --expert-detail           Enable DS4_ROUTER_TRACE=1 when starting model.
  --no-expert-trace         Start model without router tracing.
  --think                   Send "think": true. Default.
  --no-think                Send "think": false.
  --max-tokens N            Max response tokens. Default: 1024.
  --temperature N           Sampling temperature. Default: 0.

Environment:
  DS4_HOST                  DS4 host. Default: 127.0.0.1
  DS4_PORT                  DS4 port. Default: 11112
  DS4_MODEL_CASE_MODEL      API model name. Default: deepseek-chat
  DS4_MODEL_CASE_TIMEOUT    Per-request timeout seconds. Default: 900
  DS4_MODEL_CASE_THINK      true/false. Default: true

Examples:
  $0 --start-model --with-tests --expert-summary
  $0 --with-tests
  $0 --test-log-dir summary-logs/tests-20260513-131419
  $0 --start-model --case-file prompts/repo-audit.txt --max-tokens 2048
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --start-model)
            START_MODEL=1
            ;;
        --keep-model-running)
            KEEP_MODEL_RUNNING=1
            ;;
        --with-tests)
            RUN_TESTS=1
            ;;
        --test-log-dir)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --test-log-dir" >&2; exit 1; }
            TEST_LOG_DIR="$1"
            ;;
        --case-file)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --case-file" >&2; exit 1; }
            CUSTOM_CASE_FILES+=("$1")
            ;;
        --prompt)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --prompt" >&2; exit 1; }
            CUSTOM_PROMPTS+=("$1")
            ;;
        --no-default-cases)
            NO_DEFAULT_CASES=1
            ;;
        --session-name)
            shift
            [ "$#" -gt 0 ] || { echo "Missing value after --session-name" >&2; exit 1; }
            SESSION_NAME="$1"
            SESSION_DIR="$LOG_ROOT/$SESSION_NAME"
            CASES_DIR="$SESSION_DIR/cases"
            CONTEXT_DIR="$SESSION_DIR/repo-context"
            RESPONSES_DIR="$SESSION_DIR/model-responses"
            TESTS_DIR="$SESSION_DIR/tests"
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

if [ "$HOST" = "0.0.0.0" ]; then
    HOST="127.0.0.1"
fi

mkdir -p "$CASES_DIR" "$CONTEXT_DIR" "$RESPONSES_DIR"

safe_name() {
    printf "%s" "$1" | sed 's/[^A-Za-z0-9_.-]/-/g; s/--*/-/g; s/^-//; s/-$//'
}

json_escape() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf "%s" "$s"
}

run_log() {
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

run_shell_log() {
    local log="$1"
    local cmd="$2"
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

append_excerpt() {
    local title="$1"
    local file="$2"
    local head_lines="${3:-160}"
    local tail_lines="${4:-80}"

    echo
    echo "## $title"
    echo
    echo '```'
    if [ ! -s "$file" ]; then
        echo "missing or empty: $file"
    else
        local line_count
        line_count="$(wc -l < "$file" | tr -d ' ')"
        if [ "$line_count" -le $((head_lines + tail_lines + 10)) ]; then
            sed -n '1,$p' "$file"
        else
            sed -n "1,${head_lines}p" "$file"
            echo
            echo "... omitted $((line_count - head_lines - tail_lines)) lines ..."
            echo
            tail -n "$tail_lines" "$file"
        fi
    fi
    echo '```'
}

write_readme() {
    {
        echo "# DS4 Model Case Session"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Started: $(date -Iseconds)"
        echo "- Repository: $SCRIPT_DIR"
        echo "- Endpoint: http://$HOST:$PORT/v1/chat/completions"
        echo "- Model: $MODEL_NAME"
        echo "- Start model: $START_MODEL"
        echo "- Expert trace: ${EXPERT_TRACE:-disabled}"
        echo "- Think: $THINK"
        echo "- Max tokens: $MAX_TOKENS"
        echo "- Run tests first: $RUN_TESTS"
        echo
        echo "## Important"
        echo
        echo "This session forces real model inference. Every file under model-responses/"
        echo "comes from an actual HTTP request to the local DS4 server."
    } > "$SESSION_DIR/README.md"
}

gather_repo_context() {
    echo "Collecting repository context in: $CONTEXT_DIR"

    run_shell_log "$CONTEXT_DIR/git-status.log" "git status --short" || true
    run_shell_log "$CONTEXT_DIR/git-history.log" "git log --oneline --decorate -n 30" || true
    run_shell_log "$CONTEXT_DIR/git-diff-stat.log" "git diff --stat" || true
    run_shell_log "$CONTEXT_DIR/git-diff-names.log" "git diff --name-status" || true

    if command -v rg >/dev/null 2>&1; then
        run_shell_log "$CONTEXT_DIR/file-inventory.log" "rg --files | sort" || true
        run_shell_log "$CONTEXT_DIR/content-scan.log" "rg -n \"TODO|FIXME|HACK|XXX|router|expert|test|summary|log\" README.md *.sh tests ds4.c ds4_server.c 2>/dev/null" || true
    else
        run_shell_log "$CONTEXT_DIR/file-inventory.log" "find . -path './.git' -prune -o -path './summary-logs' -prune -o -type f -print | sort" || true
        run_shell_log "$CONTEXT_DIR/content-scan.log" "grep -RInE \"TODO|FIXME|HACK|XXX|router|expert|test|summary|log\" README.md *.sh tests ds4.c ds4_server.c 2>/dev/null" || true
    fi

    run_shell_log "$CONTEXT_DIR/script-syntax.log" "for f in *.sh; do echo \"== \$f ==\"; bash -n \"\$f\"; done" || true
}

run_tests_first() {
    echo "Running tests before model startup. Logs: $TESTS_DIR"
    if [ ! -x "$SCRIPT_DIR/run-tests-logged.sh" ]; then
        echo "run-tests-logged.sh is missing or not executable" > "$SESSION_DIR/test-run.log"
        return 1
    fi

    set +e
    "$SCRIPT_DIR/run-tests-logged.sh" --log-dir "$TESTS_DIR" > "$SESSION_DIR/test-run.log" 2>&1
    local status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "exit_status=$status"
    } >> "$SESSION_DIR/test-run.log"

    TEST_LOG_DIR="$TESTS_DIR"
    return 0
}

add_case() {
    local name="$1"
    local body_file="$2"
    local safe
    safe="$(safe_name "$name")"
    cp "$body_file" "$CASES_DIR/$safe.prompt.md"
    printf "%s\n" "$safe" >> "$CASES_DIR/case-order.txt"
}

create_default_cases() {
    local f

    f="$CASES_DIR/repository-assessment.tmp"
    {
        echo "You are reviewing this DS4 repository automation."
        echo
        echo "Task: assess whether the repository logging and model-review workflow is complete and trustworthy."
        echo
        echo "Return:"
        echo "1. verdict"
        echo "2. evidence from the logs/context"
        echo "3. missing validation"
        echo "4. concrete next improvements"
        append_excerpt "git status" "$CONTEXT_DIR/git-status.log"
        append_excerpt "git diff names" "$CONTEXT_DIR/git-diff-names.log"
        append_excerpt "file inventory" "$CONTEXT_DIR/file-inventory.log" 120 40
        append_excerpt "content scan" "$CONTEXT_DIR/content-scan.log" 160 80
    } > "$f"
    add_case "repository-assessment" "$f"

    f="$CASES_DIR/workflow-validation.tmp"
    {
        echo "You are validating the scripts that manage repo sessions, tests, and model work."
        echo
        echo "Task: check the workflow for setup, creation, editing, reading, content checking, git history, tools, MCP hooks, terminal commands, validation, and verification."
        echo
        echo "For each area, say: covered / partially covered / missing, with evidence."
        append_excerpt "script syntax" "$CONTEXT_DIR/script-syntax.log"
        append_excerpt "README relevant scan" "$CONTEXT_DIR/content-scan.log" 220 60
        append_excerpt "git history" "$CONTEXT_DIR/git-history.log"
    } > "$f"
    add_case "workflow-validation" "$f"

    f="$CASES_DIR/improvement-plan.tmp"
    {
        echo "You are proposing improvements for this local DS4 repo automation."
        echo
        echo "Task: produce a prioritized implementation plan. Focus on changes that improve correctness, observability, and repeatable validation."
        echo
        echo "Prefer specific scripts, flags, files, and expected logs."
        append_excerpt "git diff stat" "$CONTEXT_DIR/git-diff-stat.log"
        append_excerpt "content scan" "$CONTEXT_DIR/content-scan.log" 180 80
    } > "$f"
    add_case "improvement-plan" "$f"
}

create_test_cases() {
    local dir="$1"
    local log
    [ -d "$dir" ] || return 0

    for log in "$dir"/*.log; do
        [ -e "$log" ] || continue
        local base
        local f
        base="$(basename "$log" .log)"
        f="$CASES_DIR/test-$base.tmp"
        {
            echo "You are reviewing one DS4 test log."
            echo
            echo "Task: inspect this test case result and return:"
            echo "1. pass/fail/skip verdict"
            echo "2. exact evidence from the log"
            echo "3. likely cause if failed or skipped"
            echo "4. what command should be run next"
            echo "5. whether this indicates a model issue, environment issue, or test-runner issue"
            append_excerpt "test log: $base" "$log" 220 120
        } > "$f"
        add_case "test-$base" "$f"
    done
}

create_custom_cases() {
    local idx=1
    local prompt
    local file

    for prompt in "${CUSTOM_PROMPTS[@]}"; do
        file="$CASES_DIR/custom-inline-$idx.tmp"
        {
            echo "$prompt"
            echo
            echo "Repository context follows."
            append_excerpt "git status" "$CONTEXT_DIR/git-status.log"
            append_excerpt "content scan" "$CONTEXT_DIR/content-scan.log" 120 40
        } > "$file"
        add_case "custom-inline-$idx" "$file"
        idx=$((idx + 1))
    done

    idx=1
    for file in "${CUSTOM_CASE_FILES[@]}"; do
        if [ -f "$file" ]; then
            add_case "custom-file-$idx-$(basename "$file")" "$file"
        else
            echo "Missing custom case file: $file" >> "$SESSION_DIR/warnings.log"
        fi
        idx=$((idx + 1))
    done
}

server_ready() {
    command -v curl >/dev/null 2>&1 || return 1
    curl -fsS -m 2 "http://$HOST:$PORT/v1/models" >/dev/null 2>&1
}

wait_for_server() {
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    echo "Waiting for model server at http://$HOST:$PORT ..."

    while [ "$(date +%s)" -lt "$deadline" ]; do
        if server_ready; then
            echo "Model server is ready."
            return 0
        fi
        sleep 2
    done

    echo "Model server did not become ready within ${READY_TIMEOUT}s." >&2
    return 1
}

start_model_bg() {
    local log="$SESSION_DIR/model-runtime-wrapper.log"

    if [ ! -x "$SCRIPT_DIR/start-low-gpu-summary.sh" ]; then
        echo "start-low-gpu-summary.sh is missing or not executable" >&2
        return 1
    fi

    echo "Starting model runtime. Log: $log"
    DS4_SUMMARY_LOG_ROOT="$SESSION_DIR" \
    DS4_SUMMARY_SESSION_NAME="model-runtime" \
    DS4_SUMMARY_FINAL_REPO_SESSION=0 \
    DS4_SUMMARY_SMOKE_PROMPT="" \
    DS4_ROUTER_TRACE="$EXPERT_TRACE" \
    "$SCRIPT_DIR/start-low-gpu-summary.sh" > "$log" 2>&1 &
    MODEL_WRAPPER_PID=$!
    echo "$MODEL_WRAPPER_PID" > "$SESSION_DIR/model-runtime-wrapper.pid"
}

stop_model_bg() {
    if [ -n "${MODEL_WRAPPER_PID:-}" ] && [ "$KEEP_MODEL_RUNNING" -eq 0 ]; then
        echo "Stopping model runtime pid=$MODEL_WRAPPER_PID"
        kill "$MODEL_WRAPPER_PID" 2>/dev/null || true
        wait "$MODEL_WRAPPER_PID" 2>/dev/null || true
    fi
}

extract_response_text() {
    local response_json="$1"
    local response_txt="$2"

    if command -v jq >/dev/null 2>&1; then
        jq -r '
          .choices[0].message.content //
          .choices[0].message.reasoning_content //
          .content[0].text //
          .
        ' "$response_json" > "$response_txt" 2>/dev/null || cp "$response_json" "$response_txt"
    else
        cp "$response_json" "$response_txt"
    fi
}

run_model_case() {
    local case_name="$1"
    local prompt_file="$CASES_DIR/$case_name.prompt.md"
    local out_dir="$RESPONSES_DIR/$case_name"
    local request_json="$out_dir/request.json"
    local response_json="$out_dir/response.json"
    local response_txt="$out_dir/response.txt"
    local curl_log="$out_dir/curl.log"
    local prompt
    local prompt_json
    local system_json
    local http_code
    local status

    mkdir -p "$out_dir"
    cp "$prompt_file" "$out_dir/prompt.md"

    prompt="$(cat "$prompt_file")"
    prompt_json="$(json_escape "$prompt")"
    system_json="$(json_escape "You are a local DS4 repository validation agent. Review evidence carefully. Give concise reasoning, concrete findings, and actionable next commands. Do not invent files or test results not present in the prompt.")"

    {
        echo "{"
        echo "  \"model\": \"$(json_escape "$MODEL_NAME")\","
        echo "  \"messages\": ["
        echo "    {\"role\": \"system\", \"content\": \"$system_json\"},"
        echo "    {\"role\": \"user\", \"content\": \"$prompt_json\"}"
        echo "  ],"
        echo "  \"think\": $THINK,"
        echo "  \"temperature\": $TEMPERATURE,"
        echo "  \"max_tokens\": $MAX_TOKENS"
        echo "}"
    } > "$request_json"

    echo "Running model case: $case_name"
    {
        echo "case=$case_name"
        echo "started_at=$(date -Iseconds)"
        echo "endpoint=http://$HOST:$PORT/v1/chat/completions"
        echo "request=$request_json"
        echo
    } > "$curl_log"

    set +e
    http_code="$(curl -sS -m "$REQUEST_TIMEOUT" \
        -o "$response_json" \
        -w "%{http_code}" \
        "http://$HOST:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer $API_KEY" \
        --data @"$request_json" 2>> "$curl_log")"
    status=$?
    set -e

    {
        echo
        echo "finished_at=$(date -Iseconds)"
        echo "curl_exit_status=$status"
        echo "http_status=$http_code"
    } >> "$curl_log"

    if [ -s "$response_json" ]; then
        extract_response_text "$response_json" "$response_txt"
    else
        echo "No response body saved." > "$response_txt"
    fi

    [ "$status" -eq 0 ] && [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]
}

write_summary() {
    local final_status="$1"
    local total=0
    local passed=0
    local failed=0
    local case_name

    {
        echo "# DS4 Model Case Summary"
        echo
        echo "- Session: $SESSION_NAME"
        echo "- Updated: $(date -Iseconds)"
        echo "- Status: $final_status"
        echo "- Directory: $SESSION_DIR"
        echo "- Endpoint: http://$HOST:$PORT/v1/chat/completions"
        echo "- Think: $THINK"
        echo "- Max tokens: $MAX_TOKENS"
        echo
        echo "## Cases"
        if [ -s "$CASES_DIR/case-order.txt" ]; then
            while IFS= read -r case_name; do
                [ -n "$case_name" ] || continue
                total=$((total + 1))
                local curl_log="$RESPONSES_DIR/$case_name/curl.log"
                local http_status="missing"
                local curl_status="missing"
                if [ -f "$curl_log" ]; then
                    http_status="$(sed -n 's/^http_status=//p' "$curl_log" | tail -n 1)"
                    curl_status="$(sed -n 's/^curl_exit_status=//p' "$curl_log" | tail -n 1)"
                fi
                if [ "$curl_status" = "0" ] && [ "$http_status" -ge 200 ] 2>/dev/null && [ "$http_status" -lt 300 ] 2>/dev/null; then
                    passed=$((passed + 1))
                else
                    failed=$((failed + 1))
                fi
                echo "- $case_name: curl=$curl_status http=$http_status response=model-responses/$case_name/response.txt"
            done < "$CASES_DIR/case-order.txt"
        else
            echo "- No cases were created."
        fi
        echo
        echo "## Counts"
        echo
        echo "- Total cases: $total"
        echo "- Completed requests: $passed"
        echo "- Failed requests: $failed"
        echo
        echo "## Main Files"
        echo
        find "$SESSION_DIR" -maxdepth 3 -type f | sort | sed "s#^$SESSION_DIR/#- #"
    } > "$SESSION_DIR/summary.md"
}

cleanup() {
    stop_model_bg || true
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

write_readme
gather_repo_context

if [ "$RUN_TESTS" -ne 0 ]; then
    run_tests_first || true
fi

if [ "$NO_DEFAULT_CASES" -eq 0 ]; then
    create_default_cases
fi

if [ -n "$TEST_LOG_DIR" ]; then
    create_test_cases "$TEST_LOG_DIR"
fi

create_custom_cases

if [ ! -s "$CASES_DIR/case-order.txt" ]; then
    echo "No model cases were created. Use --prompt, --case-file, --with-tests, or omit --no-default-cases." >&2
    write_summary "failed-no-cases"
    exit 1
fi

if [ "$START_MODEL" -ne 0 ]; then
    start_model_bg
fi

if ! wait_for_server; then
    write_summary "failed-server-not-ready"
    cleanup
    exit 1
fi

overall_status=0
while IFS= read -r case_name; do
    [ -n "$case_name" ] || continue
    run_model_case "$case_name" || overall_status=1
done < "$CASES_DIR/case-order.txt"

if [ "$overall_status" -eq 0 ]; then
    write_summary "completed"
else
    write_summary "completed-with-errors"
fi

cleanup
echo "Model case session written to: $SESSION_DIR"
exit "$overall_status"
