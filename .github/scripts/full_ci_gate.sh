#!/bin/bash

# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Decides whether the heavy TPU/GPU test suites may run in this CI run, and
# whether the "Full CI Gatekeeper" job has to block the merge (b/571566462).
#
# On pull requests the first matching rule decides:
#   1. Drafts never run the heavy suites. The message says whether marking the
#      PR ready for review is enough to run them.
#   2. The `scheduled-only` label runs them: it asks for the full scheduled
#      pipeline, so it wins over the rules below.
#   3. The `skip-ci-tests` label skips them. Every new PR gets this label; a bot
#      removes it after 2 approvals once the other checks pass, and the author
#      can remove it sooner.
#   4. The first run of a new PR (action `opened`) skips them, because
#      add_skip_ci_tests_label.yml adds the label in parallel and this run's
#      payload doesn't have it yet.
#   5. Otherwise they run.
# Other events (e.g. schedule, workflow_dispatch, or a release run that calls
# ci_pipeline.yml, which reports the caller's event) always run them.
#
# Inputs (environment variables). Boolean inputs count as false unless they are
# "true" in any letter case; EVENT_NAME and EVENT_ACTION are compared exactly:
#   EVENT_NAME                github.event_name (defaults to GITHUB_EVENT_NAME,
#                             then pull_request, like analyze_code_changes.sh)
#   EVENT_ACTION              github.event.action, e.g. opened or synchronize
#   IS_DRAFT                  github.event.pull_request.draft
#   HAS_SKIP_CI_TESTS_LABEL   whether the PR has the `skip-ci-tests` label
#   HAS_SCHEDULED_ONLY_LABEL  whether the PR has the `scheduled-only` label
#   RUN_TESTS, RUN_NOTEBOOKS, RUN_PRETRAIN_TESTS, RUN_POSTTRAIN_TESTS,
#   RUN_PATHWAYS_TESTS, RUN_GPU_TESTS
#                             the flags emitted by analyze_code_changes.sh
#
# Outputs (written to GITHUB_OUTPUT when it is set, and echoed to stdout):
#   run_heavy_tests           "true" if the heavy TPU/GPU suites may run
#   heavy_tests_gated         "true" if the change needs heavy suites that the
#                             gate skipped
#   heavy_tests_gate_message  why they were skipped and what the PR still needs
#                             to run them; empty when nothing was gated
#
# The script only decides and always exits 0 (unless GITHUB_OUTPUT is set but
# cannot be written). The Full CI Gatekeeper job in ci_pipeline.yml fails the
# run when heavy_tests_gated is "true", and also when heavy_tests_gated or
# run_heavy_tests is missing or not "true"/"false", so every run must emit all
# three outputs with these values.

set -euo pipefail

SKIP_LABEL_MESSAGE="Heavy tests skipped because this PR has the 'skip-ci-tests' label. A bot removes it after 2 approvals once the other checks pass, which runs all tests; remove it yourself to run them sooner."
NEW_PR_MESSAGE="Heavy tests skipped on the first run of a new PR, which gets the 'skip-ci-tests' label. A bot removes it after 2 approvals once the other checks pass, which runs all tests; remove it yourself to run them sooner."
DRAFT_MESSAGE="Heavy tests skipped because this PR is a draft. Mark it 'Ready for review'; a bot then removes the 'skip-ci-tests' label after 2 approvals once the other checks pass, which runs all tests."
DRAFT_READY_MESSAGE="Heavy tests skipped because this PR is a draft. Mark it 'Ready for review' to run all tests before merging."

# Prints "true" if the argument is "true" in any letter case, otherwise "false".
normalize_bool() {
  local val="${1:-}"
  if [[ "${val,,}" == "true" ]]; then
    echo "true"
  else
    echo "false"
  fi
}

# Writes key=value to GITHUB_OUTPUT (if set) and to stdout.
emit_output() {
  local key="$1"
  local value="$2"
  echo "${key}=${value}"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "${key}=${value}" >> "$GITHUB_OUTPUT"
  fi
}

EVENT_NAME="${EVENT_NAME:-${GITHUB_EVENT_NAME:-pull_request}}"
EVENT_ACTION="${EVENT_ACTION:-}"
IS_DRAFT="$(normalize_bool "${IS_DRAFT:-}")"
HAS_SKIP_CI_TESTS_LABEL="$(normalize_bool "${HAS_SKIP_CI_TESTS_LABEL:-}")"
HAS_SCHEDULED_ONLY_LABEL="$(normalize_bool "${HAS_SCHEDULED_ONLY_LABEL:-}")"
RUN_TESTS="$(normalize_bool "${RUN_TESTS:-}")"
RUN_NOTEBOOKS="$(normalize_bool "${RUN_NOTEBOOKS:-}")"
RUN_PRETRAIN_TESTS="$(normalize_bool "${RUN_PRETRAIN_TESTS:-}")"
RUN_POSTTRAIN_TESTS="$(normalize_bool "${RUN_POSTTRAIN_TESTS:-}")"
RUN_PATHWAYS_TESTS="$(normalize_bool "${RUN_PATHWAYS_TESTS:-}")"
RUN_GPU_TESTS="$(normalize_bool "${RUN_GPU_TESTS:-}")"

# Heavy suites are needed when notebooks run (on TPU) or when tests run with any
# suite that has a TPU/GPU flavor. This mirrors the flag conditions of the heavy
# jobs in ci_pipeline.yml (notebooks are also skipped for forks and when
# github.actor is dependabot[bot], but a notebook-only change still counts as
# heavy here).
heavy_suite_selected="false"
for flag in "$RUN_PRETRAIN_TESTS" "$RUN_POSTTRAIN_TESTS" "$RUN_PATHWAYS_TESTS" "$RUN_GPU_TESTS"; do
  if [[ "$flag" == "true" ]]; then
    heavy_suite_selected="true"
    break
  fi
done
heavy_tests_needed="false"
if [[ "$RUN_NOTEBOOKS" == "true" ]] || [[ "$RUN_TESTS" == "true" && "$heavy_suite_selected" == "true" ]]; then
  heavy_tests_needed="true"
fi

run_heavy_tests="true"
skip_reason=""
if [[ "$EVENT_NAME" == "pull_request" ]]; then
  if [[ "$IS_DRAFT" == "true" ]]; then
    # Rule 1. Pick the message by what marking the PR ready would do: rule 2
    # or 5 runs all tests; under rule 3 or 4 the PR has, or is getting, the
    # `skip-ci-tests` label, which the bot removes after 2 approvals.
    run_heavy_tests="false"
    if [[ "$HAS_SCHEDULED_ONLY_LABEL" == "true" ]] ||
      [[ "$HAS_SKIP_CI_TESTS_LABEL" != "true" && "$EVENT_ACTION" != "opened" ]]; then
      skip_reason="$DRAFT_READY_MESSAGE"
    else
      skip_reason="$DRAFT_MESSAGE"
    fi
  elif [[ "$HAS_SCHEDULED_ONLY_LABEL" == "true" ]]; then
    run_heavy_tests="true"
  elif [[ "$HAS_SKIP_CI_TESTS_LABEL" == "true" ]]; then
    run_heavy_tests="false"
    skip_reason="$SKIP_LABEL_MESSAGE"
  elif [[ "$EVENT_ACTION" == "opened" ]]; then
    run_heavy_tests="false"
    skip_reason="$NEW_PR_MESSAGE"
  fi
fi

heavy_tests_gated="false"
heavy_tests_gate_message=""
if [[ "$run_heavy_tests" == "false" && "$heavy_tests_needed" == "true" ]]; then
  heavy_tests_gated="true"
  heavy_tests_gate_message="$skip_reason"
fi

emit_output "run_heavy_tests" "$run_heavy_tests"
emit_output "heavy_tests_gated" "$heavy_tests_gated"
emit_output "heavy_tests_gate_message" "$heavy_tests_gate_message"
