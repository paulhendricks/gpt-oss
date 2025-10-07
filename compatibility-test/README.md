# API Compatibility Test

This script uses the Agents SDK in TypeScript and the underlying OpenAI client to verify the shape of the API calls but also whether the API performs tool calling.

## What it tests

1. Verifies the provider emits tool calls that match the schema declared in `tools.ts` (AJV validation + deep argument comparison when `strict` cases are enabled).
2. Confirms that non-streaming ("unary") responses include the expected reasoning payload for each API type (Responses and Chat Completions).
3. Checks streaming runs for the presence of reasoning delta/done events when `--streaming` is set, ensuring parity with OpenAI's event model.
4. Normalizes tool-call payloads and reasoning outputs so providers that omit `rawItem` wrappers, return stringified arguments, or mix `reasoning_text` and `input_text` segments still get evaluated accurately.


## Unary path overview

When you run `npm start` without `--streaming`, each case exercises the unary surface area. The harness still uses the OpenAI Agents SDK, so the control flow mirrors how the production Agents runtime handles tool calls.

### Request lifecycle
- `runCase` builds an `Agent` with the case's `tool_name` and issues the same `caseData.input` to every API type configured for the provider (currently `responses` and `chat`).
- The `Runner` sends the first request. If the model returns `status: "incomplete"` with a `tool_call_item`, a two-stage exchange begins: the harness surfaces the tool call to the client, waits for you to execute it, and then automatically submits the follow-up turn containing the tool result.
- The loop ends once the model produces a final assistant message or the `maxTurns` limit is hit. All intermediate payloads are recorded on `result.rawResponses` and `result.newItems` for assertions and debugging.

### What counts as a failure
- **Missing/invalid tool call:** `testToolCall` fails the case if the named tool is never invoked, if the arguments fail AJV validation against the schema, or—when `strict` mode is enabled—if the arguments do not deep-equal `expected_arguments`.
- **Malformed reasoning payload:** `testOutputData` expects unary Chat runs to include at least one assistant message with a non-empty `reasoning` string, and unary Responses runs to emit a `reasoning` output item with non-empty `reasoning_text` entries. Any deviation flips `validResponse` to `false`.
- **Turn limit reached:** If the agent stays incomplete after `maxTurns`, success is set to `false` and the summary includes the captured history so you can diagnose why the run stalled.

### Crafting effective unary cases
- Supply `expected_arguments` as tightly scoped JSON so AJV can flag extra/missing fields. Reference examples in `cases.jsonl` for structure.
- Keep the `input` prompt self-contained; the harness only adds the optional `instructions` string to the agent system message.
- Use a `maxTurns` of at least `2` so a tool call followed by a final answer can complete. Bump it higher if the provider tends to send clarification turns.
- When you need to assert specific text in the final assistant response, do it inside the tool implementation or downstream checks—`testOutputData` intentionally focuses on structural validation so provider variations in prose do not break CI.

## Streaming path overview

Running `npm start -- --streaming` switches cases onto the streaming surface. The harness still enforces the unary assertions, but it also collects live events so providers can prove they conform to OpenAI's streaming contract.

The collector normalizes both nested `raw_model_stream_event` envelopes and bare model events, so providers that strip the wrapper still satisfy the reasoning delta/done checks.

### Request lifecycle
- `runCase` calls `runner.run` with `{ stream: true }`. Providers that support streaming return a `StreamedRunResult`, which is both `asyncIterable` (for events) and exposes a `completed` promise with the final summary payload.
- While iterating over the stream, the harness pushes `event.data.event` into an internal buffer: these are the raw OpenAI model events after stripping the transport wrapper. `testEvents` handles both bare events and the `raw_model_stream_event` envelope, so either shape is acceptable.
- After the stream finishes (`await result.completed`), the harness continues with the same output validations used in unary mode so the final response contents get cross-checked as well.

### What counts as a failure
- **Missing reasoning events:** `testEvents` expects every Responses run to emit both `response.reasoning_text.delta` and `response.reasoning_text.done`. Chat runs must surface at least one `choices[0].delta.reasoning` string. Absence of these flips `validEvents` to `false`, regardless of whether the provider uses the wrapper or bare event shape.
- **Malformed event shapes:** Events missing the nested `.data.event.type` field are treated as unknown and logged in the summary. When no recognizable reasoning events are present the case fails even if the rest of the run succeeds.
- **Runner fallback to unary:** If the provider ignores the stream flag and returns a plain `RunResult`, the harness treats this as a failure because there are no events to validate. You will see `validEvents: false` with `details.skippedBecauseStreaming` or `missingEvents` markers.
- **Standard unary failures:** Any tool-call or output-structure issues described in the previous section still apply; the streaming checks are additive.

### Crafting effective streaming cases
- Pick prompts that exercise the longest reasoning chains you want to validate. Richer outputs generally produce multiple delta events and make it easier to spot regressions.
- Keep tool arguments deterministic; streaming runs still parse and validate tool calls via AJV before any event checks are performed.
- When reconciling provider differences, inspect the captured event history in the compatibility report. It shows the exact type strings that `testEvents` saw, which helps map provider-specific envelopes to the expected OpenAI events.
- If a provider emits additional event types, that is fine—the harness only requires the reasoning events to be present, but logging them can surface unexpected changes during review.

## How to run

0. Run `npm install` in this directory.
1. Update `providers.ts` to create an entry for the API to test. Change `vllm` to the provider name of your choice. Use `chat` for Chat Completions tests and `responses` for Responses API tests.
2. Run an initial quick test to make sure things work. This will only run one test

```
npm start -- --provider <name> -n 1 -k 1
```

3. Run the full test (runs each test 5 times to test consistency)

```
npm start -- --provider <name> -k 5
```

## Considerations

1. The tests will fail if the API shape does not match the expected behavior
2. Chat streaming checks only look for reasoning deltas; other event types are still ignored
3. If the schema validation succeeds but the input is wrong the test will still pass for this test. That's because it's likely more of a prompt engineering issue or a validator issue than an API issue as it still nailed the input

## CLI flags

| Flag | Alias | Default | Description |
| ---- | ----- | ------- | ----------- |
| `--cases` | `-c` | `cases.jsonl` | Path to the JSONL file that defines test cases. Accepts absolute or relative paths. |
| `--provider` | `-p` | `openai` | Key from `providers.ts` describing the API under test. The harness will iterate over every `apiType` listed for this provider. |
| `--streaming` | `-s` | `false` | When set, requests streaming results from the provider and enables event validation. Leave unset for unary runs. |
| `--maxTurns` | `-t` | `10` | Maximum number of turns the runner will allow before marking the case as incomplete. Increase this if your provider typically requires multiple tool exchanges. |
| `--n` | `-n` | — | Limits execution to the first _n_ cases in the JSONL file. Useful for smoke tests. Must be a positive integer. |
| `--tries` | `-k` | `1` | Number of attempts per case. The harness records every attempt so you can assess flakiness. |
| `--strict` | `-s`* | `false` | Enforces an exact match between emitted tool arguments and `expected_arguments`. *`-s` is shared with `--streaming`; use the long form to avoid ambiguity.* |

All runs produce two artifacts in the working directory:

- `rollout_<provider>_<timestamp>.jsonl` — per-attempt records including success flags, tool-calling details, and raw provider payloads.
- `analysis_<provider>_<timestamp>.json` — aggregate stats generated by `analysis.ts`, used for quick pass/fail summaries.
