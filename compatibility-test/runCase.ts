/**
 * Compatibility harness for exercising provider implementations of OpenAI's
 * Responses and Chat Completions interfaces. The exported `runCase` utility
 * executes a single test case across every configured API type, aggregates the
 * raw provider payloads, and evaluates whether the provider satisfied the
 * expected tool-calling and reasoning contracts.
 *
 * The module favors explicit logging and structured summaries so downstream
 * tooling—CLI logs, rollout JSONL files, and analytics—can explain why a given
 * attempt passed or failed without re-running the test.
 */
import {
  Agent,
  Runner,
  OpenAIResponsesModel,
  OpenAIChatCompletionsModel,
  RunResult,
  StreamedRunResult,
  FunctionTool,
  setTracingDisabled,
} from "@openai/agents";
import { Ajv } from "ajv";
import { OpenAI } from "openai";
import { PROVIDERS } from "./providers";
import { TOOLS_MAP } from "./tools";

setTracingDisabled(true);

const ajv = new Ajv();

/**
 * Safely serialize debug payloads for log output. We defensively stringify to
 * avoid crashing when providers return circular references or other structures
 * that `JSON.stringify` cannot process.
 */
function safeStringify(payload: unknown): string {
  if (typeof payload === "string") {
    return payload;
  }
  try {
    return JSON.stringify(payload);
  } catch (error) {
    return String(payload);
  }
}

/**
 * Emit a structured diagnostic entry to the optional logger supplied by the
 * CLI. Logging is intentionally best-effort: failures are swallowed so we
 * never compromise the main compatibility run.
 */
function emitLog(
  logger: ((message: string) => void) | undefined,
  message: string,
  payload?: unknown,
) {
  if (!logger) {
    return;
  }

  const suffix = payload !== undefined ? ` ${safeStringify(payload)}` : "";
  try {
    logger(`${message}${suffix}`);
  } catch (error) {
    // Swallow logger errors so we never disrupt the run loop.
  }
}

/**
 * Per-case runner configuration supplied by the CLI.
 *
 * - `maxTurns` mirrors the agent exchange limit. Providers that require
 *   multiple clarification loops can bump this in the CLI.
 * - `streaming` toggles between unary `RunResult` and streaming
 *   `StreamedRunResult` execution paths.
 * - `strict` enforces an exact match between emitted tool arguments and the
 *   JSON supplied in the case definition.
 * - `log` (optional) accepts the CLI logger so `runCase` diagnostics land in
 *   the same verbose log file the CLI writes when `--verbose` is set.
 */
export type RunCaseOptions = {
  maxTurns: number;
  streaming: boolean;
  strict: boolean;
  log?: (message: string) => void;
};

/**
 * Minimal description of a single compatibility test. Every entry is mirrored
 * in `cases.jsonl` and can optionally supplement the agent with additional
 * instructions beyond the natural-language input.
 */
export type Case = {
  tool_name: string;
  input: string;
  expected_arguments: string;
  instructions?: string;
};

/**
 * Summary returned for each API type the provider exposes. A single call to
 * `runCase` can yield multiple entries (e.g., `responses` and `chat`).
 *
 * `details` and `toolCallingDetails` intentionally capture raw booleans and
 * warnings so higher-level reports can render human-readable diagnostics
 * without rehydrating provider payloads.
 */
export type RunCaseSummary = {
  apiType: string;
  success: boolean;
  validResponse: boolean;
  validEvents?: boolean;
  details: Record<string, any>;
  history: any[];
  successToolCall: boolean;
  toolCallingDetails: Record<string, any>;
};

/**
 * Execute a single compatibility test against the specified provider. The
 * harness iterates over every API type declared in the provider config so a
 * single case can be replayed against both Responses and Chat Completions.
 *
 * Each iteration gathers raw responses, streaming events (when enabled), and
 * validation artifacts. The function returns a summary per API type so callers
 * can persist granular pass/fail information.
 */
export async function runCase(
  provider: string,
  caseData: Case,
  options: RunCaseOptions,
): Promise<RunCaseSummary[]> {
  const config = PROVIDERS[provider];
  if (!config) {
    throw new Error(
      `Provider ${provider} not found. Valid providers are: ${Object.keys(
        PROVIDERS
      ).join(", ")}`
    );
  }

  const { maxTurns, streaming, strict, log } = options;

  const agent = new Agent({
    name: caseData.tool_name,
    instructions: caseData.instructions,
    tools: [TOOLS_MAP[caseData.tool_name]],
  });

  const client = new OpenAI({
    apiKey: config.apiKey,
    baseURL: config.apiBaseUrl,
  });

  emitLog(log, "starting runCase", {
    provider,
    streaming,
    strict,
    tool: caseData.tool_name,
  });

  const summaries: RunCaseSummary[] = [];

  for (const apiType of config.apiType) {
    const runner = new Runner({
      model:
        apiType === "responses"
          ? new OpenAIResponsesModel(client, config.modelName)
          : new OpenAIChatCompletionsModel(client, config.modelName),
      modelSettings: {
        providerData: config.providerDetails ?? {},
      },
    });

    let result: RunResult<any, any> | StreamedRunResult<any, any>;
    let streamedEvents: any[] | undefined = undefined;
    if (streaming) {
      // Request a streaming interaction so we can capture reasoning deltas in
      // real time. Providers that honor the stream flag return an async
      // iterator exposing raw model events.
      result = await runner.run(agent, caseData.input, {
        stream: streaming,
        maxTurns: maxTurns,
      });
      if (result instanceof StreamedRunResult) {
        // Accumulate streaming events for post-run validation.
        streamedEvents = [];
        for await (const event of result) {
          if (event.type === "raw_model_stream_event") {
            if (event.data.type === "model") {
              streamedEvents.push(event.data.event);
            }
          }
        }
        await result.completed;
      }
    } else {
      result = await runner.run(agent, caseData.input, {
        maxTurns: maxTurns,
      });
    }

    const { success: successToolCall, details: toolCallingDetails } = testToolCall(
      apiType,
      caseData,
      result,
      strict,
      log,
    );

    const { validResponse, details } = testOutputData(
      apiType,
      result.rawResponses,
      streaming,
      log,
    );

    const { validEvents, details: eventsDetails } = streaming
      ? testEvents(apiType, streamedEvents ?? [], log)
      : { validEvents: true, details: {} };

    let success = successToolCall && validResponse;
    if (streaming) {
      success = success && validEvents;
    }
    const summary: RunCaseSummary = {
      apiType,
      success,
      validResponse,
      validEvents,
      details: {
        ...details,
        ...eventsDetails,
      },
      history: result?.rawResponses.map((entry) => entry.providerData) ?? [],
      successToolCall,
      toolCallingDetails,
    };

    emitLog(log, "runCase summary", {
      apiType,
      success,
      toolCallingDetails,
      details: summary.details,
    });

    summaries.push(summary);
  }

  return summaries;
}

/**
 * Validate that the provider issued the expected tool call at least once and
 * that the emitted arguments satisfy the declared JSON schema. When `strict`
 * mode is active we additionally require a deep-equal match against the test
 * case's `expected_arguments`.
 */
function testToolCall(
  apiType,
  caseData,
  result,
  strict,
  log?: (message: string) => void,
) {
  const details: Record<string, any> = {};

  for (const item of result.newItems ?? []) {
    // Do not early-return: later turns may contain the first valid tool call,
    // especially when the provider emits observations before invoking tools.
    if (item?.type !== "tool_call_item") {
      continue;
    }

    const raw = item.rawItem ?? item;
    const rawType = item.rawItem?.type ?? item.rawType ?? raw.type;
    if (rawType !== "function_call") {
      continue;
    }

    const toolName = raw.name ?? item.rawItem?.name ?? item.name;
    if (toolName !== caseData.tool_name) {
      continue;
    }

    details.calledToolAtLeastOnce = true;

    const schema = (TOOLS_MAP[caseData.tool_name] as FunctionTool).parameters;
    const validate = ajv.compile(schema);

    const parsedArguments = normalizeArguments(raw.arguments);
    const schemaValid = validate(parsedArguments);
    details.calledToolWithRightSchema = schemaValid;

    emitLog(log, "tool call inspected", {
      apiType,
      toolName,
      schemaValid,
      parsedArguments,
    });

    if (!schemaValid) {
      if (validate.errors) {
        details.schemaErrors = validate.errors;
        emitLog(log, "schema validation errors", validate.errors);
      }
      continue;
    }

    const expectedArguments = normalizeArguments(caseData.expected_arguments);
    const argumentsMatch = deepEqual(parsedArguments, expectedArguments);
    details.calledToolWithRightArguments = argumentsMatch;

    if (!argumentsMatch) {
      details.warning = `Tool call with wrong arguments but correct schema. Parsed: ${JSON.stringify(parsedArguments)} Expected: ${JSON.stringify(expectedArguments)}`;
      details.actualArguments = parsedArguments;
      details.expectedArguments = expectedArguments;
      emitLog(log, "tool argument mismatch", {
        parsedArguments,
        expectedArguments,
        strict,
      });
    }
  }

  return {
    success:
      !!details.calledToolAtLeastOnce &&
      !!details.calledToolWithRightSchema &&
      (!strict || !!details.calledToolWithRightArguments),
    details,
  };
}

/**
 * Evaluate streaming event sequences for the reasoning signals exposed by the
 * OpenAI Responses and Chat Completions APIs. Providers that opt out of
 * streaming or omit the reasoning events are marked as invalid so parity gaps
 * surface during compatibility sweeps.
 */
function testEvents(apiType, events, log?: (message: string) => void) {
  let details: Record<string, boolean> = {};
  let validEvents: boolean = false;

  const observedEvents = events ?? [];

  if (observedEvents.length === 0) {
    details.missingEvents = true;
    emitLog(log, "no streaming events observed", { apiType });
  }

  if (apiType === "chat") {
    let hasReasoningDeltas = false;
    for (const event of observedEvents) {
      const reasoning = event?.choices?.[0]?.delta?.reasoning;
      if (typeof reasoning === "string" && reasoning.length > 0) {
        hasReasoningDeltas = true;
        break;
      }
    }
    details.hasReasoningDeltas = hasReasoningDeltas;
    validEvents = hasReasoningDeltas;
    emitLog(log, "chat streaming check", {
      hasReasoningDeltas,
      eventCount: observedEvents.length,
    });
  }

  if (apiType === "responses") {
    let hasReasoningDeltaEvents = false;
    let hasReasoningDoneEvents = false;

    for (const event of observedEvents) {
      const eventType = (
        event?.data?.event?.type ??
        event?.type ??
        event?.data?.type ??
        ""
      );

      if (eventType === "response.reasoning_text.delta") {
        hasReasoningDeltaEvents = true;
      }

      if (eventType === "response.reasoning_text.done") {
        hasReasoningDoneEvents = true;
      }
    }

    details.hasReasoningDeltaEvents = hasReasoningDeltaEvents;
    details.hasReasoningDoneEvents = hasReasoningDoneEvents;
    validEvents = hasReasoningDeltaEvents && hasReasoningDoneEvents;
    emitLog(log, "responses streaming check", {
      hasReasoningDeltaEvents,
      hasReasoningDoneEvents,
      eventCount: observedEvents.length,
    });
  }

  return {
    validEvents,
    details,
  };
}

/**
 * Inspect the provider's final response payloads. For unary Chat runs we
 * require at least one assistant message with textual reasoning. For Responses
 * runs we accept either populated `reasoning_text` items or a `summary` array.
 * The latter is treated as a soft pass and surfaces a warning so downstream
 * reviewers know the provider returned minimal reasoning detail.
 */
function testOutputData(apiType, rawResponses, streaming, log?: (message: string) => void) {
  let details: Record<string, boolean> = {};
  let validResponse: boolean = false;
  let warning: string | undefined;

  if (!Array.isArray(rawResponses) || rawResponses.length === 0) {
    return {
      validResponse: false,
      details: {
        missingResponses: true,
      },
    };
  }

  if (apiType === "chat") {
    for (const response of rawResponses) {
      if (streaming && !response.providerData) {
        return {
          validResponse: true,
          details: {
            skippedBecauseStreaming: true,
          },
        };
      }

      const data = response.providerData;
      const message = data.choices[0].message;
      if (message.role === "assistant" && !message.refusal) {
        details.hasReasoningField =
          details.hasReasoningField ||
          ("reasoning" in message && typeof message.reasoning === "string");
        details.hasReasoningContentField =
          details.hasReasoningContentField ||
          ("reasoning_content" in message &&
            typeof message.reasoning_content === "string");

        validResponse =
          validResponse ||
          (details.hasReasoningField && message.reasoning.length > 0);
      }
    }
  } else if (apiType === "responses") {
    const top = rawResponses[0];
    const data = top.providerData ?? top;

    if (!data || !Array.isArray(data.output)) {
      emitLog(log, "responses output missing", { dataPresent: !!data });
      return { validResponse: false, details: { missingOutput: true } };
    }

    for (const item of data.output) {
      if (item.type === "unknown" && typeof item.providerData?.type === "string") {
        item.type = item.providerData.type;
      }
    }

    for (const item of data.output) {
      if (item.type === "reasoning") {
        const contentArray = Array.isArray(item.content) ? item.content : [];
        const summaryArray = Array.isArray(item.summary) ? item.summary : undefined;
        const hasSummaryArray = Array.isArray(summaryArray);
        const summaryLength = Array.isArray(summaryArray)
          ? summaryArray.length
          : undefined;

        details.hasReasoningContentArray = Array.isArray(item.content);
        details.hasReasoningContentArrayLength = contentArray.length > 0;
        details.hasReasoningContentArrayItemType = contentArray.every(
          (c: any) => c.type === "reasoning_text" || c.type === "input_text",
        );
        details.hasReasoningContentArrayItemText = contentArray.every(
          (c: any) => typeof c.text === "string" && c.text.length > 0,
        );
        details.hasReasoningSummary = hasSummaryArray;
        if (typeof summaryLength === "number") {
          details.hasReasoningSummaryLength = summaryLength;
        }

        const hasValidContent =
          details.hasReasoningContentArray &&
          details.hasReasoningContentArrayLength &&
          details.hasReasoningContentArrayItemType &&
          details.hasReasoningContentArrayItemText;

        const acceptedBySummary = !hasValidContent && hasSummaryArray;
        if (acceptedBySummary && !warning) {
          if (summaryLength && summaryLength > 0) {
            warning = `Reasoning output lacked content but included a summary array with ${summaryLength} entr${summaryLength === 1 ? "y" : "ies"}. Counting as pass.`;
          } else {
            warning =
              "Reasoning output lacked content; summary array present but empty. Counting as pass.";
          }
        }

        validResponse = validResponse || hasValidContent || acceptedBySummary;
        emitLog(log, "responses reasoning check", {
          hasValidContent,
          hasSummaryArray,
          summaryLength,
          acceptedBySummary,
          itemSummaryLength: Array.isArray(item.summary) ? item.summary.length : undefined,
        });
      }
    }
  }

  return {
    validResponse,
    details: warning
      ? {
          ...details,
          warning,
        }
      : details,
  };
}

/**
 * Convert tool arguments into a comparable object. Providers frequently return
 * arguments as JSON strings; this helper ensures we can validate with AJV and
 * compare against the expected payload without caring about the original
 * encoding.
 */
function normalizeArguments(payload: unknown): Record<string, any> {
  if (payload == null) {
    return {};
  }

  if (typeof payload === "string") {
    try {
      return JSON.parse(payload);
    } catch (error) {
      return {};
    }
  }

  if (typeof payload === "object") {
    return payload as Record<string, any>;
  }

  return {};
}

/**
 * Recursively compare two JSON-like structures for equality. AJV guarantees
 * the schema shape, but strict mode requires us to verify the provider
 * supplied exactly the same values as the test case.
 */
function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (a && b && typeof a === "object") {
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    if (Array.isArray(a)) {
      if (a.length !== b.length) return false;
      for (let i = 0; i < a.length; i++) {
        if (!deepEqual(a[i], b[i])) return false;
      }
      return true;
    } else {
      const aKeys = Object.keys(a);
      const bKeys = Object.keys(b);
      if (aKeys.length !== bKeys.length) return false;
      for (const key of aKeys) {
        if (!b.hasOwnProperty(key)) return false;
        if (!deepEqual(a[key], b[key])) return false;
      }
      return true;
    }
  }
  return false;
}
