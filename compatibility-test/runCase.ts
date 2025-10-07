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

const LOG_NAMESPACE = "runCase";
const debugEnabled = process.env.RUN_CASE_DEBUG === "1";

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

function emitLog(
  logger: ((message: string) => void) | undefined,
  message: string,
  payload?: unknown,
) {
  if (logger) {
    const suffix = payload !== undefined ? ` ${safeStringify(payload)}` : "";
    try {
      logger(`${message}${suffix}`);
    } catch (error) {
      if (debugEnabled) {
        // eslint-disable-next-line no-console
        console.log(`[${LOG_NAMESPACE}] logger error`, error);
      }
    }
  }

  if (!debugEnabled) {
    return;
  }

  if (payload === undefined) {
    // eslint-disable-next-line no-console
    console.log(`[${LOG_NAMESPACE}] ${message}`);
  } else {
    // eslint-disable-next-line no-console
    console.log(`[${LOG_NAMESPACE}] ${message}`, payload);
  }
}

export type RunCaseOptions = {
  maxTurns: number;
  streaming: boolean;
  strict: boolean;
  log?: (message: string) => void;
};

export type Case = {
  tool_name: string;
  input: string;
  expected_arguments: string;
  instructions?: string;
};

// Summary shape for each apiType
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
      result = await runner.run(agent, caseData.input, {
        stream: streaming,
        maxTurns: maxTurns,
      });
      if (result instanceof StreamedRunResult) {
        // Collect streaming events if applicable
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

function testToolCall(apiType, caseData, result, strict, log?: (message: string) => void) {
  const details: Record<string, any> = {};

  for (const item of result.newItems ?? []) {
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

function testOutputData(apiType, rawResponses, streaming, log?: (message: string) => void) {
  let details: Record<string, boolean> = {};
  let validResponse: boolean = false;

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
        const hasSummaryArray = Array.isArray(item.summary) && item.summary.length > 0;

        details.hasReasoningContentArray = Array.isArray(item.content);
        details.hasReasoningContentArrayLength = contentArray.length > 0;
        details.hasReasoningContentArrayItemType = contentArray.every(
          (c: any) => c.type === "reasoning_text" || c.type === "input_text",
        );
        details.hasReasoningContentArrayItemText = contentArray.every(
          (c: any) => typeof c.text === "string" && c.text.length > 0,
        );
        details.hasReasoningSummary = hasSummaryArray;

        const hasValidContent =
          details.hasReasoningContentArray &&
          details.hasReasoningContentArrayLength &&
          details.hasReasoningContentArrayItemType &&
          details.hasReasoningContentArrayItemText;

        validResponse = validResponse || hasValidContent || hasSummaryArray;
        emitLog(log, "responses reasoning check", {
          hasValidContent,
          hasSummaryArray,
          itemSummaryLength: Array.isArray(item.summary) ? item.summary.length : undefined,
        });
      }
    }
  }

  return {
    validResponse,
    details,
  };
}

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
