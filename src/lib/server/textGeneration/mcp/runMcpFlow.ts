import { config } from "$lib/server/config";
import { MessageUpdateType, type MessageUpdate } from "$lib/types/MessageUpdate";
import { getMcpServers } from "$lib/server/mcp/registry";
import { isValidMcpUrl } from "$lib/server/urlSafety";
import { resetMcpToolsCache } from "$lib/server/mcp/tools";
import { getOpenAiToolsForMcp } from "$lib/server/mcp/tools";
import type {
	ChatCompletionChunk,
	ChatCompletionCreateParamsStreaming,
	ChatCompletionMessageParam,
	ChatCompletionMessageToolCall,
} from "openai/resources/chat/completions";
import type { Stream } from "openai/streaming";
import { buildToolPreprompt } from "../utils/toolPrompt";
import type { EndpointMessage } from "../../endpoints/endpoints";
import { resolveRouterTarget } from "./routerResolution";
import { executeToolCalls, type NormalizedToolCall } from "./toolInvocation";
import { drainPool } from "$lib/server/mcp/clientPool";
import type { TextGenerationContext } from "../types";
import { hasAuthHeader, isStrictHfMcpLogin, hasNonEmptyToken } from "$lib/server/mcp/hf";
import { buildImageRefResolver } from "./fileRefs";
import { prepareMessagesWithFiles } from "$lib/server/textGeneration/utils/prepareFiles";
import { makeImageProcessor } from "$lib/server/endpoints/images";

export type RunMcpFlowContext = Pick<
	TextGenerationContext,
	"model" | "conv" | "assistant" | "forceMultimodal" | "forceTools" | "locals"
> & { messages: EndpointMessage[] };

export async function* runMcpFlow({
	model,
	conv,
	messages,
	assistant,
	forceMultimodal,
	forceTools,
	locals,
	preprompt,
	abortSignal,
}: RunMcpFlowContext & { preprompt?: string; abortSignal?: AbortSignal }): AsyncGenerator<
	MessageUpdate,
	boolean,
	undefined
> {
	// Start from env-configured servers
	let servers = getMcpServers();
	try {
		console.debug(
			{ baseServers: servers.map((s) => ({ name: s.name, url: s.url })), count: servers.length },
			"[mcp] base servers loaded"
		);
	} catch {}

	// Merge in request-provided custom servers (if any)
	try {
		const reqMcp = (
			locals as unknown as {
				mcp?: {
					selectedServers?: Array<{ name: string; url: string; headers?: Record<string, string> }>;
					selectedServerNames?: string[];
				};
			}
		)?.mcp;
		const custom = Array.isArray(reqMcp?.selectedServers) ? reqMcp?.selectedServers : [];
		if (custom.length > 0) {
			// Invalidate cached tool list when the set of servers changes at request-time
			resetMcpToolsCache();
			// Deduplicate by server name (request takes precedence)
			const byName = new Map<
				string,
				{ name: string; url: string; headers?: Record<string, string> }
			>();
			for (const s of servers) byName.set(s.name, s);
			for (const s of custom) byName.set(s.name, s);
			servers = [...byName.values()];
			try {
				console.debug(
					{
						customProvidedCount: custom.length,
						mergedServers: servers.map((s) => ({
							name: s.name,
							url: s.url,
							hasAuth: !!s.headers?.Authorization,
						})),
					},
					"[mcp] merged request-provided servers"
				);
			} catch {}
		}

		// If the client specified a selection by name, filter to those
		const names = Array.isArray(reqMcp?.selectedServerNames)
			? reqMcp?.selectedServerNames
			: undefined;
		if (Array.isArray(names)) {
			const before = servers.map((s) => s.name);
			servers = servers.filter((s) => names.includes(s.name));
			try {
				console.debug(
					{ selectedNames: names, before, after: servers.map((s) => s.name) },
					"[mcp] applied name selection"
				);
			} catch {}
		}
	} catch {
		// ignore selection merge errors and proceed with env servers
	}

	// If selection/merge yielded no servers, bail early with clearer log
	if (servers.length === 0) {
		console.warn("[mcp] no MCP servers selected after merge/name filter");
		return false;
	}

	// Enforce server-side safety (public HTTPS only, no private ranges)
	{
		const before = servers.slice();
		servers = servers.filter((s) => {
			try {
				return isValidMcpUrl(s.url);
			} catch {
				return false;
			}
		});
		try {
			const rejected = before.filter((b) => !servers.includes(b));
			if (rejected.length > 0) {
				console.warn(
					{ rejected: rejected.map((r) => ({ name: r.name, url: r.url })) },
					"[mcp] rejected servers by URL safety"
				);
			}
		} catch {}
	}
	if (servers.length === 0) {
		console.warn("[mcp] all selected MCP servers rejected by URL safety guard");
		return false;
	}

	// Optionally attach the logged-in user's HF token to the official HF MCP server only.
	// Never override an explicit Authorization header, and require token to look like an HF token.
	try {
		const shouldForward = config.MCP_FORWARD_HF_USER_TOKEN === "true";
		const userToken =
			(locals as unknown as { hfAccessToken?: string } | undefined)?.hfAccessToken ??
			(locals as unknown as { token?: string } | undefined)?.token;

		if (shouldForward && hasNonEmptyToken(userToken)) {
			const overlayApplied: string[] = [];
			servers = servers.map((s) => {
				try {
					if (isStrictHfMcpLogin(s.url) && !hasAuthHeader(s.headers)) {
						overlayApplied.push(s.name);
						return {
							...s,
							headers: { ...(s.headers ?? {}), Authorization: `Bearer ${userToken}` },
						};
					}
				} catch {
					// ignore URL parse errors and leave server unchanged
				}
				return s;
			});
			if (overlayApplied.length > 0) {
				try {
					console.debug({ overlayApplied }, "[mcp] forwarded HF token to servers");
				} catch {}
			}
		}
	} catch {
		// best-effort overlay; continue if anything goes wrong
	}
	console.debug(
		{ count: servers.length, servers: servers.map((s) => s.name) },
		"[mcp] servers configured"
	);
	if (servers.length === 0) {
		return false;
	}

	// Gate MCP flow based on model tool support (aggregated) with user override
	try {
		const supportsTools = Boolean((model as unknown as { supportsTools?: boolean }).supportsTools);
		const toolsEnabled = Boolean(forceTools) || supportsTools;
		console.debug(
			{
				model: model.id ?? model.name,
				supportsTools,
				forceTools: Boolean(forceTools),
				toolsEnabled,
			},
			"[mcp] tools gate evaluation"
		);
		if (!toolsEnabled) {
			console.info(
				{ model: model.id ?? model.name },
				"[mcp] tools disabled for model; skipping MCP flow"
			);
			return false;
		}
	} catch {
		// If anything goes wrong reading the flag, proceed (previous behavior)
	}

	const resolveFileRef = buildImageRefResolver(messages);
	const imageProcessor = makeImageProcessor({
		supportedMimeTypes: ["image/png", "image/jpeg"],
		preferredMimeType: "image/jpeg",
		maxSizeInMB: 1,
		maxWidth: 1024,
		maxHeight: 1024,
	});

	const hasImageInput = messages.some((msg) =>
		(msg.files ?? []).some(
			(file) => typeof file?.mime === "string" && file.mime.startsWith("image/")
		)
	);

	const { runMcp, targetModel, candidateModelId, resolvedRoute } = await resolveRouterTarget({
		model,
		messages,
		conversationId: conv._id.toString(),
		hasImageInput,
		locals,
	});

	if (!runMcp) {
		console.info(
			{ model: targetModel.id ?? targetModel.name, resolvedRoute },
			"[mcp] runMcp=false (routing chose non-tools candidate)"
		);
		return false;
	}

	const { tools: oaTools, mapping } = await getOpenAiToolsForMcp(servers, { signal: abortSignal });
	try {
		console.info(
			{ toolCount: oaTools.length, toolNames: oaTools.map((t) => t.function.name) },
			"[mcp] openai tool defs built"
		);
	} catch {}
	if (oaTools.length === 0) {
		console.warn("[mcp] zero tools available after listing; skipping MCP flow");
		return false;
	}

	try {
		const { OpenAI } = await import("openai");

		// Capture provider header (x-inference-provider) from the upstream OpenAI-compatible server.
		let providerHeader: string | undefined;
		const captureProviderFetch = async (
			input: RequestInfo | URL,
			init?: RequestInit
		): Promise<Response> => {
			const res = await fetch(input, init);
			const p = res.headers.get("x-inference-provider");
			if (p && !providerHeader) providerHeader = p;
			return res;
		};

		const openai = new OpenAI({
			apiKey: config.OPENAI_API_KEY || config.HF_TOKEN || "sk-",
			baseURL: config.OPENAI_BASE_URL,
			fetch: captureProviderFetch,
			defaultHeaders: {
				// Bill to organization if configured (HuggingChat only)
				...(config.isHuggingChat && locals?.billingOrganization
					? { "X-HF-Bill-To": locals.billingOrganization }
					: {}),
			},
		});

		const mmEnabled = (forceMultimodal ?? false) || targetModel.multimodal;
		console.info(
			{
				targetModel: targetModel.id ?? targetModel.name,
				mmEnabled,
				route: resolvedRoute,
				candidateModelId,
				toolCount: oaTools.length,
				hasUserToken: Boolean((locals as unknown as { token?: string })?.token),
			},
			"[mcp] starting completion with tools"
		);
		let messagesOpenAI: ChatCompletionMessageParam[] = await prepareMessagesWithFiles(
			messages,
			imageProcessor,
			mmEnabled
		);
		const toolPreprompt = buildToolPreprompt(oaTools);
		const prepromptPieces: string[] = [];
		if (toolPreprompt.trim().length > 0) {
			prepromptPieces.push(toolPreprompt);
		}
		if (typeof preprompt === "string" && preprompt.trim().length > 0) {
			prepromptPieces.push(preprompt);
		}
		const mergedPreprompt = prepromptPieces.join("\n\n");
		const hasSystemMessage = messagesOpenAI.length > 0 && messagesOpenAI[0]?.role === "system";
		if (hasSystemMessage) {
			if (mergedPreprompt.length > 0) {
				const existing = messagesOpenAI[0].content ?? "";
				const existingText = typeof existing === "string" ? existing : "";
				messagesOpenAI[0].content = mergedPreprompt + (existingText ? "\n\n" + existingText : "");
			}
		} else if (mergedPreprompt.length > 0) {
			messagesOpenAI = [{ role: "system", content: mergedPreprompt }, ...messagesOpenAI];
		}

		// Work around servers that reject `system` role
		if (
			typeof config.OPENAI_BASE_URL === "string" &&
			config.OPENAI_BASE_URL.length > 0 &&
			(config.OPENAI_BASE_URL.includes("hf.space") ||
				config.OPENAI_BASE_URL.includes("gradio.app")) &&
			messagesOpenAI[0]?.role === "system"
		) {
			messagesOpenAI[0] = { ...messagesOpenAI[0], role: "user" };
		}

		const parameters = { ...targetModel.parameters, ...assistant?.generateSettings } as Record<
			string,
			unknown
		>;
		const maxTokens =
			(parameters?.max_tokens as number | undefined) ??
			(parameters?.max_new_tokens as number | undefined) ??
			(parameters?.max_completion_tokens as number | undefined);

		const stopSequences =
			typeof parameters?.stop === "string"
				? parameters.stop
				: Array.isArray(parameters?.stop)
					? (parameters.stop as string[])
					: undefined;

		const completionBase: Omit<ChatCompletionCreateParamsStreaming, "messages"> = {
			model: targetModel.id ?? targetModel.name,
			stream: true,
			temperature: typeof parameters?.temperature === "number" ? parameters.temperature : undefined,
			top_p: typeof parameters?.top_p === "number" ? parameters.top_p : undefined,
			frequency_penalty:
				typeof parameters?.frequency_penalty === "number"
					? parameters.frequency_penalty
					: typeof parameters?.repetition_penalty === "number"
						? parameters.repetition_penalty
						: undefined,
			presence_penalty:
				typeof parameters?.presence_penalty === "number" ? parameters.presence_penalty : undefined,
			stop: stopSequences,
			max_tokens: typeof maxTokens === "number" ? maxTokens : undefined,
			tools: oaTools,
			tool_choice: "auto",
		};
		const completionBaseNoTools: Omit<ChatCompletionCreateParamsStreaming, "messages"> = {
			...completionBase,
			tools: undefined,
			tool_choice: undefined,
		};

		const toPrimitive = (value: unknown) => {
			if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
				return value;
			}
			return undefined;
		};

		const tryParseJsonString = (value: string): unknown | undefined => {
			const trimmed = value.trim();
			const looksLikeJson =
				(trimmed.startsWith("{") && trimmed.endsWith("}")) ||
				(trimmed.startsWith("[") && trimmed.endsWith("]"));
			if (!looksLikeJson) return undefined;
			try {
				return JSON.parse(trimmed);
			} catch {
				return undefined;
			}
		};

		const reviveNestedJsonStrings = (value: unknown, depth = 0): unknown => {
			if (depth > 4) return value;
			if (typeof value === "string") {
				const parsed = tryParseJsonString(value);
				return parsed === undefined ? value : reviveNestedJsonStrings(parsed, depth + 1);
			}
			if (Array.isArray(value)) {
				return value.map((v) => reviveNestedJsonStrings(v, depth + 1));
			}
			if (value && typeof value === "object") {
				const out: Record<string, unknown> = {};
				for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
					out[k] = reviveNestedJsonStrings(v, depth + 1);
				}
				return out;
			}
			return value;
		};

		const parseArgs = (raw: unknown): Record<string, unknown> => {
			if (typeof raw !== "string" || raw.trim().length === 0) return {};
			try {
				const parsed = JSON.parse(raw) as unknown;
				const revived = reviveNestedJsonStrings(parsed);
				if (!revived || typeof revived !== "object" || Array.isArray(revived)) return {};
				return revived as Record<string, unknown>;
			} catch {
				return {};
			}
		};

		const processToolOutput = (
			text: string
		): {
			annotated: string;
			sources: { index: number; link: string }[];
		} => ({ annotated: text, sources: [] });

		let lastAssistantContent = "";
		let streamedContent = false;
		// Track whether we're inside a <think> block when the upstream streams
		// provider-specific reasoning tokens (e.g. `reasoning` or `reasoning_content`).
		let thinkOpen = false;
		let requiredToolRetryDone = false;

		const getLastUserText = (msgs: ChatCompletionMessageParam[]): string => {
			const lastUser = [...msgs].reverse().find((m) => m.role === "user");
			const c = (lastUser as unknown as { content?: unknown } | undefined)?.content;
			if (typeof c === "string") return c;
			if (Array.isArray(c)) {
				return c
					.map((part) => {
						if (!part || typeof part !== "object") return "";
						const p = part as { type?: unknown; text?: unknown };
						return p.type === "text" && typeof p.text === "string" ? p.text : "";
					})
					.filter((s) => typeof s === "string" && s.length > 0)
					.join("\n");
			}
			return "";
		};

		const computeShouldRequireTools = (msgs: ChatCompletionMessageParam[]): boolean => {
			if (forceTools) return true;
			try {
				const reqMcp = (
					locals as unknown as {
						mcp?: {
							selectedServers?: Array<unknown>;
							selectedServerNames?: Array<unknown>;
						};
					}
				)?.mcp;
				const hasSelection =
					(Array.isArray(reqMcp?.selectedServers) && reqMcp!.selectedServers.length > 0) ||
					(Array.isArray(reqMcp?.selectedServerNames) && reqMcp!.selectedServerNames.length > 0);
				if (hasSelection) return true;
			} catch {}
			const text = getLastUserText(msgs);
			return /\b(mcp|tool|tools|инструмент|инструменты|mcp-сервер|mcp сервер)\b/i.test(text);
		};

		if (resolvedRoute && candidateModelId) {
			yield {
				type: MessageUpdateType.RouterMetadata,
				route: resolvedRoute,
				model: candidateModelId,
			};
			console.debug(
				{ route: resolvedRoute, model: candidateModelId },
				"[mcp] router metadata emitted"
			);
		}

		for (let loop = 0; loop < 10; loop += 1) {
			lastAssistantContent = "";
			streamedContent = false;
			let nonStreamCalls: NormalizedToolCall[] | null = null;
			const modelName = String(targetModel.id ?? targetModel.name);
			const shouldRequireToolsThisLoop = computeShouldRequireTools(messagesOpenAI);
			const isDeepseekChat = /deepseek-chat/i.test(modelName);
			const isDeepseekReasonerLoop = /deepseek-reasoner/i.test(modelName);
			const useNonStreamCompletion = isDeepseekChat || isDeepseekReasonerLoop;
			const useToolsThisLoop = !((isDeepseekChat || isDeepseekReasonerLoop) && loop > 0);
			const completionBaseThisLoop = useToolsThisLoop ? completionBase : completionBaseNoTools;

			const completionRequest: ChatCompletionCreateParamsStreaming = {
				...completionBaseThisLoop,
				messages: messagesOpenAI,
			};

			const requestAbort = new AbortController();
			const requestTimeoutMs = isDeepseekChat ? 60_000 : 120_000;
			const anySignal = (AbortSignal as unknown as { any?: (signals: AbortSignal[]) => AbortSignal }).any;
			let abortHandler: (() => void) | null = null;
			const requestSignal =
				abortSignal && typeof anySignal === "function"
					? anySignal([abortSignal, requestAbort.signal])
					: requestAbort.signal;
			if (abortSignal && typeof anySignal !== "function") {
				abortHandler = () => requestAbort.abort();
				if (abortSignal.aborted) {
					requestAbort.abort();
				} else {
					abortSignal.addEventListener("abort", abortHandler, { once: true });
				}
			}
			let requestTimeout: ReturnType<typeof setTimeout> | null = null;
			requestTimeout = setTimeout(() => requestAbort.abort(), requestTimeoutMs);
			const completionStartedAt = Date.now();

			let completionStream: Stream<ChatCompletionChunk> | null = null;
			try {
				if (useNonStreamCompletion) {
					console.info(
						{ loop, model: modelName, toolsEnabled: useToolsThisLoop },
						"[mcp] deepseek: using non-stream completion request"
					);
					const nonStream = await openai.chat.completions.create(
						{ ...completionBaseThisLoop, messages: messagesOpenAI, stream: false },
						{
							signal: requestSignal,
							headers: {
								"ChatUI-Conversation-ID": conv._id.toString(),
								"X-use-cache": "false",
								...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
							},
						}
					);
					const msgObj = nonStream.choices?.[0]?.message as
						| (ChatCompletionMessageParam & {
								reasoning?: string;
								reasoning_content?: string;
								tool_calls?: ChatCompletionMessageToolCall[];
							})
						| undefined;
					const contentVal = msgObj?.content;
					const msgContent = typeof contentVal === "string" ? contentVal : "";
					const r =
						typeof msgObj?.reasoning === "string"
							? msgObj.reasoning
							: typeof msgObj?.reasoning_content === "string"
								? msgObj.reasoning_content
								: "";
					lastAssistantContent = (r && r.length > 0 ? `<think>${r}</think>` : "") + msgContent;
					thinkOpen = false;
					const tc: ChatCompletionMessageToolCall[] = Array.isArray(msgObj?.tool_calls)
						? ((msgObj?.tool_calls ?? []) as ChatCompletionMessageToolCall[])
						: [];
					if (tc.length > 0) {
						nonStreamCalls = tc.map((t) => ({
							id: t.id,
							name: t.function?.name ?? "",
							arguments: t.function?.arguments ?? "",
						}));
					}
					console.info(
						{
							loop,
							model: modelName,
							toolsEnabled: useToolsThisLoop,
							durationMs: Date.now() - completionStartedAt,
							contentLength: lastAssistantContent.length,
							toolCallCount: nonStreamCalls?.length ?? 0,
						},
						"[mcp] deepseek: non-stream completion received"
					);
				} else {
					console.info(
						{ loop, model: modelName, toolsEnabled: useToolsThisLoop },
						"[mcp] starting streamed completion request"
					);
					completionStream = await openai.chat.completions.create(completionRequest, {
						signal: requestSignal,
						headers: {
							"ChatUI-Conversation-ID": conv._id.toString(),
							"X-use-cache": "false",
							...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
						},
					});
				}
			} catch (err) {
				if (requestAbort.signal.aborted && !abortSignal?.aborted) {
					throw new Error("MCP completion timed out");
				}
				throw err;
			} finally {
				if (requestTimeout) clearTimeout(requestTimeout);
				if (abortSignal && abortHandler) {
					try {
						abortSignal.removeEventListener("abort", abortHandler);
					} catch {}
				}
				if (useNonStreamCompletion) {
					console.debug(
						{ loop, model: modelName, durationMs: Date.now() - completionStartedAt },
						"[mcp] deepseek: completion request finished"
					);
				}
			}

			// If provider header was exposed, notify UI so it can render "via {provider}".
			if (providerHeader) {
				yield {
					type: MessageUpdateType.RouterMetadata,
					route: "",
					model: "",
					provider: providerHeader as unknown as import("@huggingface/inference").InferenceProvider,
				};
				console.debug({ provider: providerHeader }, "[mcp] provider metadata emitted");
			}

			const toolCallState: Record<number, { id?: string; name?: string; arguments: string }> = {};
			let firstToolDeltaLogged = false;
			let sawToolCall = false;
			let tokenCount = 0;
			if (completionStream) {
				try {
					for await (const chunk of completionStream) {
						const choice = chunk.choices?.[0];
						const delta = choice?.delta;
						if (!delta) continue;

						const chunkToolCalls = delta.tool_calls ?? [];
						if (chunkToolCalls.length > 0) {
							sawToolCall = true;
							for (const call of chunkToolCalls) {
								const toolCall = call as unknown as {
									index?: number;
									id?: string;
									function?: { name?: string; arguments?: string };
								};
								const index = toolCall.index ?? 0;
								const current = toolCallState[index] ?? { arguments: "" };
								if (toolCall.id) current.id = toolCall.id;
								if (toolCall.function?.name) current.name = toolCall.function.name;
								if (toolCall.function?.arguments) current.arguments += toolCall.function.arguments;
								toolCallState[index] = current;
							}
							if (!firstToolDeltaLogged) {
								try {
									const first =
										toolCallState[
											Object.keys(toolCallState)
												.map((k) => Number(k))
												.sort((a, b) => a - b)[0] ?? 0
										];
									console.info(
										{ firstCallName: first?.name, hasId: Boolean(first?.id) },
										"[mcp] observed streamed tool_call delta"
									);
									firstToolDeltaLogged = true;
								} catch {}
							}
						}

						const deltaContent = (() => {
							if (typeof delta.content === "string") return delta.content;
							const maybeParts = delta.content as unknown;
							if (Array.isArray(maybeParts)) {
								return maybeParts
									.map((part) =>
										typeof part === "object" &&
										part !== null &&
										"text" in part &&
										typeof (part as Record<string, unknown>).text === "string"
											? String((part as Record<string, unknown>).text)
											: ""
									)
									.join("");
							}
							return "";
						})();

						// Provider-dependent reasoning fields (e.g., `reasoning` or `reasoning_content`).
						const deltaReasoning: string =
							typeof (delta as unknown as Record<string, unknown>)?.reasoning === "string"
								? ((delta as unknown as { reasoning?: string }).reasoning as string)
								: typeof (delta as unknown as Record<string, unknown>)?.reasoning_content === "string"
									? ((delta as unknown as { reasoning_content?: string }).reasoning_content as string)
									: "";

						let combined = "";
						if (deltaReasoning.trim().length > 0) {
							if (!thinkOpen) {
								combined += "<think>" + deltaReasoning;
								thinkOpen = true;
							} else {
								combined += deltaReasoning;
							}
						}

						if (deltaContent && deltaContent.length > 0) {
							if (thinkOpen) {
								combined += "</think>" + deltaContent;
								thinkOpen = false;
							} else {
								combined += deltaContent;
							}
						}

						if (combined.length > 0) {
							lastAssistantContent += combined;
							if (!sawToolCall && !(shouldRequireToolsThisLoop && loop === 0)) {
								streamedContent = true;
								yield { type: MessageUpdateType.Stream, token: combined };
								tokenCount += combined.length;
							}
						}
					}

					if (!firstToolDeltaLogged) {
						try {
							const first =
								toolCallState[
									Object.keys(toolCallState)
										.map((k) => Number(k))
										.sort((a, b) => a - b)[0] ?? 0
								];
							console.info(
								{ firstCallName: first?.name, hasId: Boolean(first?.id) },
								"[mcp] observed streamed tool_call delta"
							);
							firstToolDeltaLogged = true;
						} catch {}
					}
				} catch (err) {
					const msg = String(err ?? "");
					if (msg.includes("TypeError: terminated") || msg.includes("terminated")) {
						console.warn(
							{ err: msg, loop },
							"[mcp] stream terminated during read; retrying non-stream"
						);
						const nonStream = await openai.chat.completions.create(
							{ ...completionBase, messages: messagesOpenAI, stream: false },
							{
								signal: abortSignal,
								headers: {
									"ChatUI-Conversation-ID": conv._id.toString(),
									"X-use-cache": "false",
									...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
								},
							}
						);

						const msgObj = nonStream.choices?.[0]?.message as
							| (ChatCompletionMessageParam & {
									reasoning?: string;
									reasoning_content?: string;
									tool_calls?: ChatCompletionMessageToolCall[];
								})
							| undefined;
						const contentVal = msgObj?.content;
						const msgContent = typeof contentVal === "string" ? contentVal : "";
						const r =
							typeof msgObj?.reasoning === "string"
								? msgObj.reasoning
								: typeof msgObj?.reasoning_content === "string"
									? msgObj.reasoning_content
									: "";
						lastAssistantContent =
							(r && r.length > 0 ? `<think>${r}</think>` : "") + msgContent;
						thinkOpen = false;

						const tc: ChatCompletionMessageToolCall[] = Array.isArray(msgObj?.tool_calls)
							? ((msgObj?.tool_calls ?? []) as ChatCompletionMessageToolCall[])
							: [];
						if (tc.length > 0) {
							nonStreamCalls = tc.map((t) => ({
								id: t.id,
								name: t.function?.name ?? "",
								arguments: t.function?.arguments ?? "",
							}));
						}
					} else {
						throw err;
					}
				}
			}
			console.info(
				{
					sawToolCalls:
						(nonStreamCalls && nonStreamCalls.length > 0) ||
						Object.keys(toolCallState).length > 0,
					tokens: tokenCount,
					loop,
				},
				"[mcp] completion stream closed"
			);

			if (
				!requiredToolRetryDone &&
				shouldRequireToolsThisLoop &&
				loop === 0 &&
				!useNonStreamCompletion &&
				(!nonStreamCalls || nonStreamCalls.length === 0) &&
				Object.keys(toolCallState).length === 0
			) {
				requiredToolRetryDone = true;
				try {
					console.info(
						{ loop },
						"[mcp] no tool_calls with tool_choice=auto; retrying once with tool_choice=required"
					);
					const nonStream = await openai.chat.completions.create(
						{
							...completionBase,
							messages: messagesOpenAI,
							stream: false,
							tool_choice: "required",
						},
						{
							signal: abortSignal,
							headers: {
								"ChatUI-Conversation-ID": conv._id.toString(),
								"X-use-cache": "false",
								...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
							},
						}
					);

					const msgObj = nonStream.choices?.[0]?.message as
						| (ChatCompletionMessageParam & { tool_calls?: ChatCompletionMessageToolCall[] })
						| undefined;
					const tc: ChatCompletionMessageToolCall[] = Array.isArray(msgObj?.tool_calls)
						? (msgObj?.tool_calls ?? [])
						: [];
					if (tc.length > 0) {
						nonStreamCalls = tc.map((t) => ({
							id: t.id,
							name: t.function?.name ?? "",
							arguments: t.function?.arguments ?? "",
						}));
					}
				} catch {}
			}

			if ((nonStreamCalls && nonStreamCalls.length > 0) || Object.keys(toolCallState).length > 0) {
				// If any streamed call is missing id, perform a quick non-stream retry to recover full tool_calls with ids
				const missingId =
					!nonStreamCalls && Object.values(toolCallState).some((c) => c?.name && !c?.id);
				let calls: NormalizedToolCall[];
				if (nonStreamCalls) {
					calls = nonStreamCalls;
				} else if (missingId) {
					console.debug(
						{ loop },
						"[mcp] missing tool_call id in stream; retrying non-stream to recover ids"
					);
					const nonStream = await openai.chat.completions.create(
						{ ...completionBase, messages: messagesOpenAI, stream: false },
						{
							signal: abortSignal,
							headers: {
								"ChatUI-Conversation-ID": conv._id.toString(),
								"X-use-cache": "false",
								...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
							},
						}
					);
					const tc: ChatCompletionMessageToolCall[] = Array.isArray(
						nonStream.choices?.[0]?.message?.tool_calls
					)
						? ((nonStream.choices?.[0]?.message?.tool_calls ?? []) as ChatCompletionMessageToolCall[])
						: [];
					calls = tc.map((t) => ({
						id: t.id,
						name: t.function?.name ?? "",
						arguments: t.function?.arguments ?? "",
					}));
				} else {
					calls = Object.values(toolCallState)
						.map((c) => (c?.id && c?.name ? c : undefined))
						.filter(Boolean)
						.map((c) => ({
							id: c?.id ?? "",
							name: c?.name ?? "",
							arguments: c?.arguments ?? "",
						})) as NormalizedToolCall[];
				}

				// Include the assistant message with tool_calls so the next round
				// sees both the calls and their outputs, matching MCP branch behavior.
				const toolCalls: ChatCompletionMessageToolCall[] = calls.map((call) => ({
					id: call.id,
					type: "function",
					function: { name: call.name, arguments: call.arguments },
				}));

				// Avoid sending <think> content back to the model alongside tool_calls
				// to prevent confusing follow-up reasoning. Strip any think blocks.
				const thinkMatch = lastAssistantContent.match(/<think>([\s\S]*?)(?:<\/think>|$)/);
				const reasoningContent = thinkMatch?.[1] ?? "";
				const assistantContentForToolMsg = lastAssistantContent.replace(
					/<think>[\s\S]*?(?:<\/think>|$)/g,
					""
				);

				const isDeepseekReasoner = /deepseek-reasoner/i.test(
					String(targetModel.id ?? targetModel.name)
				);

				// DeepSeek thinking-mode tool-calls require reasoning_content to be present
				// in the assistant message that contains tool_calls.
				const assistantToolMessage = (
					isDeepseekReasoner
						? ({
								role: "assistant",
								content: assistantContentForToolMsg,
								tool_calls: toolCalls,
								reasoning_content: reasoningContent,
							} as unknown as ChatCompletionMessageParam)
						: ({
								role: "assistant",
								content: assistantContentForToolMsg,
								tool_calls: toolCalls,
							} as ChatCompletionMessageParam)
				);

				const maxToolOutputCharsForLlm = (() => {
					const raw = config.MCP_MAX_TOOL_OUTPUT_CHARS_FOR_LLM;
					if (typeof raw !== "string" || raw.trim().length === 0) return undefined;
					const n = Number(raw);
					return Number.isFinite(n) ? n : undefined;
				})();

				const exec = executeToolCalls({
					calls,
					mapping,
					servers,
					parseArgs,
					resolveFileRef,
					toPrimitive,
					processToolOutput,
					maxToolOutputCharsForLlm,
					abortSignal,
				});
				let toolMsgCount = 0;
				let toolRunCount = 0;
				for await (const event of exec) {
					if (event.type === "update") {
						yield event.update;
					} else {
						if (isDeepseekChat || isDeepseekReasoner) {
							const toolRuns = event.summary.toolRuns ?? [];
							const MAX_TOOL_OUTPUT_CHARS_PER_TOOL = 4000;
							const MAX_TOOL_OUTPUT_CHARS_TOTAL = 8000;
							const truncate = (value: unknown, max: number) => {
								const s = value == null ? "" : typeof value === "string" ? value : String(value);
								if (s.length <= max) return s;
								return `${s.slice(0, max)}\n...[truncated ${s.length - max} chars]`;
							};
							const toolText = toolRuns
								.map((r) => {
									const params = (() => {
										try {
											return JSON.stringify(r.parameters ?? {}, null, 2);
										} catch {
											return "";
										}
									})();
									const outputStr = (() => {
										const out = (r as unknown as { output?: unknown })?.output;
										if (typeof out === "string") return out;
										try {
											return JSON.stringify(out ?? "", null, 2);
										} catch {
											return String(out ?? "");
										}
									})();
									const output = truncate(outputStr, MAX_TOOL_OUTPUT_CHARS_PER_TOOL);
									return (
										`[tool:${r.name}]` +
										(params ? `\nparams:\n${truncate(params, 1000)}` : "") +
										`\noutput:\n${output}`
									);
								})
								.join("\n\n");
							const toolTextLimited = truncate(toolText, MAX_TOOL_OUTPUT_CHARS_TOTAL);
							messagesOpenAI = [
								...messagesOpenAI,
								{ role: "assistant", content: assistantContentForToolMsg },
								{
									role: "assistant",
									content:
										(toolTextLimited && toolTextLimited.length > 0
											? `Результаты инструментов:\n\n${toolTextLimited}`
											: "Результаты инструментов: (пусто)"),
								},
								{
									role: "user",
									content:
										"Используй результаты инструментов выше и дай окончательный ответ пользователю. Не вызывай инструменты повторно.",
								},
							];
						} else {
							messagesOpenAI = [
								...messagesOpenAI,
								assistantToolMessage,
								...(event.summary.toolMessages ?? []),
							];
						}
						toolMsgCount = event.summary.toolMessages?.length ?? 0;
						toolRunCount = event.summary.toolRuns?.length ?? 0;
						console.info(
							{ toolMsgCount, toolRunCount },
							"[mcp] tools executed; continuing loop for follow-up completion"
						);
					}
				}
				// Continue loop: next iteration will use tool messages to get the final content
				continue;
			}

			// No tool calls: finalize and return
			// If a <think> block is still open, close it for the final output
			if (thinkOpen) {
				lastAssistantContent += "</think>";
			}
			if (!streamedContent && lastAssistantContent.trim().length > 0) {
				yield { type: MessageUpdateType.Stream, token: lastAssistantContent };
			}
			yield {
				type: MessageUpdateType.FinalAnswer,
				text: lastAssistantContent,
				interrupted: false,
			};
			console.info(
				{ length: lastAssistantContent.length, loop },
				"[mcp] final answer emitted (no tool_calls)"
			);
			return true;
		}
		console.warn("[mcp] exceeded tool-followup loops; falling back");
	} catch (err) {
		const msg = String(err ?? "");
		const isAbort =
			(abortSignal && abortSignal.aborted) ||
			msg.includes("AbortError") ||
			msg.includes("APIUserAbortError") ||
			msg.includes("Request was aborted");
		if (isAbort) {
			// Expected on user stop; keep logs quiet and do not treat as error
			console.debug("[mcp] aborted by user");
			return false;
		}
		console.warn({ err: msg }, "[mcp] flow failed, falling back to default endpoint");
	} finally {
		// ensure MCP clients are closed after the turn
		await drainPool();
	}

	return false;
}
