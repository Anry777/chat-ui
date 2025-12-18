import { z } from "zod";
import * as fs from "fs";
import { openAICompletionToTextGenerationStream } from "./openAICompletionToTextGenerationStream";
import {
	openAIChatToTextGenerationSingle,
	openAIChatToTextGenerationStream,
} from "./openAIChatToTextGenerationStream";
import type { CompletionCreateParamsStreaming } from "openai/resources/completions";
import type {
	ChatCompletionCreateParamsNonStreaming,
	ChatCompletionCreateParamsStreaming,
} from "openai/resources/chat/completions";
import { buildPrompt } from "$lib/buildPrompt";
import { config } from "$lib/server/config";
import type { Endpoint } from "../endpoints";
import type OpenAI from "openai";
import { createImageProcessorOptionsValidator, makeImageProcessor } from "../images";
import { prepareMessagesWithFiles } from "$lib/server/textGeneration/utils/prepareFiles";
// uuid import removed (no tool call ids)

export const endpointOAIParametersSchema = z.object({
	weight: z.number().int().positive().default(1),
	model: z.any(),
	type: z.literal("openai"),
	baseURL: z.string().url().default("https://api.openai.com/v1"),
	// Canonical auth token is OPENAI_API_KEY; keep HF_TOKEN as legacy alias
	apiKey: z.string().default(config.OPENAI_API_KEY || config.HF_TOKEN || "sk-"),
	completion: z
		.union([z.literal("completions"), z.literal("chat_completions")])
		.default("chat_completions"),
	defaultHeaders: z.record(z.string()).optional(),
	defaultQuery: z.record(z.string()).optional(),
	extraBody: z.record(z.any()).optional(),
	multimodal: z
		.object({
			image: createImageProcessorOptionsValidator({
				supportedMimeTypes: [
					// Restrict to the most widely-supported formats
					"image/png",
					"image/jpeg",
				],
				preferredMimeType: "image/jpeg",
				maxSizeInMB: 1,
				maxWidth: 1024,
				maxHeight: 1024,
			}),
		})
		.default({}),
	/* enable use of max_completion_tokens in place of max_tokens */
	useCompletionTokens: z.boolean().default(false),
	streamingSupported: z.boolean().default(true),
});

export async function endpointOai(
	input: z.input<typeof endpointOAIParametersSchema>
): Promise<Endpoint> {
	const {
		baseURL,
		apiKey,
		completion,
		model,
		defaultHeaders,
		defaultQuery,
		multimodal,
		extraBody,
		useCompletionTokens,
		streamingSupported,
	} = endpointOAIParametersSchema.parse(input);

	let OpenAI;
	try {
		OpenAI = (await import("openai")).OpenAI;
	} catch (e) {
		throw new Error("Failed to import OpenAI", { cause: e });
	}

	// Store router metadata if captured
	let routerMetadata: { route?: string; model?: string; provider?: string } = {};

	// Custom fetch wrapper to capture response headers for router metadata
	const customFetch = async (url: RequestInfo, init?: RequestInit): Promise<Response> => {
		const response = await fetch(url, init);

		// Capture router headers if present (fallback for non-streaming)
		const routeHeader = response.headers.get("X-Router-Route");
		const modelHeader = response.headers.get("X-Router-Model");
		const providerHeader = response.headers.get("x-inference-provider");

		if (routeHeader && modelHeader) {
			routerMetadata = {
				route: routeHeader,
				model: modelHeader,
				provider: providerHeader || undefined,
			};
		} else if (providerHeader) {
			// Even without router metadata, capture provider info
			routerMetadata = {
				provider: providerHeader,
			};
		}

		return response;
	};

	const openai = new OpenAI({
		apiKey: apiKey || "sk-",
		baseURL,
		defaultHeaders: {
			...(config.PUBLIC_APP_NAME === "HuggingChat" && { "User-Agent": "huggingchat" }),
			...defaultHeaders,
		},
		defaultQuery,
		fetch: customFetch,
	});

	const imageProcessor = makeImageProcessor(multimodal.image);

	if (completion === "completions") {
		return async ({
			messages,
			preprompt,
			generateSettings,
			conversationId,
			locals,
			abortSignal,
		}) => {
			const prompt = await buildPrompt({
				messages,
				preprompt,
				model,
			});

			const parameters = { ...model.parameters, ...generateSettings };
			const body: CompletionCreateParamsStreaming = {
				model: model.id ?? model.name,
				prompt,
				stream: true,
				max_tokens: parameters?.max_tokens,
				stop: parameters?.stop,
				temperature: parameters?.temperature,
				top_p: parameters?.top_p,
				frequency_penalty: parameters?.frequency_penalty,
				presence_penalty: parameters?.presence_penalty,
			};

			const openAICompletion = await openai.completions.create(body, {
				body: { ...body, ...extraBody },
				headers: {
					"ChatUI-Conversation-ID": conversationId?.toString() ?? "",
					"X-use-cache": "false",
					...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
					// Bill to organization if configured (HuggingChat only)
					...(config.isHuggingChat && locals?.billingOrganization
						? { "X-HF-Bill-To": locals.billingOrganization }
						: {}),
				},
				signal: abortSignal,
			});

			return openAICompletionToTextGenerationStream(openAICompletion);
		};
	} else if (completion === "chat_completions") {
		return async ({
			messages,
			preprompt,
			generateSettings,
			conversationId,
			isMultimodal,
			locals,
			abortSignal,
		}) => {
			// AGGRESSIVE DEBUG LOGGING
			const logPath = "/app/debug.log";
			const log = (msg: string) => {
				try {
					fs.appendFileSync(logPath, `[${new Date().toISOString()}] ${msg}\n`);
				} catch (e) {
					console.error("!!!!!!!! FAILED TO WRITE LOG !!!!!!!!", e);
					// Still ignore, but log to stderr
				}
			};

			try {
				log("DEBUG: Step 1 - Entered chat_completions handler.");

				// PREPARE MESSAGES
				log("DEBUG: Step 2 - Preparing messages with files.");
				let messagesOpenAI: OpenAI.Chat.Completions.ChatCompletionMessageParam[] =
					await prepareMessagesWithFiles(messages, imageProcessor, isMultimodal ?? model.multimodal);
				log("DEBUG: Step 3 - Finished preparing messages.");

				// NORMALIZE PREPROMPT
				log("DEBUG: Step 4 - Normalizing preprompt.");
				const normalizedPreprompt = typeof preprompt === "string" ? preprompt.trim() : "";

				const hasSystemMessage = messagesOpenAI.length > 0 && messagesOpenAI[0]?.role === "system";

				if (hasSystemMessage) {
					if (normalizedPreprompt) {
						const userSystemPrompt =
							(typeof messagesOpenAI[0].content === "string"
								? (messagesOpenAI[0].content as string)
								: "") || "";
						messagesOpenAI[0].content =
							normalizedPreprompt + (userSystemPrompt ? "\n\n" + userSystemPrompt : "");
					}
				} else {
					if (normalizedPreprompt) {
						messagesOpenAI = [
							{ role: "system", content: normalizedPreprompt },
							...messagesOpenAI,
						];
					}
				}
				log("DEBUG: Step 5 - Finished normalizing preprompt.");

				// PREPARE BODY
				log("DEBUG: Step 6 - Preparing request body.");
				const parameters = { ...model.parameters, ...generateSettings };
				const body = {
					model: model.id ?? model.name,
					messages: messagesOpenAI,
					stream: streamingSupported,
					...(useCompletionTokens
						? { max_completion_tokens: parameters?.max_tokens }
						: { max_tokens: parameters?.max_tokens }),
					stop: parameters?.stop,
					temperature: parameters?.temperature,
					top_p: parameters?.top_p,
					frequency_penalty: parameters?.frequency_penalty,
					presence_penalty: parameters?.presence_penalty,
				};
				log("DEBUG: Step 7 - Finished preparing request body.");

				// SEND REQUEST
				if (streamingSupported) {
					log("DEBUG: Step 8a - Sending STREAMING request.");
					const openChatAICompletion = await openai.chat.completions.create(
						body as ChatCompletionCreateParamsStreaming,
						{
							body: { ...body, ...extraBody },
							headers: {
								"ChatUI-Conversation-ID": conversationId?.toString() ?? "",
								"X-use-cache": "false",
								...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
								...(config.isHuggingChat && locals?.billingOrganization
									? { "X-HF-Bill-To": locals.billingOrganization }
									: {}),
							},
							signal: abortSignal,
						}
					);
					log("DEBUG: Step 9a - STREAMING request sent.");
					return openAIChatToTextGenerationStream(openChatAICompletion, () => routerMetadata);
				} else {
					log("DEBUG: Step 8b - Sending NON-STREAMING request.");
					const openChatAICompletion = await openai.chat.completions.create(
						body as ChatCompletionCreateParamsNonStreaming,
						{
							body: { ...body, ...extraBody },
							headers: {
								"ChatUI-Conversation-ID": conversationId?.toString() ?? "",
								"X-use-cache": "false",
								...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
								...(config.isHuggingChat && locals?.billingOrganization
									? { "X-HF-Bill-To": locals.billingOrganization }
									: {}),
							},
							signal: abortSignal,
						}
					);
					log("DEBUG: Step 9b - NON-STREAMING request sent.");
					return openAIChatToTextGenerationSingle(openChatAICompletion, () => routerMetadata);
				}
			} catch (e) {
				log(`ERROR: CRASH IN HANDLER! Error: ${e.message}\nStack: ${e.stack}`);
				throw e; // Re-throw the error to ensure the caller knows something went wrong
			}
		};
	} else {
		throw new Error("Invalid completion type");
	}
}
