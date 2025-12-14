import { Client } from "@modelcontextprotocol/sdk/client";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import type { McpServerConfig } from "./httpClient";
// use console.* for lightweight diagnostics in production logs

export type OpenAiTool = {
	type: "function";
	function: { name: string; description?: string; parameters?: Record<string, unknown> };
};

export interface McpToolMapping {
	fnName: string;
	server: string;
	tool: string;
}

interface CacheEntry {
	fetchedAt: number;
	ttlMs: number;
	tools: OpenAiTool[];
	mapping: Record<string, McpToolMapping>;
}

const DEFAULT_TTL_MS = 60_000;
const cache = new Map<string, CacheEntry>();

// Per OpenAI tool/function name guidelines most providers enforce:
//   ^[a-zA-Z0-9_-]{1,64}$
// Dots are not universally accepted (e.g., MiniMax via HF router rejects them).
// Normalize any disallowed characters (including ".") to underscore and trim to 64 chars.
function sanitizeName(name: string) {
	return name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function inferJsonSchemaType(schema: Record<string, unknown>): string | undefined {
	const explicit = schema["type"];
	if (typeof explicit === "string") return explicit;
	if (Array.isArray(explicit)) return undefined;

	if (isRecord(schema["properties"]) || Array.isArray(schema["required"])) return "object";
	if (schema["items"] !== undefined) return "array";

	if (schema["enum"] !== undefined && Array.isArray(schema["enum"])) {
		const first = (schema["enum"] as unknown[])[0];
		if (first === null) return "null";
		if (Array.isArray(first)) return "array";
		const t = typeof first;
		if (t === "string" || t === "number" || t === "boolean") return t;
		if (t === "object") return "object";
	}

	if (schema["const"] !== undefined) {
		const v = schema["const"];
		if (v === null) return "null";
		if (Array.isArray(v)) return "array";
		const t = typeof v;
		if (t === "string" || t === "number" || t === "boolean") return t;
		if (t === "object") return "object";
	}

	if (typeof schema["pattern"] === "string") return "string";
	if (typeof schema["minLength"] === "number" || typeof schema["maxLength"] === "number") return "string";

	if (
		typeof schema["minimum"] === "number" ||
		typeof schema["maximum"] === "number" ||
		typeof schema["exclusiveMinimum"] === "number" ||
		typeof schema["exclusiveMaximum"] === "number" ||
		typeof schema["multipleOf"] === "number"
	) {
		return "number";
	}

	return undefined;
}

function sanitizeJsonSchemaNode(node: unknown): unknown {
	if (Array.isArray(node)) {
		return node.map((v) => sanitizeJsonSchemaNode(v));
	}
	if (!isRecord(node)) return node;

	const out: Record<string, unknown> = { ...node };
	if (out["type"] === "integer") {
		out["type"] = "number";
	}
	if (out["type"] === "null") {
		out["type"] = "string";
	}

	const oneOf = out["oneOf"];
	const anyOf = out["anyOf"];
	if (Array.isArray(oneOf)) {
		const mapped = oneOf.map((v) => sanitizeJsonSchemaNode(v));
		if (Array.isArray(anyOf)) {
			out["anyOf"] = [...anyOf.map((v) => sanitizeJsonSchemaNode(v)), ...mapped];
		} else {
			out["anyOf"] = mapped;
		}
		delete out["oneOf"];
	}

	const typeVal = out["type"];
	if (Array.isArray(typeVal)) {
		const { type: _t, ...rest } = out;
		out["anyOf"] = (typeVal as unknown[])
			.filter((t) => typeof t === "string")
			.map((t) => sanitizeJsonSchemaNode({ ...rest, type: t }));
		delete out["type"];
	}

	if (isRecord(out["properties"])) {
		const props = out["properties"] as Record<string, unknown>;
		const next: Record<string, unknown> = {};
		for (const [k, v] of Object.entries(props)) {
			next[k] = sanitizeJsonSchemaNode(v);
		}
		out["properties"] = next;
	}

	if (out["items"] !== undefined) {
		out["items"] = sanitizeJsonSchemaNode(out["items"]);
	}

	if (isRecord(out["additionalProperties"])) {
		out["additionalProperties"] = sanitizeJsonSchemaNode(out["additionalProperties"]);
	}

	for (const key of ["anyOf", "allOf", "not", "if", "then", "else"]) {
		const v = out[key];
		if (Array.isArray(v)) {
			const mapped = v.map((x) => sanitizeJsonSchemaNode(x));
			if (key === "anyOf") {
				out[key] = mapped.filter((x) => !(isRecord(x) && x["type"] === "null"));
			} else {
				out[key] = mapped;
			}
		} else if (isRecord(v)) {
			out[key] = sanitizeJsonSchemaNode(v);
		}
	}

	if (!out["type"] && !out["anyOf"]) {
		const inferred = inferJsonSchemaType(out);
		if (inferred) {
			out["type"] = inferred;
		} else {
			out["anyOf"] = [
				{ type: "string" },
				{ type: "number" },
				{ type: "boolean" },
				{ type: "object" },
				{ type: "array" },
			].map((x) => sanitizeJsonSchemaNode(x));
		}
	}

	return out;
}

function sanitizeOpenAiParametersSchema(schema: Record<string, unknown>): Record<string, unknown> {
	const sanitized = sanitizeJsonSchemaNode(schema);
	if (isRecord(sanitized)) {
		if (!sanitized["type"] && !sanitized["anyOf"]) {
			sanitized["type"] = "object";
		}
		if (sanitized["type"] !== "object" && !sanitized["properties"]) {
			return { type: "object", properties: {}, additionalProperties: true };
		}
		if (sanitized["type"] === "object" && !sanitized["properties"]) {
			sanitized["properties"] = {};
		}
		return sanitized;
	}
	return { type: "object", properties: {}, additionalProperties: true };
}

function buildCacheKey(servers: McpServerConfig[]): string {
	const normalized = servers
		.map((server) => ({
			name: server.name,
			url: server.url,
			headers: server.headers
				? Object.entries(server.headers)
						.sort(([a], [b]) => a.localeCompare(b))
						.map(([key, value]) => [key, value])
				: [],
		}))
		.sort((a, b) => {
			const byName = a.name.localeCompare(b.name);
			if (byName !== 0) return byName;
			return a.url.localeCompare(b.url);
		});

	return JSON.stringify(normalized);
}

type ListedTool = {
	name?: string;
	inputSchema?: Record<string, unknown>;
	description?: string;
	annotations?: { title?: string };
};

const LIST_TOOLS_TIMEOUT_MS = 30_000;

async function listServerTools(
	server: McpServerConfig,
	opts: { signal?: AbortSignal } = {}
): Promise<ListedTool[]> {
	const startedAt = Date.now();
	const abortController = new AbortController();
	let timeout: ReturnType<typeof setTimeout> | null = null;
	let abortHandler: (() => void) | null = null;
	if (opts.signal) {
		abortHandler = () => abortController.abort();
		if (opts.signal.aborted) {
			abortController.abort();
		} else {
			opts.signal.addEventListener("abort", abortHandler, { once: true });
		}
	}
	timeout = setTimeout(() => abortController.abort(), LIST_TOOLS_TIMEOUT_MS);

	const url = new URL(server.url);
	const client = new Client({ name: "chat-ui-mcp", version: "0.1.0" });
	try {
		try {
			console.debug(
				{ server: server.name, url: server.url, timeoutMs: LIST_TOOLS_TIMEOUT_MS },
				"[mcp] listTools: starting"
			);
		} catch {}

		try {
			const transport = new StreamableHTTPClientTransport(url, {
				requestInit: { headers: server.headers, signal: abortController.signal },
			});
			await client.connect(transport);
			try {
				console.debug(
					{ server: server.name, url: server.url, durationMs: Date.now() - startedAt },
					"[mcp] listTools: connected via streamable-http"
				);
			} catch {}
		} catch {
			const transport = new SSEClientTransport(url, {
				requestInit: { headers: server.headers, signal: abortController.signal },
			});
			await client.connect(transport);
			try {
				console.debug(
					{ server: server.name, url: server.url, durationMs: Date.now() - startedAt },
					"[mcp] listTools: connected via sse"
				);
			} catch {}
		}

		const response = await client.listTools({});
		const tools = Array.isArray(response?.tools) ? (response.tools as ListedTool[]) : [];
		try {
			console.debug(
				{
					server: server.name,
					url: server.url,
					durationMs: Date.now() - startedAt,
					count: tools.length,
					toolNames: tools.map((t) => t?.name).filter(Boolean),
				},
				"[mcp] listed tools from server"
			);
		} catch {}
		return tools;
	} catch (err) {
		const msg = String(err ?? "");
		try {
			console.warn(
				{
					server: server.name,
					url: server.url,
					durationMs: Date.now() - startedAt,
					aborted: abortController.signal.aborted,
					err: msg,
				},
				"[mcp] listTools failed"
			);
		} catch {}
		return [];
	} finally {
		if (timeout) clearTimeout(timeout);
		if (opts.signal && abortHandler) {
			try {
				opts.signal.removeEventListener("abort", abortHandler);
			} catch {}
		}
		try {
			await client.close?.();
		} catch {
			// ignore close errors
		}
	}
}

export async function getOpenAiToolsForMcp(
	servers: McpServerConfig[],
	{ ttlMs = DEFAULT_TTL_MS, signal }: { ttlMs?: number; signal?: AbortSignal } = {}
): Promise<{ tools: OpenAiTool[]; mapping: Record<string, McpToolMapping> }> {
	const now = Date.now();
	const cacheKey = buildCacheKey(servers);
	const cached = cache.get(cacheKey);
	if (cached && now - cached.fetchedAt < cached.ttlMs) {
		return { tools: cached.tools, mapping: cached.mapping };
	}

	const tools: OpenAiTool[] = [];
	const mapping: Record<string, McpToolMapping> = {};

	const seenNames = new Set<string>();

	const pushToolDefinition = (
		name: string,
		description: string | undefined,
		parameters: Record<string, unknown> | undefined
	) => {
		if (seenNames.has(name)) return;
		tools.push({
			type: "function",
			function: {
				name,
				description,
				parameters,
			},
		});
		seenNames.add(name);
	};

	// Fetch tools in parallel; tolerate individual failures
	const tasks = servers.map((server) => listServerTools(server, { signal }));
	const results = await Promise.allSettled(tasks);

	for (let i = 0; i < results.length; i++) {
		const server = servers[i];
		const r = results[i];
		if (r.status === "fulfilled") {
			const serverTools = r.value;
			for (const tool of serverTools) {
				if (typeof tool.name !== "string" || tool.name.trim().length === 0) {
					continue;
				}

				const parameters =
					tool.inputSchema && typeof tool.inputSchema === "object"
						? sanitizeOpenAiParametersSchema(tool.inputSchema as Record<string, unknown>)
						: undefined;
				const description = tool.description ?? tool.annotations?.title;
				const toolName = tool.name;

				// Emit a collision-aware function name.
				// Prefer the plain tool name; on conflict, suffix with server name.
				let plainName = sanitizeName(toolName);
				if (plainName in mapping) {
					const suffix = sanitizeName(server.name);
					const candidate = `${plainName}_${suffix}`.slice(0, 64);
					if (!(candidate in mapping)) {
						plainName = candidate;
					} else {
						let i = 2;
						let next = `${candidate}_${i}`;
						while (i < 10 && next in mapping) {
							i += 1;
							next = `${candidate}_${i}`;
						}
						plainName = next.slice(0, 64);
					}
				}

				pushToolDefinition(plainName, description, parameters);
				mapping[plainName] = {
					fnName: plainName,
					server: server.name,
					tool: toolName,
				};
			}
		} else {
			// ignore failure for this server
			continue;
		}
	}

	cache.set(cacheKey, { fetchedAt: now, ttlMs, tools, mapping });
	return { tools, mapping };
}

export function resetMcpToolsCache() {
	cache.clear();
}
