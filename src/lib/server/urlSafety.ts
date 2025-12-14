// Shared server-side URL safety helper (exact behavior preserved)
import { env as serverEnv } from "$env/dynamic/private";

export function isValidUrl(urlString: string): boolean {
	try {
		const url = new URL(urlString.trim());
		// Only allow HTTPS protocol
		if (url.protocol !== "https:") {
			return false;
		}
		// Prevent localhost/private IPs (basic check)
		const hostname = url.hostname.toLowerCase();
		if (
			hostname === "localhost" ||
			hostname.startsWith("127.") ||
			hostname.startsWith("192.168.") ||
			hostname.startsWith("172.16.") ||
			hostname === "[::1]" ||
			hostname === "0.0.0.0"
		) {
			return false;
		}
		return true;
	} catch {
		return false;
	}
}

export function isValidMcpUrl(urlString: string): boolean {
	if (serverEnv.ALLOW_INSECURE_MCP === "true") {
		try {
			const url = new URL(urlString.trim());
			return url.protocol === "https:" || url.protocol === "http:";
		} catch {
			return false;
		}
	}

	return isValidUrl(urlString);
}
