import type {
  ChatSummary,
  FeishuSettingsUpdate,
  MemorySettingsResponse,
  MemorySettingsUpdate,
  MemorySettingsUpdateResponse,
  MemoryTypedListResponse,
  MemoryTypedRetireResponse,
  MemoryType,
  ProviderSettingsUpdate,
  SettingsPayload,
  SettingsUpdate,
  SlashCommand,
  WebSearchSettingsUpdate,
  WebuiThreadPersistedPayload,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(
  url: string,
  token: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(url, {
    ...(init ?? {}),
    headers: {
      ...(init?.headers ?? {}),
      Authorization: `Bearer ${token}`,
    },
    credentials: "same-origin",
  });
  if (!res.ok) {
    throw new ApiError(res.status, `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

function splitKey(key: string): { channel: string; chatId: string } {
  const idx = key.indexOf(":");
  if (idx === -1) return { channel: "", chatId: key };
  return { channel: key.slice(0, idx), chatId: key.slice(idx + 1) };
}

export async function listSessions(
  token: string,
  base: string = "",
): Promise<ChatSummary[]> {
  type Row = {
    key: string;
    created_at: string | null;
    updated_at: string | null;
    title?: string;
    preview?: string;
  };
  const body = await request<{ sessions: Row[] }>(
    `${base}/api/sessions`,
    token,
  );
  return body.sessions.map((s) => ({
    key: s.key,
    ...splitKey(s.key),
    createdAt: s.created_at,
    updatedAt: s.updated_at,
    title: s.title ?? "",
    preview: s.preview ?? "",
  }));
}

/** Disk-backed WebUI display thread snapshot (separate from agent session). */
export async function fetchWebuiThread(
  token: string,
  key: string,
  base: string = "",
): Promise<WebuiThreadPersistedPayload | null> {
  const url = `${base}/api/sessions/${encodeURIComponent(key)}/webui-thread`;
  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${token}` },
    credentials: "same-origin",
  });
  if (res.status === 404) return null;
  if (!res.ok) throw new ApiError(res.status, `HTTP ${res.status}`);
  return (await res.json()) as WebuiThreadPersistedPayload;
}

export async function deleteSession(
  token: string,
  key: string,
  base: string = "",
): Promise<boolean> {
  const body = await request<{ deleted: boolean }>(
    `${base}/api/sessions/${encodeURIComponent(key)}/delete`,
    token,
  );
  return body.deleted;
}

export async function fetchSettings(
  token: string,
  base: string = "",
): Promise<SettingsPayload> {
  return request<SettingsPayload>(`${base}/api/settings`, token);
}

export async function listSlashCommands(
  token: string,
  base: string = "",
): Promise<SlashCommand[]> {
  type Row = {
    command: string;
    title: string;
    description: string;
    icon: string;
    arg_hint?: string;
  };
  const body = await request<{ commands: Row[] }>(`${base}/api/commands`, token);
  return body.commands
    .filter((command) => !["/stop", "/restart"].includes(command.command))
    .map((command) => ({
      command: command.command,
      title: command.title,
      description: command.description,
      icon: command.icon,
      argHint: command.arg_hint ?? "",
    }));
}

export async function updateSettings(
  token: string,
  update: SettingsUpdate,
  base: string = "",
): Promise<SettingsPayload> {
  const query = new URLSearchParams();
  if (update.model !== undefined) query.set("model", update.model);
  if (update.provider !== undefined) query.set("provider", update.provider);
  return request<SettingsPayload>(`${base}/api/settings/update?${query}`, token);
}

export async function updateProviderSettings(
  token: string,
  update: ProviderSettingsUpdate,
  base: string = "",
): Promise<SettingsPayload> {
  const query = new URLSearchParams();
  query.set("provider", update.provider);
  if (update.apiKey !== undefined) query.set("api_key", update.apiKey);
  if (update.apiBase !== undefined) query.set("api_base", update.apiBase);
  return request<SettingsPayload>(
    `${base}/api/settings/provider/update?${query}`,
    token,
  );
}

export async function updateWebSearchSettings(
  token: string,
  update: WebSearchSettingsUpdate,
  base: string = "",
): Promise<SettingsPayload> {
  const query = new URLSearchParams();
  query.set("provider", update.provider);
  if (update.apiKey !== undefined) query.set("api_key", update.apiKey);
  if (update.baseUrl !== undefined) query.set("base_url", update.baseUrl);
  return request<SettingsPayload>(
    `${base}/api/settings/web-search/update?${query}`,
    token,
  );
}

export async function updateFeishuSettings(
  token: string,
  update: FeishuSettingsUpdate,
  base: string = "",
): Promise<SettingsPayload> {
  const query = new URLSearchParams();
  if (update.enabled !== undefined) query.set("enabled", String(update.enabled));
  if (update.appId !== undefined) query.set("app_id", update.appId);
  if (update.appSecret !== undefined) query.set("app_secret", update.appSecret);
  if (update.encryptKey !== undefined) query.set("encrypt_key", update.encryptKey);
  if (update.verificationToken !== undefined) {
    query.set("verification_token", update.verificationToken);
  }
  if (update.allowFrom !== undefined) {
    if (update.allowFrom.length === 0) {
      query.append("allow_from", "");
    } else {
      for (const value of update.allowFrom) query.append("allow_from", value);
    }
  }
  if (update.groupPolicy !== undefined) query.set("group_policy", update.groupPolicy);
  if (update.streaming !== undefined) query.set("streaming", String(update.streaming));
  if (update.domain !== undefined) query.set("domain", update.domain);
  return request<SettingsPayload>(
    `${base}/api/settings/feishu/update?${query}`,
    token,
  );
}

export async function fetchMemorySettings(
  token: string,
  base: string = "",
): Promise<MemorySettingsResponse> {
  return request<MemorySettingsResponse>(`${base}/api/settings/memory`, token);
}

export async function updateMemorySettings(
  token: string,
  update: MemorySettingsUpdate,
  base: string = "",
): Promise<MemorySettingsUpdateResponse> {
  const query = new URLSearchParams();
  if (update.enabled !== undefined) query.set("enabled", String(update.enabled));
  if (update.injectionMode !== undefined) query.set("injection_mode", update.injectionMode);
  if (update.retrievalLimit !== undefined) query.set("retrieval_limit", String(update.retrievalLimit));
  if (update.packetCharLimit !== undefined) query.set("packet_char_limit", String(update.packetCharLimit));
  if (update.dbPath !== undefined) query.set("db_path", update.dbPath ?? "");
  return request<MemorySettingsUpdateResponse>(
    `${base}/api/settings/memory/update?${query}`,
    token,
  );
}

export async function listMemoryTyped(
  token: string,
  opts: { memoryType?: MemoryType | ""; limit?: number; clientId?: string | null } = {},
  base: string = "",
): Promise<MemoryTypedListResponse> {
  const query = new URLSearchParams();
  if (opts.memoryType) query.set("memory_type", opts.memoryType);
  if (opts.limit !== undefined) query.set("limit", String(opts.limit));
  if (opts.clientId) query.set("client_id", opts.clientId);
  const suffix = query.toString();
  return request<MemoryTypedListResponse>(
    `${base}/api/memory/typed${suffix ? `?${suffix}` : ""}`,
    token,
  );
}

export async function retireMemoryTyped(
  token: string,
  id: string,
  opts: { status?: "inactive" | "deleted"; reason?: string } = {},
  base: string = "",
): Promise<MemoryTypedRetireResponse> {
  const query = new URLSearchParams();
  if (opts.status !== undefined) query.set("status", opts.status);
  if (opts.reason !== undefined) query.set("reason", opts.reason);
  return request<MemoryTypedRetireResponse>(
    `${base}/api/memory/typed/${encodeURIComponent(id)}/retire?${query}`,
    token,
  );
}
