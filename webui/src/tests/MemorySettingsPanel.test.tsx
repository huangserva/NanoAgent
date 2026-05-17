import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { MemorySettingsPanel } from "@/components/settings/SettingsView";
import {
  fetchMemorySettings,
  listMemoryTyped,
  retireMemoryTyped,
  updateMemorySettings,
} from "@/lib/api";

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    fetchMemorySettings: vi.fn(),
    listMemoryTyped: vi.fn(),
    retireMemoryTyped: vi.fn(),
    updateMemorySettings: vi.fn(),
  };
});

const memorySettings = {
  enabled: true,
  injectionMode: "both" as const,
  retrievalLimit: 3,
  packetCharLimit: 2000,
  dbPath: "/tmp/memory.db",
  supportedTypes: ["preference", "profile_fact", "task_state", "project_fact"],
  requiresRestart: false,
};

const typedMemory = {
  id: "mem-1",
  user_id: "client-a",
  memory_type: "preference" as const,
  text: "I prefer concise answers.",
  confidence: 0.91,
  evidence_event_id: "event-123456",
  provenance: null,
  scope: null,
  dedupe_key: null,
  created_at: "2026-05-01T00:00:00Z",
  updated_at: "2026-05-02T00:00:00Z",
  status: "active",
  retired_at: null,
  retired_reason: null,
  retired_evidence_event_id: null,
  superseded_by_id: null,
  deleted_at: null,
};

describe("MemorySettingsPanel", () => {
  beforeEach(() => {
    vi.mocked(fetchMemorySettings).mockResolvedValue({
      available: true,
      memory: memorySettings,
    });
    vi.mocked(listMemoryTyped).mockResolvedValue({
      available: true,
      memories: [typedMemory],
    });
    vi.mocked(retireMemoryTyped).mockResolvedValue({ memory: { ...typedMemory, status: "inactive" } });
    vi.mocked(updateMemorySettings).mockResolvedValue({
      available: true,
      memory: { ...memorySettings, retrievalLimit: 5 },
      requires_restart: false,
    });
    vi.spyOn(window, "confirm").mockReturnValue(true);
  });

  it("renders typed memories and retires one", async () => {
    render(<MemorySettingsPanel token="tok" clientId="client-a" />);

    expect(await screen.findByText("I prefer concise answers.")).toBeInTheDocument();
    expect(listMemoryTyped).toHaveBeenCalledWith("tok", {
      memoryType: "",
      limit: 50,
      clientId: "client-a",
    });

    fireEvent.click(screen.getByRole("button", { name: "Retire" }));

    await waitFor(() => {
      expect(retireMemoryTyped).toHaveBeenCalledWith("tok", "mem-1", {
        status: "inactive",
        reason: "retired from WebUI",
      });
    });
  });

  it("shows the not available branch", async () => {
    vi.mocked(fetchMemorySettings).mockResolvedValueOnce({
      available: false,
      memory: { ...memorySettings, enabled: false },
    });

    render(<MemorySettingsPanel token="tok" clientId="client-a" />);

    expect(await screen.findByText("External memory is disabled")).toBeInTheDocument();
    expect(screen.queryByText("I prefer concise answers.")).not.toBeInTheDocument();
  });
});
