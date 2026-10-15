import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import type { Evaluation, Score } from "./ModelPerformance";

const champions = ["Ahri", "Ashe", "Annie", "Bard", "Braum", "Caitlyn", "Camille", "Darius", "Diana", "Ezreal"];
const catalog = {
  teams: [
    { id: "alpha", name: "Team Alpha", leagues: ["LCK"], games: 20 },
    { id: "beta", name: "Team Beta", leagues: ["LCK"], games: 20 },
    { id: "gamma", name: "Team Gamma", leagues: ["LCK"], games: 20 },
  ],
  leagues: ["LCK"], champions, games: 30, latest_patch: "15.24", data_end: "2025-12-29",
};
const score: Score = { games: 20, accuracy: .6, roc_auc: .65, log_loss: .65, brier_score: .23, ece: .03 };
const candidates = {
  logistic_regression: score, logistic_regression_calibrated: score,
  random_forest: score, random_forest_calibrated: score,
};
const diagnostic = {
  calibration: [{ games: 20, predicted: .6, observed: .6 }],
  roc: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }], monthly: [{ month: "2025-09", ...score }],
};
const evaluation: Evaluation = {
  method: "Chronological test", test_caveat: "Historical comparison", windows: {},
  selection: {
    pre_draft: { selected: "logistic_regression", validation_scores: candidates },
    post_draft: { selected: "random_forest", validation_scores: candidates },
  },
  partitions: { test: {
    pre_draft: score, post_draft: score, pre_draft_on_complete: score, constant_baseline: score,
    start: "2025-09-01", end: "2025-12-29",
    candidates: { pre_draft: candidates, post_draft: candidates },
    diagnostics: { pre_draft: diagnostic, post_draft: diagnostic },
  } },
};
let performanceFails = false;
let predictionFails = false;
let predictionHangs = false;
let reducedMotion = false;
const scroll = vi.fn();
const fetchMock = vi.fn<(url: RequestInfo | URL, init?: RequestInit) => Promise<Response>>();

function jsonResponse(value: unknown, ok = true): Response {
  return { ok, json: async () => value } as Response;
}

async function ready() {
  render(<App />);
  await screen.findByRole("button", { name: "Get pre-draft odds" });
}

function fillDraft() {
  const blue = screen.getByRole("group", { name: "Blue · Team Alpha" });
  const red = screen.getByRole("group", { name: "Red · Team Beta" });
  for (const [index, role] of ["Top", "Jungle", "Mid", "Bot", "Support"].entries()) {
    fireEvent.change(within(blue).getByRole("combobox", { name: role }), { target: { value: champions[index] } });
    fireEvent.change(within(red).getByRole("combobox", { name: role }), { target: { value: champions[index + 5] } });
  }
}

beforeEach(() => {
  performanceFails = predictionFails = predictionHangs = reducedMotion = false;
  scroll.mockReset();
  fetchMock.mockReset();
  HTMLElement.prototype.scrollIntoView = scroll;
  vi.stubGlobal("matchMedia", vi.fn(() => ({ matches: reducedMotion })));
  fetchMock.mockImplementation(async (url, init) => {
    if (String(url).endsWith("/catalog")) return jsonResponse(catalog);
    if (String(url).endsWith("/evaluation")) {
      if (performanceFails) throw new TypeError("Offline");
      return jsonResponse(evaluation);
    }
    if (predictionFails) return jsonResponse({ detail: "Please choose a valid draft." }, false);
    if (predictionHangs) return new Promise((_, reject) => {
      init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    });
    const request = JSON.parse(String(init?.body));
    const history = { games: 20, wins: 12, elo: 1530, form: .6 };
    return jsonResponse({
      blue_team: catalog.teams.find((team) => team.id === request.blue_team_id)?.name,
      red_team: catalog.teams.find((team) => team.id === request.red_team_id)?.name,
      pre_draft: { blue: .6, red: .4 }, post_draft: request.blue_draft ? { blue: .65, red: .35 } : null,
      blue_history: history, red_history: history, warnings: [], draft_warnings: [],
    });
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("desktop prediction flow", () => {
  it("keeps predictions usable when charts fail, and retries charts without resetting picks", async () => {
    performanceFails = true;
    await ready();
    await screen.findByText(/You can still make predictions/);
    const top = within(screen.getByRole("group", { name: "Blue · Team Alpha" })).getByRole("combobox", { name: "Top" });
    fireEvent.change(top, { target: { value: "Ahri" } });
    fireEvent.click(screen.getByRole("button", { name: "Get pre-draft odds" }));
    await screen.findByText(/Pre-draft estimate/);
    performanceFails = false;
    fireEvent.click(screen.getByRole("button", { name: "Retry performance details" }));
    await screen.findByText("Model performance");
    expect((top as HTMLSelectElement).value).toBe("Ahri");
  });

  it("swaps teams and their drafted champions together", async () => {
    await ready();
    const blue = screen.getByRole("group", { name: "Blue · Team Alpha" });
    const red = screen.getByRole("group", { name: "Red · Team Beta" });
    fireEvent.change(within(blue).getByRole("combobox", { name: "Top" }), { target: { value: "Ahri" } });
    fireEvent.change(within(red).getByRole("combobox", { name: "Top" }), { target: { value: "Camille" } });
    fireEvent.click(screen.getByRole("button", { name: "Swap sides" }));
    expect((screen.getByLabelText("BLUE SIDE") as HTMLSelectElement).value).toBe("beta");
    expect((screen.getByLabelText("RED SIDE") as HTMLSelectElement).value).toBe("alpha");
    expect((within(screen.getByRole("group", { name: "Blue · Team Beta" })).getByRole("combobox", { name: "Top" }) as HTMLSelectElement).value).toBe("Camille");
    expect((within(screen.getByRole("group", { name: "Red · Team Alpha" })).getByRole("combobox", { name: "Top" }) as HTMLSelectElement).value).toBe("Ahri");
  });

  it("explains missing picks and prevents duplicate selections", async () => {
    await ready();
    const update = screen.getByRole("button", { name: "Update with completed draft" }) as HTMLButtonElement;
    expect(update.disabled).toBe(true);
    expect(screen.getByText(/10 picks remaining:/)).toBeTruthy();
    const blue = screen.getByRole("group", { name: "Blue · Team Alpha" });
    const red = screen.getByRole("group", { name: "Red · Team Beta" });
    fireEvent.change(within(blue).getByRole("combobox", { name: "Top" }), { target: { value: "Ahri" } });
    const redTop = within(red).getByRole("combobox", { name: "Top" });
    expect((within(redTop).getByRole("option", { name: "Ahri" }) as HTMLOptionElement).disabled).toBe(true);
    expect(screen.getByRole("button", { name: "Get pre-draft odds" }).textContent).toBe("Get pre-draft odds");
    expect(update.textContent).toBe("Update with completed draft");
  });

  it.each([false, true])("focuses the completed estimate and respects reduced motion (%s)", async (reduced) => {
    reducedMotion = reduced;
    await ready();
    fillDraft();
    fireEvent.click(screen.getByRole("button", { name: "Update with completed draft" }));
    await screen.findByText(/Completed-draft estimate/);
    await waitFor(() => expect(document.activeElement?.id).toBe("result-title"));
    expect(scroll).toHaveBeenCalledWith({ behavior: reduced ? "auto" : "smooth", block: "start" });
    expect(screen.getByText("Team Alpha draft")).toBeTruthy();
    expect(screen.getByRole("link", { name: "Edit draft" }).getAttribute("href")).toBe("#draft-title");
    const blue = screen.getByRole("group", { name: "Blue · Team Alpha" });
    fireEvent.change(within(blue).getByRole("combobox", { name: "Top" }), { target: { value: "" } });
    expect(screen.queryByText(/Completed-draft estimate/)).toBeNull();
    expect(screen.getByText(/Pre-draft estimate/)).toBeTruthy();
    expect(screen.getByText("1 pick remaining: Blue top.")).toBeTruthy();
  });

  it("focuses a prediction error beside the submitted form without jumping to the estimate", async () => {
    predictionFails = true;
    await ready();
    fillDraft();
    fireEvent.click(screen.getByRole("button", { name: "Update with completed draft" }));
    const alert = await screen.findByRole("alert");
    await waitFor(() => expect(document.activeElement).toBe(alert));
    expect(alert.closest("form")?.getAttribute("aria-labelledby")).toBe("draft-title");
    expect(scroll).not.toHaveBeenCalled();
  });

  it("recovers from a stalled prediction instead of leaving the form disabled", async () => {
    await ready();
    predictionHangs = true;
    vi.useFakeTimers();
    fireEvent.click(screen.getByRole("button", { name: "Get pre-draft odds" }));
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(screen.getByRole("alert").textContent).toContain("took too long");
    expect((screen.getByRole("button", { name: "Get pre-draft odds" }) as HTMLButtonElement).disabled).toBe(false);
  });
});
