import { useEffect, useRef, useState } from "react";
import { ModelPerformance } from "./ModelPerformance";
import type { Evaluation } from "./ModelPerformance";
import "./App.css";

type Team = { id: string; name: string; leagues: string[]; games: number };
type Catalog = { teams: Team[]; leagues: string[]; champions: string[]; games: number; data_end: string; latest_patch: string };
type History = { games: number; wins: number; elo: number; form: number };
type PredictionPhase = "pre_draft" | "post_draft";
type Prediction = {
  blue_team: string; red_team: string;
  pre_draft: { blue: number; red: number }; post_draft: { blue: number; red: number } | null;
  blue_history: History; red_history: History; warnings: string[]; draft_warnings: string[];
};
const roles = ["top", "jng", "mid", "bot", "sup"] as const;
const labels = { top: "Top", jng: "Jungle", mid: "Mid", bot: "Bot", sup: "Support" };
type Picks = Record<(typeof roles)[number], string>;
const emptyPicks = (): Picks => ({ top: "", jng: "", mid: "", bot: "", sup: "" });
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const api = import.meta.env.VITE_API_BASE_URL ?? "/api";

async function responseData(response: Response) {
  const data = await response.json();
  if (!response.ok) {
    throw new Error(typeof data.detail === "string" ? data.detail : "Please check the teams and completed drafts.");
  }
  return data;
}

function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [evaluation, setEvaluation] = useState<Evaluation | null>(null);
  const [league, setLeague] = useState("");
  const [blueTeam, setBlueTeam] = useState("");
  const [redTeam, setRedTeam] = useState("");
  const [bluePicks, setBluePicks] = useState<Picks>(emptyPicks);
  const [redPicks, setRedPicks] = useState<Picks>(emptyPicks);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [pendingPhase, setPendingPhase] = useState<PredictionPhase | null>(null);
  const [predictionError, setPredictionError] = useState<{ phase: PredictionPhase; message: string } | null>(null);
  const [performanceError, setPerformanceError] = useState("");
  const [performanceLoading, setPerformanceLoading] = useState(true);
  const [performanceRetry, setPerformanceRetry] = useState(0);
  const predictionErrorMessage = useRef<HTMLDivElement>(null);
  const estimateHeading = useRef<HTMLHeadingElement>(null);
  const showEstimateAfterPrediction = useRef(false);
  const busy = pendingPhase !== null;

  useEffect(() => {
    if (predictionError) predictionErrorMessage.current?.focus();
  }, [predictionError]);

  useEffect(() => {
    if (!showEstimateAfterPrediction.current || !prediction?.post_draft) return;
    showEstimateAfterPrediction.current = false;
    estimateHeading.current?.focus({ preventScroll: true });
    estimateHeading.current?.scrollIntoView({
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
      block: "start",
    });
  }, [prediction]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${api}/catalog`, { signal: controller.signal }).then(responseData).then((data: Catalog) => {
      setCatalog(data);
      const initialLeague = data.leagues.includes("LCK") ? "LCK" : data.leagues[0];
      const teams = data.teams.filter((team) => team.leagues.includes(initialLeague));
      setLeague(initialLeague);
      setBlueTeam(teams[0]?.id ?? "");
      setRedTeam(teams[1]?.id ?? "");
    }).catch((reason: Error) => {
      if (!controller.signal.aborted) setError(reason.message);
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${api}/evaluation`, { signal: controller.signal }).then(responseData).then((report: Evaluation) => {
      setEvaluation(report);
    }).catch(() => {
      if (!controller.signal.aborted) setPerformanceError("Model performance details could not load. You can still make predictions.");
    }).finally(() => {
      if (!controller.signal.aborted) setPerformanceLoading(false);
    });
    return () => controller.abort();
  }, [performanceRetry]);

  const teams = catalog?.teams.filter((team) => team.leagues.includes(league)) ?? [];
  const selected = [...Object.values(bluePicks), ...Object.values(redPicks)].filter(Boolean);
  const complete = selected.length === 10 && new Set(selected).size === 10;
  const missingPicks = (["blue", "red"] as const).flatMap((side) =>
    roles.filter((role) => !(side === "blue" ? bluePicks : redPicks)[role])
      .map((role) => `${side === "blue" ? "Blue" : "Red"} ${labels[role].toLowerCase()}`),
  );
  const draftUnderperforms = evaluation && evaluation.partitions.test.post_draft.log_loss > evaluation.partitions.test.pre_draft_on_complete.log_loss;

  function resetResult() { setPrediction(null); setPredictionError(null); }
  function clearDraftEstimate() {
    setPrediction((current) => current ? { ...current, post_draft: null, draft_warnings: [] } : null);
    setPredictionError(null);
  }
  function changeLeague(next: string) {
    const available = catalog?.teams.filter((team) => team.leagues.includes(next)) ?? [];
    setLeague(next);
    setBlueTeam(available[0]?.id ?? "");
    setRedTeam(available[1]?.id ?? "");
    resetResult();
  }

  function swapSides() {
    setBlueTeam(redTeam);
    setRedTeam(blueTeam);
    setBluePicks(redPicks);
    setRedPicks(bluePicks);
    resetResult();
  }

  async function predict(withDraft: boolean) {
    if (busy || !blueTeam || !redTeam || blueTeam === redTeam || (withDraft && !complete)) return;
    const phase = withDraft ? "post_draft" : "pre_draft";
    setPendingPhase(phase);
    setPredictionError(null);
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    try {
      const result = await fetch(`${api}/predict`, {
        method: "POST",
        signal: controller.signal,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          league, blue_team_id: blueTeam, red_team_id: redTeam,
          ...(withDraft ? { blue_draft: bluePicks, red_draft: redPicks } : {}),
        }),
      }).then(responseData);
      showEstimateAfterPrediction.current = withDraft;
      setPrediction(result);
    } catch (reason) {
      const message = controller.signal.aborted ? "The prediction took too long. Please try again."
        : reason instanceof TypeError ? "Could not reach the prediction service. Please try again."
        : reason instanceof Error ? reason.message : "Prediction failed. Please try again.";
      setPredictionError({ phase, message });
    } finally {
      window.clearTimeout(timeout);
      setPendingPhase(null);
    }
  }

  return (
    <main className="workspace">
      <header className="masthead">
        <a className="brand" href="/"><span className="brand-mark" aria-hidden="true">R</span> RiftPredict</a>
        <span className="badge">PRO MATCH LAB</span>
      </header>

      <section className="intro">
        <p className="eyebrow">TEAM HISTORY + CHAMPION DRAFTS</p>
        <h1>Every matchup starts<br />before the Rift.</h1>
        <p className="intro-copy">Compare professional teams, set both compositions, and see how the model's estimate changes.</p>
        {catalog && <p className="data-note">2025 historical sandbox · {catalog.games.toLocaleString()} games · {catalog.leagues.length} competitions · Patch {catalog.latest_patch} · Data through {catalog.data_end.slice(0, 10)}</p>}
      </section>

      {loading && <p role="status">Loading match data…</p>}
      {error && <div className="error" role="alert">{error}{!catalog && <button onClick={() => { setError(""); setLoading(true); setRetry(retry + 1); }}>Retry</button>}</div>}

      {catalog && <>
        <form className="matchup panel" aria-labelledby="matchup-title" aria-busy={pendingPhase === "pre_draft"} onSubmit={(event) => { event.preventDefault(); void predict(false); }}>
          <div className="section-heading"><div><p className="eyebrow">01 / THE MATCHUP</p><h2 id="matchup-title" tabIndex={-1}>Choose your teams</h2></div>
            <div className="matchup-controls">
              <button className="text-button" type="button" disabled={busy || !blueTeam || !redTeam} onClick={swapSides}>Swap sides</button>
            <label className="competition">Competition
              <select value={league} disabled={busy} onChange={(event) => changeLeague(event.target.value)}>
                {catalog.leagues.map((name) => <option key={name}>{name}</option>)}
              </select>
            </label>
            </div>
          </div>
          <div className="team-grid">
            {(["blue", "red"] as const).map((side) => {
              const value = side === "blue" ? blueTeam : redTeam;
              const opposite = side === "blue" ? redTeam : blueTeam;
              const team = teams.find((item) => item.id === value);
              return <div className={`team-card ${side}`} key={side}>
                <label htmlFor={`${side}-team`} className="side-label">{side.toUpperCase()} SIDE</label>
                <select id={`${side}-team`} value={value} disabled={busy} onChange={(event) => {
                  (side === "blue" ? setBlueTeam : setRedTeam)(event.target.value); resetResult();
                }}>
                  {teams.map((item) => <option key={item.id} value={item.id} disabled={item.id === opposite}>{item.name}</option>)}
                </select>
                <p>{team?.games ?? 0} recorded games across competitions</p>
              </div>;
            })}
          </div>
          {predictionError?.phase === "pre_draft" && <div className="error prediction-error" ref={predictionErrorMessage} tabIndex={-1} role="alert">{predictionError.message}</div>}
          <button className="primary" type="submit" disabled={busy || !blueTeam || !redTeam || blueTeam === redTeam}>{pendingPhase === "pre_draft" ? "Calculating pre-draft odds…" : "Get pre-draft odds"}</button>
        </form>

        <form className="panel draft-section" aria-labelledby="draft-title" aria-busy={pendingPhase === "post_draft"} onSubmit={(event) => { event.preventDefault(); void predict(true); }}>
          <div className="section-heading"><div><p className="eyebrow">02 / THE COMPOSITIONS</p><h2 id="draft-title" tabIndex={-1}>Complete both drafts</h2></div><span className="draft-count">{selected.length}/10 champions</span></div>
          {draftUnderperforms && <p className="warning">Experimental: the draft model currently has higher held-out probability error than the pre-draft model.</p>}<p className="muted">Assign one champion per role. Picks stay on the board when you compare teams.</p>
          <div className="team-grid">
            {(["blue", "red"] as const).map((side) => {
              const picks = side === "blue" ? bluePicks : redPicks;
              return <fieldset className={`draft-card ${side}`} key={side}>
                <legend>{side === "blue" ? "Blue" : "Red"} · {teams.find((team) => team.id === (side === "blue" ? blueTeam : redTeam))?.name ?? "Choose a team"}</legend>
                {roles.map((role) => <label className="pick-row" key={role}>
                  <span>{labels[role]}</span>
                  <select value={picks[role]} disabled={busy} onChange={(event) => {
                    (side === "blue" ? setBluePicks : setRedPicks)({ ...picks, [role]: event.target.value }); clearDraftEstimate();
                  }}>
                    <option value="">Select champion</option>
                    {catalog.champions.map((champion) => <option key={champion} disabled={selected.includes(champion) && picks[role] !== champion}>{champion}</option>)}
                  </select>
                </label>)}
              </fieldset>;
            })}
          </div>
          <p id="draft-progress" className="draft-progress" role="status">{complete
            ? "Both drafts are complete. Update the estimate when ready."
            : `${10 - selected.length} ${selected.length === 9 ? "pick" : "picks"} remaining: ${missingPicks.join(", ")}.`}</p>
          {predictionError?.phase === "post_draft" && <div className="error prediction-error" ref={predictionErrorMessage} tabIndex={-1} role="alert">{predictionError.message}</div>}
          <div className="draft-actions">
            <button className="primary" type="submit" aria-describedby="draft-progress" disabled={busy || !complete || !blueTeam || !redTeam || blueTeam === redTeam}>{pendingPhase === "post_draft" ? "Updating completed-draft estimate…" : "Update with completed draft"}</button>
            <button className="text-button" type="button" disabled={busy || selected.length === 0} onClick={() => { setBluePicks(emptyPicks()); setRedPicks(emptyPicks()); clearDraftEstimate(); }}>Clear picks</button>
          </div>
        </form>

        <section className="panel result-section" aria-labelledby="result-title" aria-live="polite">
          <p className="eyebrow">03 / THE ESTIMATE</p><h2 id="result-title" ref={estimateHeading} tabIndex={-1}>{prediction ? "The matchup, in probabilities." : "Build a matchup. Find an edge."}</h2>
          {prediction ? <>
            <div className="odds-grid">
              <div><span className="side-label blue-text">{prediction.blue_team}</span><strong>{percent((prediction.post_draft ?? prediction.pre_draft).blue)}</strong></div>
              <span className="versus">VS</span>
              <div className="red-odds"><span className="side-label red-text">{prediction.red_team}</span><strong>{percent((prediction.post_draft ?? prediction.pre_draft).red)}</strong></div>
            </div>
            <div className="probability-bar" role="img" aria-label={`Blue ${percent((prediction.post_draft ?? prediction.pre_draft).blue)}, red ${percent((prediction.post_draft ?? prediction.pre_draft).red)}`}><div style={{ width: percent((prediction.post_draft ?? prediction.pre_draft).blue) }} /></div>
            <p className="muted">{prediction.post_draft ? "Completed-draft estimate" : "Pre-draft estimate"} · Single-game winner</p>
            {prediction.post_draft && <p className="draft-change">Blue pre-draft: {percent(prediction.pre_draft.blue)} → after draft: {percent(prediction.post_draft.blue)}. These are separate models; the difference is not a causal measure of draft strength.</p>}
            <div className="history-grid">{(["blue", "red"] as const).map((side) => {
              const history = side === "blue" ? prediction.blue_history : prediction.red_history;
              return <p key={side}><b>{side === "blue" ? prediction.blue_team : prediction.red_team}</b><br />{history.wins}W – {history.games - history.wins}L · Elo {Math.round(history.elo)}</p>;
            })}</div>
            {prediction.post_draft && <div className="estimate-drafts">{(["blue", "red"] as const).map((side) => <div key={side}>
              <h3>{side === "blue" ? prediction.blue_team : prediction.red_team} draft</h3>
              <dl>{roles.map((role) => <div key={role}><dt>{labels[role]}</dt><dd>{(side === "blue" ? bluePicks : redPicks)[role]}</dd></div>)}</dl>
            </div>)}</div>}
            <nav className="estimate-actions" aria-label="Edit prediction"><a href="#matchup-title">Edit teams</a><a href="#draft-title">Edit draft</a></nav>
            {[...prediction.warnings, ...(prediction.post_draft ? prediction.draft_warnings : [])].map((warning) => <p className="warning" key={warning}>{warning}</p>)}
          </> : <p className="muted">Start with team history, then compare it with a completed champion draft.</p>}
          <p className="scope-note">Historical estimates use team history through the end of 2025. They do not account for current rosters, current patches, or upcoming schedules.</p>
        </section>

        {performanceLoading && <p className="muted" role="status">Loading model performance…</p>}
        {performanceError && <div className="panel performance-unavailable" role="status"><p>{performanceError}</p><button className="text-button" type="button" onClick={() => { setPerformanceError(""); setPerformanceLoading(true); setPerformanceRetry((value) => value + 1); }}>Retry performance details</button></div>}
        {evaluation && <ModelPerformance evaluation={evaluation} />}
      </>}
      <footer>RiftPredict <span>Professional League of Legends · Research in progress</span></footer>
    </main>
  );
}

export default App;
