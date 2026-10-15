import { useId, useState } from "react";

export type Score = {
  games: number; accuracy: number; roc_auc: number | null;
  log_loss: number; brier_score: number; ece: number;
};
type Phase = "pre_draft" | "post_draft";
type Diagnostics = {
  calibration: { games: number; predicted: number; observed: number }[];
  roc: { fpr: number; tpr: number }[];
  monthly: (Score & { month: string })[];
};
export type Evaluation = {
  method: string; test_caveat: string;
  selection: Record<Phase, { selected: string; validation_scores: Record<string, Score> }>;
  windows: Record<string, { games: number; start: string; end: string }>;
  partitions: { test: {
    pre_draft: Score; post_draft: Score; pre_draft_on_complete: Score;
    constant_baseline: Score; start: string; end: string;
    candidates: Record<Phase, Record<string, Score>>;
    diagnostics: Record<Phase, Diagnostics>;
  } };
};
const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const modelNames: Record<string, string> = {
  logistic_regression: "Logistic Regression",
  logistic_regression_calibrated: "Logistic Regression · calibrated",
  random_forest: "Random Forest",
  random_forest_calibrated: "Random Forest · calibrated",
};
type Point = { x: number; y: number; label?: string };

function ResultChart({ title, xLabel, yLabel, points, diagonal = false, months }: {
  title: string; xLabel: string; yLabel: string; points: Point[]; diagonal?: boolean; months?: string[];
}) {
  const id = useId();
  const x = (value: number) => 44 + value * 252;
  const y = (value: number) => 174 - value * 148;
  const line = points.map((point) => `${x(point.x).toFixed(2)},${y(point.y).toFixed(2)}`).join(" ");
  return <figure className="result-chart">
    <figcaption>{title}</figcaption>
    <svg viewBox="0 0 320 222" role="img" aria-labelledby={`${id}-title ${id}-description`}>
      <title id={`${id}-title`}>{title}</title>
      <desc id={`${id}-description`}>{xLabel} on the horizontal axis; {yLabel} on the vertical axis. Values come from the chronological test set.{diagonal ? " The dashed diagonal is the reference line." : " The dashed line marks 50%."}</desc>
      {[0, .5, 1].map((tick) => <g key={tick}>
        <line x1="44" x2="296" y1={y(tick)} y2={y(tick)} className="chart-grid" />
        <text x="36" y={y(tick) + 4} textAnchor="end">{Math.round(tick * 100)}%</text>
      </g>)}
      {diagonal ? <line x1="44" x2="296" y1="174" y2="26" className="chart-reference" /> : <line x1="44" x2="296" y1={y(.5)} y2={y(.5)} className="chart-reference" />}
      <line x1="44" x2="44" y1="26" y2="174" className="chart-axis" />
      <line x1="44" x2="296" y1="174" y2="174" className="chart-axis" />
      {months ? months.map((month, index) => <text key={month} x={x(months.length === 1 ? .5 : index / (months.length - 1))} y="191" textAnchor="middle">{month.slice(5)}</text>) : [0, .5, 1].map((tick) => <text key={tick} x={x(tick)} y="191" textAnchor="middle">{Math.round(tick * 100)}%</text>)}
      <polyline points={line} className="chart-series" />
      {points.length <= 12 && points.map((point, index) => <circle key={index} cx={x(point.x)} cy={y(point.y)} r="3.5" className="chart-point"><title>{point.label ?? `${percent(point.x)} predicted, ${percent(point.y)} observed`}</title></circle>)}
      <text x="170" y="213" textAnchor="middle">{xLabel}</text>
      <text x="12" y="100" textAnchor="middle" transform="rotate(-90 12 100)">{yLabel}</text>
    </svg>
  </figure>;
}

export function ModelPerformance({ evaluation }: { evaluation: Evaluation }) {
  const [phase, setPhase] = useState<Phase>("pre_draft");
  const test = evaluation.partitions.test;
  if (!evaluation.selection || !test.candidates) return null;
  const selected = evaluation.selection[phase].selected;
  const diagnostic = test.diagnostics[phase];
  const score = test[phase];
  return <details className="panel performance-panel">
    <summary>
      <span><span className="eyebrow">MEASURED, NOT ASSUMED</span><strong>Model performance</strong></span>
      <span className="performance-summary">Pre-draft {percent(test.pre_draft.accuracy)} accuracy · AUC {test.pre_draft.roc_auc?.toFixed(3) ?? "—"}</span>
    </summary>
    <div className="performance-content">
      <div className="section-heading"><div><h2>Compare the models</h2><p className="muted">Test matches: {test.start.slice(0, 10)} to {test.end.slice(0, 10)}. All leagues combined.</p></div>
        <label className="competition">Prediction phase<select value={phase} onChange={(event) => setPhase(event.target.value as Phase)}><option value="pre_draft">Pre-draft</option><option value="post_draft">Completed draft</option></select></label>
      </div>
      <p className="muted">Selected on validation log loss: <b>{modelNames[selected]}</b>. Test scores did not choose the winner. These are earlier-trained evaluation models; the matchup sandbox uses separate serving refits.</p>
      <div className="table-scroll"><table>
        <caption className="sr-only">Model comparison for the selected phase on identical test games. Lower log loss, Brier score, and calibration error are better.</caption>
        <thead><tr><th>Model</th><th>Games</th><th>Accuracy</th><th>ROC AUC</th><th>Log loss ↓</th><th>Brier ↓</th><th>ECE ↓</th></tr></thead>
        <tbody>{Object.entries(test.candidates[phase]).map(([name, metric]) => <tr key={name} className={selected === name ? "selected-model" : undefined}>
          <th scope="row">{modelNames[name]}{selected === name && <span className="selection-tag">Selected</span>}</th><td>{metric.games.toLocaleString()}</td><td>{percent(metric.accuracy)}</td><td>{metric.roc_auc?.toFixed(3) ?? "—"}</td><td>{metric.log_loss.toFixed(3)}</td><td>{metric.brier_score.toFixed(3)}</td><td>{metric.ece.toFixed(3)}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="scope-note">ROC AUC measures ranking across thresholds. Log loss and Brier score assess probability error. ECE compares predicted and observed blue-side win rates in ten probability bins; it depends on binning and sample size.</p>
      <div className="performance-charts">
        <ResultChart title={`ROC curve · AUC ${score.roc_auc?.toFixed(3) ?? "—"}`} xLabel="False positive rate" yLabel="True positive rate" diagonal points={diagnostic.roc.map((point) => ({ x: point.fpr, y: point.tpr }))} />
        <ResultChart title="Probability reliability" xLabel="Predicted blue win rate" yLabel="Observed blue win rate" diagonal points={diagnostic.calibration.map((point) => ({ x: point.predicted, y: point.observed, label: `${point.games} games: ${percent(point.predicted)} predicted, ${percent(point.observed)} observed` }))} />
        <ResultChart title="Accuracy through the test period" xLabel="Month (2025)" yLabel="Accuracy" months={diagnostic.monthly.map((point) => point.month)} points={diagnostic.monthly.map((point, index, all) => ({ x: all.length === 1 ? .5 : index / (all.length - 1), y: point.accuracy, label: `${point.month}: ${percent(point.accuracy)} accuracy over ${point.games} games` }))} />
      </div>
      <details className="chart-data"><summary>View monthly scores and sample sizes</summary><div className="table-scroll"><table><thead><tr><th>Month</th><th>Games</th><th>Accuracy</th><th>AUC</th><th>Log loss</th></tr></thead><tbody>{diagnostic.monthly.map((month) => <tr key={month.month}><th scope="row">{month.month}</th><td>{month.games}</td><td>{percent(month.accuracy)}</td><td>{month.roc_auc?.toFixed(3) ?? "—"}</td><td>{month.log_loss.toFixed(3)}</td></tr>)}</tbody></table></div></details>
      <div className="evaluation-notes">
        <p>Constant blue-side baseline: {percent(test.constant_baseline.accuracy)} accuracy, AUC {test.constant_baseline.roc_auc?.toFixed(3) ?? "—"}.</p>
        <p>On the same {test.post_draft.games.toLocaleString()} complete-draft games: pre-draft log loss {test.pre_draft_on_complete.log_loss.toFixed(3)}, completed-draft log loss {test.post_draft.log_loss.toFixed(3)}. Lower is better.</p>
        <p>{evaluation.method}</p><p>{evaluation.test_caveat}</p>
      </div>
    </div>
  </details>;
}
