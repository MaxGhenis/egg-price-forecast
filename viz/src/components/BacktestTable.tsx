"use client";

interface BacktestTableProps {
  backtest: {
    accuracy: number;
    brier_score: number;
    predictions: Array<{
      year: number;
      predicted_prob: number;
      actual_up: boolean;
      correct: boolean;
    }>;
  };
}

export function BacktestTable({ backtest }: BacktestTableProps) {
  // Show last 15 years
  const recent = backtest.predictions.slice(-15);

  return (
    <div className="space-y-4">
      <h3
        className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
        style={{ fontFamily: "var(--font-data)" }}
      >
        Backtest results
      </h3>

      {/* Summary stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-3">
          <p className="text-[10px] uppercase text-[var(--color-terminal-muted)]">Accuracy</p>
          <p
            className="text-2xl font-light text-[var(--color-yolk)]"
            style={{ fontFamily: "var(--font-data)" }}
          >
            {(backtest.accuracy * 100).toFixed(1)}%
          </p>
        </div>
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-3">
          <p className="text-[10px] uppercase text-[var(--color-terminal-muted)]">Brier score</p>
          <p
            className="text-2xl font-light text-[var(--color-terminal-text)]"
            style={{ fontFamily: "var(--font-data)" }}
          >
            {backtest.brier_score.toFixed(3)}
          </p>
        </div>
      </div>

      {/* Dot matrix: each year is a cell */}
      <div className="flex flex-wrap gap-1.5">
        {recent.map((p) => (
          <div
            key={p.year}
            data-correct={p.correct ? "true" : "false"}
            className="group relative flex h-9 w-9 flex-col items-center justify-center rounded"
            style={{
              backgroundColor: p.correct
                ? "rgba(52, 211, 153, 0.12)"
                : "rgba(248, 113, 113, 0.12)",
              border: `1px solid ${p.correct ? "rgba(52, 211, 153, 0.3)" : "rgba(248, 113, 113, 0.3)"}`,
            }}
          >
            <span
              className="text-[9px] font-medium"
              style={{
                fontFamily: "var(--font-data)",
                color: p.correct ? "var(--color-up)" : "var(--color-down)",
              }}
            >
              {String(p.year).slice(2)}
            </span>
            <span
              className="text-[7px]"
              style={{
                fontFamily: "var(--font-data)",
                color: "var(--color-terminal-muted)",
              }}
            >
              {p.actual_up ? "↑" : "↓"}
            </span>

            {/* Tooltip */}
            <div className="pointer-events-none absolute -top-14 left-1/2 z-10 hidden -translate-x-1/2 rounded border border-[var(--color-terminal-border)] bg-[var(--color-terminal-bg)] px-2 py-1 text-[10px] whitespace-nowrap group-hover:block"
                 style={{ fontFamily: "var(--font-data)" }}>
              <div>{p.year}: {(p.predicted_prob * 100).toFixed(0)}% pred</div>
              <div>Actual: {p.actual_up ? "UP" : "DOWN"} {p.correct ? "✓" : "✗"}</div>
            </div>
          </div>
        ))}
      </div>

      <p className="text-[10px] text-[var(--color-terminal-muted)]">
        Each cell = one February. Green = correct prediction, red = wrong. Arrow shows actual direction.
      </p>
    </div>
  );
}
