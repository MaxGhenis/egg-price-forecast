"use client";

interface ModelExplainerProps {
  modelInfo: {
    type: string;
    n_quantile_models: number;
    n_samples: number;
    feature_importances: Record<string, number>;
  };
  backtest: {
    accuracy: number;
    direction_accuracy: number;
    mape: number;
    coverage_90: number;
    coverage_50: number;
    brier_score: number;
  };
}

export function ModelExplainer({
  modelInfo,
  backtest,
}: ModelExplainerProps) {
  const dirAcc = backtest.direction_accuracy;

  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-[var(--color-terminal-bright)]">
        How the model works
      </h3>

      <div className="space-y-5 text-sm leading-relaxed text-[var(--color-terminal-text)]">
        {/* What it is */}
        <div className="rounded-lg border border-[var(--color-yolk-dim)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-[var(--color-yolk)]"
             style={{ fontFamily: "var(--font-data)" }}>
            Approach
          </p>
          <p>
            This is a <strong>LightGBM multi-quantile regression</strong> trained on
            month-over-month price data across{" "}
            <strong>15 BLS food commodity</strong> series
            (eggs, beef, chicken, milk, bread, flour, rice, and more). Rather than
            predicting binary direction, the model forecasts <strong>actual price values</strong>{" "}
            at {modelInfo.n_quantile_models} quantile levels (5th, 10th, 25th, 50th, 75th, 90th, 95th), then derives
            P(price increase) from the simulated price distribution. This preserves magnitude
            information and provides prediction intervals.
          </p>
        </div>

        {/* Data source */}
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-[var(--color-yolk)]"
             style={{ fontFamily: "var(--font-data)" }}>
            Training data
          </p>
          <p>
            The model uses BLS Average Price series downloaded from{" "}
            <a href="https://fred.stlouisfed.org/series/APU0000708111"
               target="_blank" rel="noopener noreferrer"
               className="underline decoration-[var(--color-terminal-border)] underline-offset-2 hover:decoration-[var(--color-yolk)]">
              FRED
            </a>{" "}
            — 15 food commodities with monthly data going back to ~1980.
            Additionally, 1-year CPI inflation expectations from the Cleveland
            Fed Nowcast provide a macro signal. All data is publicly available
            and requires no API key.
          </p>
        </div>

        {/* Features */}
        <div>
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-[var(--color-terminal-muted)]"
             style={{ fontFamily: "var(--font-data)" }}>
            Features
          </p>
          <p className="mb-3">
            For each commodity and month, the model computes features capturing
            price dynamics at multiple timescales:
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            <FeatureItem
              color="var(--color-yolk)"
              label="Price lags (1-6, 12)"
              description="Recent price history and same-month-last-year for seasonal comparison"
            />
            <FeatureItem
              color="var(--color-yolk)"
              label="Momentum (1m, 3m, 12m)"
              description="Percentage changes over multiple windows capture trend and acceleration"
            />
            <FeatureItem
              color="var(--color-up)"
              label="Seasonal encoding"
              description="Sine/cosine month encoding for smooth cyclical patterns"
            />
            <FeatureItem
              color="var(--color-up)"
              label="CPI expectations"
              description="1-year inflation expectations from Cleveland Fed Nowcast"
            />
            <FeatureItem
              color="var(--color-terminal-muted)"
              label="Cross-commodity momentum"
              description="Average price momentum across other food commodities"
            />
            <FeatureItem
              color="var(--color-terminal-muted)"
              label="Egg indicator + price level"
              description="Egg-specific dynamics and absolute price scale"
            />
          </div>
        </div>

        {/* How P(up) is derived */}
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-[var(--color-yolk)]"
             style={{ fontFamily: "var(--font-data)" }}>
            From price forecast to P(up)
          </p>
          <p>
            {modelInfo.n_quantile_models} separate LightGBM models predict the 5th through 95th percentile
            of next month&apos;s price. A skew-normal distribution is fit to these
            quantile predictions, then {modelInfo.n_samples.toLocaleString()} price
            scenarios are sampled. <strong>P(up)</strong> is simply the fraction
            of simulated prices that exceed the current price. This approach captures
            both the expected direction and magnitude of price changes, and naturally
            produces prediction intervals.
          </p>
        </div>

        {/* Backtest performance */}
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-[var(--color-yolk)]"
             style={{ fontFamily: "var(--font-data)" }}>
            Backtest accuracy
          </p>
          <p className="mb-3">
            Expanding-window backtest starting from 2010 — at each month, the model
            trains only on prior data to simulate real-time forecasting:
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <MetricBox
              label="Direction"
              value={`${(dirAcc * 100).toFixed(0)}%`}
              description="Correct up/down calls"
            />
            <MetricBox
              label="MAPE"
              value={`${(backtest.mape * 100).toFixed(1)}%`}
              description="Mean absolute % error"
            />
            <MetricBox
              label="90% coverage"
              value={`${(backtest.coverage_90 * 100).toFixed(0)}%`}
              description="Actuals in 90% interval"
            />
            <MetricBox
              label="Brier"
              value={backtest.brier_score.toFixed(3)}
              description="Probability calibration"
            />
          </div>
        </div>

        {/* Limitations */}
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-[var(--color-down)]"
             style={{ fontFamily: "var(--font-data)" }}>
            Limitations
          </p>
          <p>
            The model pools data across food commodities, assuming partially shared
            price dynamics. While this increases training data, commodity-specific
            shocks (HPAI outbreaks, trade policy) may not be well-captured by
            historical patterns alone. Quantile crossing is corrected post-hoc
            but the distribution fit assumes a skew-normal shape. Use predictions
            directionally, not as precise fair values.
          </p>
        </div>
      </div>
    </div>
  );
}

function FeatureItem({ color, label, description }: { color: string; label: string; description: string }) {
  return (
    <div className="rounded border-l-2 bg-[var(--color-terminal-surface)] p-2.5" style={{ borderColor: color }}>
      <p className="text-xs font-medium text-[var(--color-terminal-bright)]">{label}</p>
      <p className="mt-0.5 text-[10px] text-[var(--color-terminal-muted)]">{description}</p>
    </div>
  );
}

function MetricBox({ label, value, description }: { label: string; value: string; description: string }) {
  return (
    <div className="rounded border border-[var(--color-terminal-border)] p-2 text-center">
      <p className="text-[10px] uppercase text-[var(--color-terminal-muted)]" style={{ fontFamily: "var(--font-data)" }}>
        {label}
      </p>
      <p className="text-lg font-light text-[var(--color-yolk)]" style={{ fontFamily: "var(--font-data)" }}>
        {value}
      </p>
      <p className="text-[9px] text-[var(--color-terminal-muted)]">{description}</p>
    </div>
  );
}
