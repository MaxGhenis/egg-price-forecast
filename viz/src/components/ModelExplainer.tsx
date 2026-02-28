"use client";

interface ModelExplainerProps {
  declineStreak: number;
  seasonalRate: number;
  recentMomentum: number;
  modelInfo?: {
    n_obs: number;
    n_commodities: number;
    coefficients: Record<string, number>;
  };
}

export function ModelExplainer({
  declineStreak,
  seasonalRate,
  recentMomentum,
  modelInfo,
}: ModelExplainerProps) {
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
            This is a <strong>Bayesian logistic regression</strong> trained on
            month-over-month price direction changes across{" "}
            <strong>{modelInfo?.n_commodities ?? 15} BLS food commodity</strong> series
            (eggs, beef, chicken, milk, bread, flour, and more). By pooling data
            across commodities and all calendar months, the model trains on{" "}
            <strong>{modelInfo?.n_obs?.toLocaleString() ?? "~6,000"} observations</strong>{" "}
            rather than just ~46 egg-only Februaries. Coefficients are estimated via
            MAP (maximum a posteriori) with mildly regularizing normal priors, and
            an egg-specific intercept captures egg-specific dynamics.
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
            Model features
          </p>
          <p className="mb-3">
            Five features are computed per commodity per month. At prediction time,
            egg-specific values are used:
          </p>

          <div className="space-y-3">
            {/* 1. Seasonal rate */}
            <div className="rounded border-l-2 border-[var(--color-yolk)] bg-[var(--color-terminal-surface)] p-3">
              <p className="text-sm font-medium text-[var(--color-terminal-bright)]">
                1. Seasonal up rate
                <span className="ml-2 text-xs text-[var(--color-yolk)]" style={{ fontFamily: "var(--font-data)" }}>
                  = {(seasonalRate * 100).toFixed(1)}%
                </span>
              </p>
              <p className="mt-1 text-xs text-[var(--color-terminal-muted)]">
                Historical fraction of times this calendar month saw a price
                increase vs the preceding month. Captures pure seasonal patterns.
              </p>
            </div>

            {/* 2. Decline streak */}
            <div className="rounded border-l-2 border-[var(--color-up)] bg-[var(--color-terminal-surface)] p-3">
              <p className="text-sm font-medium text-[var(--color-terminal-bright)]">
                2. Decline streak
                <span className="ml-2 text-xs text-[var(--color-up)]" style={{ fontFamily: "var(--font-data)" }}>
                  = {declineStreak} months
                </span>
              </p>
              <p className="mt-1 text-xs text-[var(--color-terminal-muted)]">
                Consecutive months of price declines leading into the target month.
                Longer streaks suggest mean-reversion pressure.
              </p>
            </div>

            {/* 3. Recent momentum */}
            <div className="rounded border-l-2 border-[var(--color-up)] bg-[var(--color-terminal-surface)] p-3">
              <p className="text-sm font-medium text-[var(--color-terminal-bright)]">
                3. Recent momentum
                <span className="ml-2 text-xs text-[var(--color-terminal-text)]" style={{ fontFamily: "var(--font-data)" }}>
                  = {(recentMomentum * 100).toFixed(1)}%
                </span>
              </p>
              <p className="mt-1 text-xs text-[var(--color-terminal-muted)]">
                Average month-over-month percentage change over the prior 3 months.
                Captures whether prices have been trending up or down recently.
              </p>
            </div>

            {/* 4. CPI expectations */}
            <div className="rounded border-l-2 border-[var(--color-terminal-muted)] bg-[var(--color-terminal-surface)] p-3">
              <p className="text-sm font-medium text-[var(--color-terminal-bright)]">
                4. CPI inflation expectations
              </p>
              <p className="mt-1 text-xs text-[var(--color-terminal-muted)]">
                1-year expected inflation rate from the Cleveland Fed Nowcast (FRED:
                EXPINF1YR). Higher inflation expectations may correlate with upward
                food price pressure.
              </p>
            </div>

            {/* 5. Egg-specific */}
            <div className="rounded border-l-2 border-[var(--color-terminal-muted)] bg-[var(--color-terminal-surface)] p-3">
              <p className="text-sm font-medium text-[var(--color-terminal-bright)]">
                5. Egg-specific intercept
              </p>
              <p className="mt-1 text-xs text-[var(--color-terminal-muted)]">
                A binary indicator that allows the model to learn egg-specific
                dynamics (e.g., supply shocks from HPAI outbreaks) separate from
                the general food commodity baseline.
              </p>
            </div>
          </div>
        </div>

        {/* Combination */}
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-[var(--color-yolk)]"
             style={{ fontFamily: "var(--font-data)" }}>
            How it combines
          </p>
          <p>
            Features are combined via logistic regression: each feature is multiplied
            by a learned coefficient, summed with an intercept, and passed through
            the logistic sigmoid to produce a probability. The output is a single
            probability — no confidence intervals, since a probability already
            encodes the model&apos;s uncertainty about the outcome.
          </p>
          <p className="my-2 rounded bg-[var(--color-terminal-bg)] px-3 py-2 text-xs text-[var(--color-yolk)]"
             style={{ fontFamily: "var(--font-data)" }}>
            P(up) = sigmoid(intercept + &beta;&#8321; &middot; seasonal + &beta;&#8322; &middot; streak + &beta;&#8323; &middot; momentum + &beta;&#8324; &middot; cpi + &beta;&#8325; &middot; is_eggs)
          </p>
        </div>

        {/* Limitations */}
        <div className="rounded-lg border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-4">
          <p className="mb-1 text-xs font-medium uppercase tracking-wider text-[var(--color-down)]"
             style={{ fontFamily: "var(--font-data)" }}>
            Limitations
          </p>
          <p>
            The model treats price direction as binary (up vs not up), discarding
            magnitude information. Training on pooled commodities assumes some
            shared dynamics across food prices, which may not always hold. The
            model does not account for supply shocks, trade policy, or input
            costs beyond what&apos;s captured by CPI expectations. Use the edge
            estimate directionally, not as a precise fair value.
          </p>
        </div>
      </div>
    </div>
  );
}
