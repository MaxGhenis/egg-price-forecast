"use client";

interface ComponentWaterfallProps {
  components: Record<string, number>;
  total: number;
}

const LABELS: Record<string, { label: string; description: string }> = {
  intercept: {
    label: "Baseline",
    description: "Model intercept (prior expectation)",
  },
  seasonal: {
    label: "Seasonal pattern",
    description: "Historical rate of increase for this month",
  },
  streak: {
    label: "Decline streak",
    description: "Consecutive months of decline before target",
  },
  momentum: {
    label: "Recent momentum",
    description: "Average MoM change over prior 3 months",
  },
  cpi_signal: {
    label: "CPI expectations",
    description: "1-year expected inflation rate",
  },
  egg_specific: {
    label: "Egg-specific",
    description: "Egg intercept adjustment vs other commodities",
  },
  // Legacy labels for backward compatibility
  base_rate: {
    label: "Seasonal base",
    description: "Historical Feb increase rate",
  },
  streak_adj: {
    label: "Streak rebound",
    description: "Longer decline → higher rebound odds",
  },
  analog_boost: {
    label: "HPAI analog",
    description: "Similarity to 2023 recovery pattern",
  },
  wholesale_boost: {
    label: "Wholesale signal",
    description: "Wholesale-retail gap pressure",
  },
};

function fallbackLabel(key: string) {
  return {
    label: key.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()),
    description: "",
  };
}

export function ComponentWaterfall({
  components,
  total,
}: ComponentWaterfallProps) {
  const entries = Object.entries(components) as [string, number][];
  let cumulative = 0;
  // For logit-space contributions, use absolute sum for bar scaling
  const absSum = entries.reduce((s, [, v]) => s + Math.abs(v), 0) || 1;

  return (
    <div className="space-y-1">
      <h3
        className="mb-4 text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
        style={{ fontFamily: "var(--font-data)" }}
      >
        Component breakdown
      </h3>

      {entries.map(([key, value]) => {
        const meta = LABELS[key] || fallbackLabel(key);
        const isFirst = cumulative === 0 && key === entries[0][0];
        const isInactive = Math.abs(value) < 0.001;
        cumulative += value;
        const isPositive = value >= 0;

        return (
          <div
            key={key}
            className="group grid grid-cols-[1fr_60px] items-center gap-3"
            data-inactive={isInactive || undefined}
          >
            <div className="space-y-1">
              <div className="flex items-baseline justify-between">
                <span
                  className="text-sm font-medium"
                  style={{
                    color: isInactive
                      ? "var(--color-terminal-muted)"
                      : "var(--color-terminal-text)",
                  }}
                >
                  {meta.label}
                </span>
                <span
                  className="text-xs"
                  style={{
                    fontFamily: "var(--font-data)",
                    color: isInactive
                      ? "var(--color-terminal-muted)"
                      : isPositive
                        ? "var(--color-up)"
                        : "var(--color-down)",
                  }}
                >
                  {isPositive ? "+" : ""}{value.toFixed(2)}
                </span>
              </div>

              {/* Bar */}
              <div className="relative h-2 overflow-hidden rounded-full bg-[var(--color-terminal-border)]">
                <div
                  className="absolute inset-y-0 rounded-full transition-all duration-500"
                  style={{
                    left: isPositive ? "50%" : `${50 - (Math.abs(value) / absSum) * 50}%`,
                    width: `${(Math.abs(value) / absSum) * 50}%`,
                    backgroundColor: isInactive
                      ? "transparent"
                      : isPositive
                        ? "var(--color-up)"
                        : "var(--color-down)",
                  }}
                />
              </div>

              <p className="text-[10px] text-[var(--color-terminal-muted)]">
                {meta.description}
              </p>
            </div>

            {/* Cumulative */}
            <div
              className="text-right text-[10px] text-[var(--color-terminal-muted)]"
              style={{ fontFamily: "var(--font-data)" }}
            >
              &Sigma; {cumulative.toFixed(2)}
            </div>
          </div>
        );
      })}

      {/* Total */}
      <div className="mt-3 flex items-baseline justify-between border-t border-[var(--color-terminal-border)] pt-3">
        <span className="text-sm font-semibold text-[var(--color-terminal-bright)]">
          Total
        </span>
        <span
          className="text-lg font-medium text-[var(--color-yolk)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          {(total * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  );
}
