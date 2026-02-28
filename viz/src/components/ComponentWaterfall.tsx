"use client";

interface ComponentWaterfallProps {
  components: Record<string, number>;
  total: number;
}

const LABELS: Record<string, { label: string; description: string }> = {
  lag_1: {
    label: "Last month price",
    description: "Most recent monthly price (strongest signal)",
  },
  lag_2: {
    label: "2-month lag",
    description: "Price two months ago",
  },
  lag_3: {
    label: "3-month lag",
    description: "Price three months ago",
  },
  lag_12: {
    label: "Same month last year",
    description: "Year-over-year seasonal comparison",
  },
  pct_change_1m: {
    label: "1-month momentum",
    description: "Month-over-month percentage change",
  },
  pct_change_3m: {
    label: "3-month momentum",
    description: "Percentage change over 3 months",
  },
  pct_change_12m: {
    label: "12-month momentum",
    description: "Year-over-year percentage change",
  },
  price_level: {
    label: "Price level",
    description: "Current absolute price (scale-dependent dynamics)",
  },
  month_sin: {
    label: "Seasonal (sin)",
    description: "Cyclic month encoding — sine component",
  },
  month_cos: {
    label: "Seasonal (cos)",
    description: "Cyclic month encoding — cosine component",
  },
  cpi_expectation: {
    label: "CPI expectations",
    description: "1-year inflation expectations from Cleveland Fed",
  },
  cross_commodity_momentum: {
    label: "Cross-commodity",
    description: "Average momentum across other food commodities",
  },
  is_eggs: {
    label: "Egg-specific",
    description: "Egg-specific dynamics (supply shocks, HPAI, etc.)",
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
  const absSum = entries.reduce((s, [, v]) => s + Math.abs(v), 0) || 1;

  return (
    <div className="space-y-1">
      <h3
        className="mb-4 text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
        style={{ fontFamily: "var(--font-data)" }}
      >
        Feature importance
      </h3>

      {entries.map(([key, value]) => {
        const meta = LABELS[key] || fallbackLabel(key);
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
          P(up)
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
