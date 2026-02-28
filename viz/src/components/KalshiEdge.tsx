"use client";

interface KalshiEdgeProps {
  kalshi: {
    ticker: string;
    market_yes_price: number;
    model_estimate: number;
    edge: number;
    kelly_fraction: number;
    recommendation: string;
  };
}

export function KalshiEdge({ kalshi }: KalshiEdgeProps) {
  const isPositive = kalshi.edge > 0;
  const recSlug = kalshi.recommendation.includes("YES")
    ? "buy-yes"
    : kalshi.recommendation.includes("NO")
      ? "buy-no"
      : "neutral";

  return (
    <div data-recommendation={recSlug} className="space-y-4">
      <div className="flex items-center justify-between">
        <h3
          className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          Kalshi edge
        </h3>
        <span
          className="rounded px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider"
          style={{
            fontFamily: "var(--font-data)",
            backgroundColor: isPositive
              ? "rgba(52, 211, 153, 0.15)"
              : "rgba(248, 113, 113, 0.15)",
            color: isPositive ? "var(--color-up)" : "var(--color-down)",
          }}
        >
          {kalshi.recommendation}
        </span>
      </div>

      {/* Ticker */}
      <a
        href="https://kalshi.com/markets/kxeggs/egg-prices"
        target="_blank"
        rel="noopener noreferrer"
        className="block text-lg font-medium text-[var(--color-terminal-bright)] underline decoration-[var(--color-terminal-border)] underline-offset-4 transition-colors hover:decoration-[var(--color-yolk)]"
        style={{ fontFamily: "var(--font-data)" }}
      >
        {kalshi.ticker}
      </a>

      {/* Comparison bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-[var(--color-terminal-muted)]">Market YES</span>
          <span style={{ fontFamily: "var(--font-data)", color: "var(--color-terminal-text)" }}>
            {(kalshi.market_yes_price * 100).toFixed(0)}¢
          </span>
        </div>
        <div className="relative h-6 overflow-hidden rounded bg-[var(--color-terminal-border)]">
          {/* Market bar */}
          <div
            className="absolute inset-y-0 left-0 rounded bg-[var(--color-terminal-muted)]"
            style={{ width: `${kalshi.market_yes_price * 100}%`, opacity: 0.4 }}
          />
          {/* Model bar */}
          <div
            className="absolute inset-y-0 left-0 rounded"
            style={{
              width: `${kalshi.model_estimate * 100}%`,
              backgroundColor: isPositive ? "var(--color-up)" : "var(--color-down)",
              opacity: 0.6,
            }}
          />
          {/* Edge zone */}
          {isPositive && (
            <div
              className="absolute inset-y-0 rounded-r"
              style={{
                left: `${kalshi.market_yes_price * 100}%`,
                width: `${kalshi.edge * 100}%`,
                background: "repeating-linear-gradient(45deg, transparent, transparent 3px, rgba(52,211,153,0.2) 3px, rgba(52,211,153,0.2) 6px)",
              }}
            />
          )}
        </div>
        <div className="flex items-center justify-between text-xs">
          <span className="text-[var(--color-terminal-muted)]">Model</span>
          <span style={{ fontFamily: "var(--font-data)", color: isPositive ? "var(--color-up)" : "var(--color-down)" }}>
            {(kalshi.model_estimate * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-3 border-t border-[var(--color-terminal-border)] pt-3">
        <div>
          <p className="text-[10px] uppercase text-[var(--color-terminal-muted)]">Edge</p>
          <p
            className="text-xl font-light"
            style={{
              fontFamily: "var(--font-data)",
              color: isPositive ? "var(--color-up)" : "var(--color-down)",
            }}
          >
            {isPositive ? "+" : ""}{(kalshi.edge * 100).toFixed(1)}%
          </p>
        </div>
        <div>
          <p className="text-[10px] uppercase text-[var(--color-terminal-muted)]">Kelly</p>
          <p
            className="text-xl font-light text-[var(--color-terminal-text)]"
            style={{ fontFamily: "var(--font-data)" }}
          >
            {(kalshi.kelly_fraction * 100).toFixed(1)}%
          </p>
        </div>
      </div>
    </div>
  );
}
