"use client";

import type { ForecastData } from "@/lib/types";
import { BacktestTable } from "./BacktestTable";
import { ComponentWaterfall } from "./ComponentWaterfall";
import { KalshiEdge } from "./KalshiEdge";
import { ModelExplainer } from "./ModelExplainer";
import { PriceChart } from "./PriceChart";
import { ProbabilityGauge } from "./ProbabilityGauge";

interface DashboardProps {
  data: ForecastData;
}

export function Dashboard({ data }: DashboardProps) {
  return (
    <div className="mx-auto max-w-6xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <header className="mb-8 border-b border-[var(--color-terminal-border)] pb-6">
        <div className="flex items-end justify-between">
          <div>
            <p
              className="text-[10px] uppercase tracking-[0.3em] text-[var(--color-yolk-dim)]"
              style={{ fontFamily: "var(--font-data)" }}
            >
              Egg price forecast
            </p>
            <h1
              className="mt-1 text-2xl font-light text-[var(--color-terminal-bright)]"
              style={{ fontFamily: "var(--font-display)" }}
            >
              Will eggs go up in February 2026?
            </h1>
          </div>
          <div className="text-right">
            <p
              className="text-[10px] text-[var(--color-terminal-muted)]"
              style={{ fontFamily: "var(--font-data)" }}
            >
              Data through {data.latest_month}
            </p>
            <p
              className="text-[10px] text-[var(--color-terminal-muted)]"
              style={{ fontFamily: "var(--font-data)" }}
            >
              {data.total_observations} monthly observations
            </p>
          </div>
        </div>
      </header>

      {/* Main grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Left column: Gauge + Components */}
        <div className="space-y-6">
          {/* Probability gauge */}
          <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-6">
            <ProbabilityGauge
              probability={data.forecast.probability}
            />
          </div>

          {/* Component breakdown */}
          <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-5">
            <ComponentWaterfall
              components={data.forecast.components}
              total={data.forecast.probability}
            />
          </div>
        </div>

        {/* Center column: Price chart */}
        <div className="lg:col-span-2 space-y-6">
          <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-5">
            <PriceChart
              data={data.price_history}
              latestPrice={data.latest_price}
            />
          </div>

          {/* Kalshi + Backtest row */}
          <div className="grid gap-6 sm:grid-cols-2">
            <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-5">
              <KalshiEdge kalshi={data.kalshi} />
            </div>

            <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-5">
              <BacktestTable backtest={data.backtest} />
            </div>
          </div>
        </div>
      </div>

      {/* Model explanation — full width below */}
      <div className="mt-8 rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-6 lg:p-8">
        <ModelExplainer
          declineStreak={data.decline_streak}
          seasonalRate={data.seasonal_rate}
          recentMomentum={data.recent_momentum}
          modelInfo={data.model_info}
        />
      </div>

      {/* Footer */}
      <footer className="mt-8 border-t border-[var(--color-terminal-border)] pt-4">
        <p
          className="text-[10px] text-[var(--color-terminal-muted)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          Not financial advice. Bayesian logistic regression trained on {data.model_info?.n_commodities ?? "multiple"} food
          commodities — backtest accuracy ~{(data.backtest.accuracy * 100).toFixed(0)}%. BLS Average Price series via FRED.
        </p>
      </footer>
    </div>
  );
}
