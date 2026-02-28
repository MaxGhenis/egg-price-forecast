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
  const pf = data.price_forecast;
  const bt = data.backtest;

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
        {/* Left column: Gauge + Forecast card + Components */}
        <div className="space-y-6">
          {/* Probability gauge */}
          <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-6">
            <ProbabilityGauge
              probability={data.forecast.probability}
              rawProbability={data.forecast.raw_probability}
              baseRate={data.forecast.calibration?.base_rate}
            />
          </div>

          {/* Price forecast card */}
          <div className="rounded-xl border border-[var(--color-terminal-border)] bg-[var(--color-terminal-surface)] p-5">
            <h3
              className="mb-3 text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
              style={{ fontFamily: "var(--font-data)" }}
            >
              Price forecast
            </h3>
            <div className="space-y-3">
              {/* Point forecast */}
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-[var(--color-terminal-text)]">Predicted price</span>
                <span
                  className="text-xl font-light text-[var(--color-up)]"
                  style={{ fontFamily: "var(--font-data)" }}
                >
                  ${pf.point_forecast.toFixed(2)}
                </span>
              </div>

              {/* Current price */}
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-[var(--color-terminal-text)]">Current price</span>
                <span
                  className="text-sm text-[var(--color-terminal-muted)]"
                  style={{ fontFamily: "var(--font-data)" }}
                >
                  ${pf.current_price.toFixed(2)}
                </span>
              </div>

              {/* Intervals */}
              <div className="border-t border-[var(--color-terminal-border)] pt-2">
                <div className="flex items-baseline justify-between text-xs">
                  <span className="text-[var(--color-terminal-muted)]">50% interval</span>
                  <span style={{ fontFamily: "var(--font-data)", color: "var(--color-terminal-text)" }}>
                    ${pf.intervals["50"].low.toFixed(2)} – ${pf.intervals["50"].high.toFixed(2)}
                  </span>
                </div>
                <div className="mt-1 flex items-baseline justify-between text-xs">
                  <span className="text-[var(--color-terminal-muted)]">90% interval</span>
                  <span style={{ fontFamily: "var(--font-data)", color: "var(--color-terminal-text)" }}>
                    ${pf.intervals["90"].low.toFixed(2)} – ${pf.intervals["90"].high.toFixed(2)}
                  </span>
                </div>
              </div>

              {/* Change direction */}
              <div className="border-t border-[var(--color-terminal-border)] pt-2">
                <div className="flex items-baseline justify-between text-xs">
                  <span className="text-[var(--color-terminal-muted)]">Expected change</span>
                  <span
                    style={{
                      fontFamily: "var(--font-data)",
                      color: pf.point_forecast > pf.current_price ? "var(--color-up)" : "var(--color-down)",
                    }}
                  >
                    {pf.point_forecast > pf.current_price ? "+" : ""}
                    {((pf.point_forecast - pf.current_price) / pf.current_price * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
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
              priceForecast={data.price_forecast}
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
          modelInfo={data.model_info}
          backtest={bt}
        />
      </div>

      {/* Footer */}
      <footer className="mt-8 border-t border-[var(--color-terminal-border)] pt-4">
        <p
          className="text-[10px] text-[var(--color-terminal-muted)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          Not financial advice.{" "}
          {data.model_info.type === "bsts_ucm"
            ? "Bayesian Structural Time Series (local level + seasonal + PPI distributed lags + AR(1))"
            : `LightGBM quantile regression (${data.model_info.n_quantile_models} quantile models) trained on 15 food commodities`}
          {" "}— backtest direction accuracy ~{(bt.direction_accuracy * 100).toFixed(0)}%
          , MAPE {(bt.mape * 100).toFixed(0)}%
          , 90% coverage {(bt.coverage_90 * 100).toFixed(0)}%.
          BLS Average Price series via FRED.
        </p>
      </footer>
    </div>
  );
}
