"use client";

import type { PriceForecast } from "@/lib/types";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

interface PriceChartProps {
  data: Array<{ date: string; price: number }>;
  latestPrice: number;
  priceForecast: PriceForecast;
}

export function PriceChart({ data, latestPrice, priceForecast }: PriceChartProps) {
  // Build chart data with forecast extension
  const chartData = data.map((d) => ({
    date: d.date,
    label: new Date(d.date).toLocaleDateString("en-US", {
      year: "2-digit",
      month: "short",
    }),
    price: d.price,
    // These will be undefined for historical points
    forecastPrice: undefined as number | undefined,
    interval90Low: undefined as number | undefined,
    interval90High: undefined as number | undefined,
    interval50Low: undefined as number | undefined,
    interval50High: undefined as number | undefined,
  }));

  // Add forecast point extending one month into the future
  const lastDate = new Date(data[data.length - 1].date);
  const forecastDate = new Date(lastDate);
  forecastDate.setMonth(forecastDate.getMonth() + 1);
  const forecastLabel = forecastDate.toLocaleDateString("en-US", {
    year: "2-digit",
    month: "short",
  });

  const i90 = priceForecast.intervals["90"];
  const i50 = priceForecast.intervals["50"];

  // Bridge: the last historical point also starts the forecast bands
  if (chartData.length > 0) {
    const last = chartData[chartData.length - 1];
    last.forecastPrice = last.price;
    last.interval90Low = last.price;
    last.interval90High = last.price;
    last.interval50Low = last.price;
    last.interval50High = last.price;
  }

  chartData.push({
    date: forecastDate.toISOString().slice(0, 10),
    label: forecastLabel,
    price: undefined as unknown as number,
    forecastPrice: priceForecast.point_forecast,
    interval90Low: i90.low,
    interval90High: i90.high,
    interval50Low: i50.low,
    interval50High: i50.high,
  });

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h3
          className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          BLS egg price — per dozen
        </h3>
        <div className="flex items-baseline gap-3">
          <span
            className="text-xs text-[var(--color-terminal-muted)]"
            style={{ fontFamily: "var(--font-data)" }}
          >
            forecast: ${priceForecast.point_forecast.toFixed(2)}
          </span>
          <span
            className="text-lg font-light text-[var(--color-yolk)]"
            style={{ fontFamily: "var(--font-data)" }}
          >
            ${latestPrice.toFixed(2)}
          </span>
        </div>
      </div>

      <div className="h-[300px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={chartData}
            margin={{ top: 8, right: 8, bottom: 0, left: -10 }}
          >
            <defs>
              <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#f5a623" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#f5a623" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="2 6"
              stroke="var(--color-terminal-border)"
              vertical={false}
            />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 9, fill: "var(--color-terminal-muted)", fontFamily: "var(--font-data)" }}
              tickLine={false}
              axisLine={{ stroke: "var(--color-terminal-border)" }}
              interval="preserveStartEnd"
              minTickGap={60}
            />
            <YAxis
              tick={{ fontSize: 9, fill: "var(--color-terminal-muted)", fontFamily: "var(--font-data)" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) => `$${v.toFixed(1)}`}
              domain={["dataMin - 0.2", "dataMax + 0.3"]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-terminal-bg)",
                border: "1px solid var(--color-terminal-border)",
                borderRadius: 6,
                fontSize: 11,
                fontFamily: "var(--font-data)",
                color: "var(--color-terminal-text)",
              }}
              formatter={(v: number, name: string) => {
                if (v == null) return [null, null];
                const labels: Record<string, string> = {
                  price: "Price",
                  forecastPrice: "Forecast",
                  interval90High: "90% high",
                  interval90Low: "90% low",
                  interval50High: "50% high",
                  interval50Low: "50% low",
                };
                return [`$${v.toFixed(3)}`, labels[name] || name];
              }}
              labelFormatter={(label: string) => label}
            />
            <ReferenceLine
              y={latestPrice}
              stroke="var(--color-yolk)"
              strokeDasharray="4 4"
              strokeOpacity={0.5}
            />

            {/* 90% prediction interval (light band) */}
            <Area
              type="monotone"
              dataKey="interval90High"
              stroke="none"
              fill="rgba(52, 211, 153, 0.08)"
              dot={false}
              activeDot={false}
              connectNulls={false}
            />
            <Area
              type="monotone"
              dataKey="interval90Low"
              stroke="none"
              fill="var(--color-terminal-bg)"
              dot={false}
              activeDot={false}
              connectNulls={false}
            />

            {/* 50% prediction interval (medium band) */}
            <Area
              type="monotone"
              dataKey="interval50High"
              stroke="none"
              fill="rgba(52, 211, 153, 0.15)"
              dot={false}
              activeDot={false}
              connectNulls={false}
            />
            <Area
              type="monotone"
              dataKey="interval50Low"
              stroke="none"
              fill="var(--color-terminal-bg)"
              dot={false}
              activeDot={false}
              connectNulls={false}
            />

            {/* Forecast line (dashed) */}
            <Area
              type="monotone"
              dataKey="forecastPrice"
              stroke="var(--color-up)"
              strokeWidth={1.5}
              strokeDasharray="6 3"
              fill="none"
              dot={{ r: 4, fill: "var(--color-up)", stroke: "var(--color-terminal-bg)", strokeWidth: 2 }}
              activeDot={{ r: 5, fill: "var(--color-up)", stroke: "var(--color-terminal-bg)", strokeWidth: 2 }}
              connectNulls={false}
            />

            {/* Historical price line */}
            <Area
              type="monotone"
              dataKey="price"
              stroke="var(--color-yolk)"
              strokeWidth={1.5}
              fill="url(#priceGradient)"
              dot={false}
              activeDot={{
                r: 3,
                fill: "var(--color-yolk)",
                stroke: "var(--color-terminal-bg)",
                strokeWidth: 2,
              }}
              connectNulls={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend for forecast bands */}
      <div className="flex items-center gap-4 text-[10px] text-[var(--color-terminal-muted)]" style={{ fontFamily: "var(--font-data)" }}>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-4 rounded-sm" style={{ backgroundColor: "rgba(52, 211, 153, 0.15)" }} />
          50% interval
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-4 rounded-sm" style={{ backgroundColor: "rgba(52, 211, 153, 0.08)" }} />
          90% interval
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-0.5 w-4 border-t border-dashed border-[var(--color-up)]" />
          point forecast
        </span>
      </div>
    </div>
  );
}
