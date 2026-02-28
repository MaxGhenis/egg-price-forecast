"use client";

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
}

export function PriceChart({ data, latestPrice }: PriceChartProps) {
  // Format dates for display
  const chartData = data.map((d) => ({
    ...d,
    label: new Date(d.date).toLocaleDateString("en-US", {
      year: "2-digit",
      month: "short",
    }),
  }));

  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <h3
          className="text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          BLS egg price — per dozen
        </h3>
        <span
          className="text-lg font-light text-[var(--color-yolk)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          ${latestPrice.toFixed(2)}
        </span>
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
              formatter={(v: number) => [`$${v.toFixed(3)}`, "Price"]}
              labelFormatter={(label: string) => label}
            />
            <ReferenceLine
              y={latestPrice}
              stroke="var(--color-yolk)"
              strokeDasharray="4 4"
              strokeOpacity={0.5}
            />
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
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
