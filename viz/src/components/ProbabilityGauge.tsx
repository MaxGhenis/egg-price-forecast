"use client";

interface ProbabilityGaugeProps {
  probability: number;
  rawProbability?: number;
  baseRate?: number;
}

export function ProbabilityGauge({ probability, rawProbability, baseRate }: ProbabilityGaugeProps) {
  const pct = (probability * 100).toFixed(1);
  const isBullish = probability >= 0.5;
  const signal = isBullish ? "bullish" : "bearish";

  // Arc geometry for the gauge
  const radius = 88;
  const cx = 100;
  const cy = 100;
  const startAngle = -210;
  const endAngle = 30;
  const range = endAngle - startAngle;

  function angleToXY(angleDeg: number) {
    const rad = (angleDeg * Math.PI) / 180;
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) };
  }

  const needleAngle = startAngle + range * probability;

  // Arc path helper
  function arcPath(start: number, end: number, r: number) {
    const s = angleToXY(start);
    const e = angleToXY(end);
    const sweep = end - start > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${sweep} 1 ${e.x} ${e.y}`;
  }

  const needleEnd = angleToXY(needleAngle);

  return (
    <div
      data-signal={signal}
      className="relative flex flex-col items-center"
    >
      <p className="mb-3 text-xs font-medium uppercase tracking-[0.2em] text-[var(--color-terminal-muted)]"
         style={{ fontFamily: "var(--font-data)" }}>
        Probability of Feb increase
      </p>

      <svg viewBox="0 0 200 130" className="w-full max-w-[280px]">
        {/* Background arc */}
        <path
          d={arcPath(startAngle, endAngle, radius)}
          fill="none"
          stroke="var(--color-terminal-border)"
          strokeWidth={6}
          strokeLinecap="round"
        />

        {/* Filled arc up to needle */}
        <path
          d={arcPath(startAngle, needleAngle, radius)}
          fill="none"
          stroke={isBullish ? "var(--color-up)" : "var(--color-down)"}
          strokeWidth={6}
          strokeLinecap="round"
        />

        {/* Needle */}
        <line
          x1={cx}
          y1={cy}
          x2={needleEnd.x}
          y2={needleEnd.y}
          stroke={isBullish ? "var(--color-up)" : "var(--color-down)"}
          strokeWidth={2.5}
          strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={4} fill={isBullish ? "var(--color-up)" : "var(--color-down)"} />

        {/* 50% tick mark */}
        {(() => {
          const midAngle = startAngle + range * 0.5;
          const outer = angleToXY(midAngle);
          const innerR = radius - 10;
          const rad = (midAngle * Math.PI) / 180;
          const inner = { x: cx + innerR * Math.cos(rad), y: cy + innerR * Math.sin(rad) };
          return (
            <line
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke="var(--color-terminal-muted)"
              strokeWidth={1}
              opacity={0.5}
            />
          );
        })()}

        {/* Labels */}
        <text x={20} y={120} fill="var(--color-terminal-muted)" fontSize={8} fontFamily="var(--font-data)">0%</text>
        <text x={168} y={120} fill="var(--color-terminal-muted)" fontSize={8} fontFamily="var(--font-data)">100%</text>
      </svg>

      {/* Main percentage */}
      <p
        className="-mt-6 text-5xl font-light tracking-tight"
        style={{
          fontFamily: "var(--font-display)",
          color: isBullish ? "var(--color-up)" : "var(--color-down)",
        }}
      >
        {pct}%
      </p>
      {(rawProbability !== undefined || baseRate !== undefined) && (
        <p
          className="mt-1 text-[10px] text-[var(--color-terminal-muted)]"
          style={{ fontFamily: "var(--font-data)" }}
        >
          {rawProbability !== undefined && `raw: ${(rawProbability * 100).toFixed(0)}%`}
          {rawProbability !== undefined && baseRate !== undefined && " · "}
          {baseRate !== undefined && `base rate: ${(baseRate * 100).toFixed(0)}%`}
        </p>
      )}
    </div>
  );
}
