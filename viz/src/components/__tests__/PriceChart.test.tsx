import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PriceChart } from "../PriceChart";

vi.mock("recharts", async () => {
  const actual = await vi.importActual("recharts");
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <div data-testid="chart-container" style={{ width: 800, height: 400 }}>
        {children}
      </div>
    ),
  };
});

const mockPrices = [
  { date: "2024-01-01", price: 2.5 },
  { date: "2024-06-01", price: 3.2 },
  { date: "2024-12-01", price: 3.65 },
  { date: "2025-01-01", price: 4.95 },
  { date: "2025-06-01", price: 3.78 },
  { date: "2026-01-01", price: 2.58 },
];

const mockForecast = {
  point_forecast: 2.70,
  current_price: 2.58,
  intervals: {
    "90": { low: 2.45, high: 3.22 },
    "50": { low: 2.57, high: 2.70 },
  },
  quantiles: { "0.5": 2.70 },
  samples_summary: { mean: 2.71, std: 0.19, n: 1000 },
};

describe("PriceChart", () => {
  it("renders the chart title", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} priceForecast={mockForecast} />);
    expect(screen.getByText(/egg price/i)).toBeInTheDocument();
  });

  it("shows the latest price", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} priceForecast={mockForecast} />);
    expect(screen.getByText(/\$2\.58/)).toBeInTheDocument();
  });

  it("renders a container for the chart", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} priceForecast={mockForecast} />);
    expect(screen.getByTestId("chart-container")).toBeInTheDocument();
  });

  it("shows forecast price in header", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} priceForecast={mockForecast} />);
    expect(screen.getByText(/forecast.*\$2\.70/i)).toBeInTheDocument();
  });

  it("renders forecast legend", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} priceForecast={mockForecast} />);
    expect(screen.getByText(/50% interval/)).toBeInTheDocument();
    expect(screen.getByText(/90% interval/)).toBeInTheDocument();
    expect(screen.getByText(/point forecast/)).toBeInTheDocument();
  });
});
