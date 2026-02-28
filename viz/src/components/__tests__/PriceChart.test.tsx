import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { PriceChart } from "../PriceChart";

// Recharts uses ResponsiveContainer which needs dimensions
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

describe("PriceChart", () => {
  it("renders the chart title", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} />);
    expect(screen.getByText(/egg price/i)).toBeInTheDocument();
  });

  it("shows the latest price", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} />);
    expect(screen.getByText(/\$2\.58/)).toBeInTheDocument();
  });

  it("renders a container for the chart", () => {
    render(<PriceChart data={mockPrices} latestPrice={2.58} />);
    expect(screen.getByTestId("chart-container")).toBeInTheDocument();
  });
});
