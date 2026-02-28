import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { KalshiEdge } from "../KalshiEdge";

const mockKalshi = {
  ticker: "KXEGGS-FEB26-0",
  market_yes_price: 0.31,
  model_estimate: 0.553,
  edge: 0.243,
  kelly_fraction: 0.352,
  recommendation: "BUY YES",
};

describe("KalshiEdge", () => {
  it("renders the ticker", () => {
    render(<KalshiEdge kalshi={mockKalshi} />);
    expect(screen.getByText(/KXEGGS-FEB26-0/)).toBeInTheDocument();
  });

  it("renders the recommendation", () => {
    render(<KalshiEdge kalshi={mockKalshi} />);
    expect(screen.getByText(/BUY YES/)).toBeInTheDocument();
  });

  it("shows the edge value", () => {
    render(<KalshiEdge kalshi={mockKalshi} />);
    expect(screen.getByText(/24\.3%/)).toBeInTheDocument();
  });

  it("shows market vs model comparison", () => {
    render(<KalshiEdge kalshi={mockKalshi} />);
    expect(screen.getByText(/31/)).toBeInTheDocument();
    expect(screen.getByText(/55\.3%/)).toBeInTheDocument();
  });

  it("applies positive styling for BUY YES", () => {
    const { container } = render(<KalshiEdge kalshi={mockKalshi} />);
    expect(
      container.querySelector("[data-recommendation='buy-yes']")
    ).toBeTruthy();
  });
});
