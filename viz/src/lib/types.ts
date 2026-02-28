export interface ForecastData {
  generated_at: string;
  latest_price: number;
  latest_month: string;
  total_observations: number;
  forecast: {
    probability: number;
    components: Record<string, number>;
  };
  model_info: {
    n_obs: number;
    n_commodities: number;
    coefficients: Record<string, number>;
  };
  kalshi: {
    ticker: string;
    market_yes_price: number;
    model_estimate: number;
    edge: number;
    kelly_fraction: number;
    recommendation: string;
  };
  backtest: {
    accuracy: number;
    brier_score: number;
    predictions: Array<{
      year: number;
      predicted_prob: number;
      actual_up: boolean;
      correct: boolean;
    }>;
  };
  price_history: Array<{ date: string; price: number }>;
  all_prices: Array<{ date: string; price: number }>;
  mom_changes: Array<{
    date: string;
    abs_change: number;
    pct_change: number;
  }>;
  feb_history: Array<{
    year: number;
    jan_price: number;
    feb_price: number;
    change: number;
    pct_change: number;
    increased: boolean;
  }>;
  decline_streak: number;
  seasonal_rate: number;
  recent_momentum: number;
}
