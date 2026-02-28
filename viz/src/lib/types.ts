export interface PriceForecast {
  point_forecast: number;
  current_price: number;
  intervals: Record<string, { low: number; high: number }>;
  quantiles: Record<string, number>;
  samples_summary: {
    mean: number;
    std: number;
    n: number;
  };
}

export interface BacktestPrediction {
  year: number;
  month: number;
  date: string;
  predicted_prob: number;
  predicted_price: number;
  actual_price: number;
  current_price: number;
  actual_up: boolean;
  correct: boolean;
  error: number;
  in_90_interval: boolean;
  in_50_interval: boolean;
}

export interface ForecastData {
  generated_at: string;
  latest_price: number;
  latest_month: string;
  total_observations: number;
  forecast: {
    probability: number;
    raw_probability?: number;
    calibration?: {
      raw_p_up: number;
      calibrated_p_up: number;
      base_rate: number;
      model_weight: number;
      brier_score: number;
    };
    components: Record<string, number>;
  };
  price_forecast: PriceForecast;
  model_info: {
    type: string;
    n_quantile_models: number;
    n_samples: number;
    feature_importances: Record<string, number>;
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
    direction_accuracy: number;
    mape: number;
    coverage_90: number;
    coverage_50: number;
    brier_score: number;
    predictions: BacktestPrediction[];
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
