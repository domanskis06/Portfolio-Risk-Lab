-- Schema for the Quant Research & Market Risk Warehouse.
-- Apply with scripts/init_db.py (idempotent). Timestamps are TIMESTAMPTZ, stored in UTC.

CREATE TABLE IF NOT EXISTS instruments (
  instrument_id SERIAL PRIMARY KEY,
  ticker TEXT NOT NULL UNIQUE,
  name TEXT,
  asset_class TEXT,
  currency TEXT,
  is_benchmark BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS prices_raw (
  ticker TEXT NOT NULL,
  dt DATE NOT NULL,
  open DOUBLE PRECISION,
  high DOUBLE PRECISION,
  low DOUBLE PRECISION,
  close DOUBLE PRECISION,
  adj_close DOUBLE PRECISION,
  volume DOUBLE PRECISION,
  ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  PRIMARY KEY (ticker, dt)
);

CREATE TABLE IF NOT EXISTS prices_clean (
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  dt DATE NOT NULL,
  adj_close DOUBLE PRECISION NOT NULL,
  quality_flag TEXT,
  PRIMARY KEY (instrument_id, dt)
);

CREATE TABLE IF NOT EXISTS features (
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  dt DATE NOT NULL,
  ret_1d DOUBLE PRECISION,
  vol_21d DOUBLE PRECISION,
  beta_63d DOUBLE PRECISION,
  PRIMARY KEY (instrument_id, dt)
);

CREATE TABLE IF NOT EXISTS portfolios (
  portfolio_id SERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS portfolio_weights (
  portfolio_id INT NOT NULL REFERENCES portfolios(portfolio_id),
  instrument_id INT NOT NULL REFERENCES instruments(instrument_id),
  weight DOUBLE PRECISION NOT NULL,
  valid_from DATE NOT NULL,
  valid_to DATE,
  PRIMARY KEY (portfolio_id, instrument_id, valid_from)
);

-- One row per risk experiment: method, parameters and results.
CREATE TABLE IF NOT EXISTS risk_runs (
  run_id SERIAL PRIMARY KEY,
  portfolio_id INT REFERENCES portfolios(portfolio_id),
  method TEXT NOT NULL,
  alpha DOUBLE PRECISION NOT NULL,
  horizon_days INT NOT NULL,
  window_start DATE,
  window_end DATE,
  var_value DOUBLE PRECISION,
  es_value DOUBLE PRECISION,
  params_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_runs (
  run_id SERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  train_start DATE,
  train_end DATE,
  test_start DATE,
  test_end DATE,
  metrics_json JSONB,
  params_json JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
  run_id INT NOT NULL REFERENCES model_runs(run_id),
  dt DATE NOT NULL,
  instrument_id INT,
  y_true DOUBLE PRECISION,
  y_pred DOUBLE PRECISION,
  PRIMARY KEY (run_id, dt, instrument_id)
);
