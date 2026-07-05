-- bqml_forecast.sql -- Part 3: BigQuery ML ARIMA_PLUS
-- Usage: python src/ml/bqml_forecast.py

-- Step 1: Daily revenue view
CREATE OR REPLACE VIEW `{project_id}.{dataset}.daily_sku_revenue` AS
SELECT sku_id, sku_name, category, CAST(date AS DATE) AS ds,
  SUM(total_amount) AS revenue, SUM(quantity) AS qty_sold,
  COUNT(*) AS txn_count, AVG(unit_price) AS avg_price
FROM `{project_id}.{dataset}.raw_transactions`
GROUP BY 1,2,3,4;

-- Step 2: Multi-SKU ARIMA_PLUS model
CREATE OR REPLACE MODEL `{project_id}.{dataset}.arima_sku_forecast`
OPTIONS(
  model_type='ARIMA_PLUS',
  time_series_timestamp_col='ds',
  time_series_data_col='revenue',
  time_series_id_col='sku_id',
  data_frequency='DAILY',
  decompose_time_series=TRUE,
  clean_spikes_and_dips=TRUE,
  adjust_step_changes=TRUE
)
AS SELECT sku_id, ds, revenue
FROM `{project_id}.{dataset}.daily_sku_revenue`
WHERE ds >= DATE_SUB(CURRENT_DATE(), INTERVAL 2 YEAR);

-- Step 3: 30-day forecast
SELECT sku_id,
  CAST(forecast_timestamp AS DATE) AS forecast_date,
  ROUND(forecast_value, 2) AS forecast_revenue,
  ROUND(prediction_interval_lower_bound, 2) AS lower_bound,
  ROUND(prediction_interval_upper_bound, 2) AS upper_bound
FROM ML.FORECAST(
  MODEL `{project_id}.{dataset}.arima_sku_forecast`,
  STRUCT(30 AS horizon, 0.90 AS confidence_level)
)
ORDER BY sku_id, forecast_date;

-- Step 4: Model evaluation
SELECT sku_id, mean_absolute_error, mean_absolute_percentage_error,
  symmetric_mean_absolute_percentage_error
FROM ML.EVALUATE(MODEL `{project_id}.{dataset}.arima_sku_forecast`)
ORDER BY mean_absolute_percentage_error;
