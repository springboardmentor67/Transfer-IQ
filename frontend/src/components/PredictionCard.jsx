import React from 'react';
import { TrendingDown, TrendingUp, Sparkles } from 'lucide-react';

function normalizePredictedValue(raw) {
  const v = Number(raw) || 0;
  if (v <= 0) return 0;

  // If value looks like model output in millions, convert to EUR.
  if (v > 0 && v < 10000) return v * 1000000;
  return v;
}

function formatCompactEuro(value) {
  const abs = Math.abs(value);
  let compactValue = value;
  let suffix = '';

  if (abs >= 1e12) {
    compactValue = value / 1e12;
    suffix = ' Trillion';
  } else if (abs >= 1e9) {
    compactValue = value / 1e9;
    suffix = ' Billion';
  } else if (abs >= 1e6) {
    compactValue = value / 1e6;
    suffix = ' Million';
  } else if (abs >= 1e3) {
    compactValue = value / 1e3;
    suffix = ' Thousand';
  }

  return `EUR ${compactValue.toFixed(2)}${suffix}`;
}

function PredictionCard({ prediction, trendDirection = 'up' }) {
  const confidence = Math.round(((prediction?.confidence_score ?? prediction?.confidence ?? 0.88) * 100));
  const value = normalizePredictedValue(prediction?.predicted_transfer_value ?? prediction?.value ?? prediction?.predicted_value);
  const TrendIcon = trendDirection === 'down' ? TrendingDown : TrendingUp;

  return (
    <div className="metric-card glass glass-neon prediction-glow rounded-xl shadow-xl">
      <div className="prediction-head">
        <span className="metric-label">Predicted Market Value</span>
        <span className="metric-badge">Ensemble AI</span>
      </div>

      <div className="metric-value gradient compact-value">{formatCompactEuro(value)}</div>

      <div className="prediction-foot">
        <div className={`trend-chip ${trendDirection === 'down' ? 'down' : 'up'}`}>
          <TrendIcon size={14} />
          <span>{trendDirection === 'down' ? 'Decreasing trend' : 'Increasing trend'}</span>
        </div>
        <div className="ai-chip">
          <Sparkles size={14} />
          <span>{prediction?.model_used || 'LSTM + XGBoost'}</span>
        </div>
      </div>

      <div className="conf-wrap">
        <div className="conf-topline">
          <span>Model Confidence</span>
          <strong>{confidence}%</strong>
        </div>
        <div className="conf-bar-track">
          <div className="conf-bar-fill" style={{ width: `${confidence}%` }} />
        </div>
      </div>
    </div>
  );
}

export default PredictionCard;
