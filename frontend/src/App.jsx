import React, { useState } from 'react';
import './App.css';

const FEATURES = ['Goals', 'Assists', 'Shots', 'Passes', 'xG', 'Sentiment', 'Social Impact', 'Tweet Volume', 'Market Value (€)'];
const FEATURE_KEYS = ['goals', 'assists', 'shots', 'passes', 'xg', 'daily_sentiment', 'daily_impact', 'daily_tweet_vol', 'market_value_eur'];
const DEFAULTS = [1, 0, 3, 60, 0.35, 0.2, 25000, 200, 50000000];

function App() {
  const [playerName, setPlayerName] = useState('');
  const [days, setDays] = useState(
    Array.from({ length: 7 }, (_, i) => Object.fromEntries(FEATURE_KEYS.map((k, j) => [k, DEFAULTS[j]])))
  );
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeDay, setActiveDay] = useState(0);

  const updateDay = (dayIndex, key, value) => {
    setDays(prev => prev.map((d, i) => i === dayIndex ? { ...d, [key]: parseFloat(value) || 0 } : d));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);
    const data = days.map(d => FEATURE_KEYS.map(k => d[k]));
    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data }),
      });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const result = await res.json();
      setPrediction(result.predicted_market_value_eur);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleRandom = () => {
    setDays(Array.from({ length: 7 }, () => ({
      goals: Math.floor(Math.random() * 3),
      assists: Math.floor(Math.random() * 3),
      shots: Math.floor(Math.random() * 6),
      passes: Math.floor(Math.random() * 80) + 20,
      xg: parseFloat((Math.random()).toFixed(2)),
      daily_sentiment: parseFloat((Math.random() - 0.5).toFixed(2)),
      daily_impact: Math.floor(Math.random() * 50000),
      daily_tweet_vol: Math.floor(Math.random() * 500),
      market_value_eur: 50000000,
    })));
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="app">
      <div className="bg-orbs">
        <div className="orb orb1" />
        <div className="orb orb2" />
        <div className="orb orb3" />
      </div>

      <header className="hero">
        <div className="badge">AI-Powered</div>
        <h1>⚽ Player Market Value Predictor</h1>
        <p>Enter 7 days of player stats to get a live market value prediction from our trained LSTM model.</p>
      </header>

      <main className="container">
        <div className="card input-card">
          <div className="card-header">
            <h2>Player Input</h2>
            <div className="controls">
              <input
                className="player-input"
                placeholder="Player name (optional)"
                value={playerName}
                onChange={e => setPlayerName(e.target.value)}
              />
              <button className="btn-secondary" onClick={handleRandom}>🎲 Randomize</button>
            </div>
          </div>

          <div className="day-tabs">
            {days.map((_, i) => (
              <button
                key={i}
                className={`day-tab ${activeDay === i ? 'active' : ''}`}
                onClick={() => setActiveDay(i)}
              >
                Day {i + 1}
              </button>
            ))}
          </div>

          <div className="features-grid">
            {FEATURE_KEYS.map((key, j) => (
              <div key={key} className="feature-input">
                <label>{FEATURES[j]}</label>
                <input
                  type="number"
                  value={days[activeDay][key]}
                  onChange={e => updateDay(activeDay, key, e.target.value)}
                  step={key === 'xg' || key === 'daily_sentiment' ? '0.01' : '1'}
                />
              </div>
            ))}
          </div>

          <button
            className="btn-primary"
            onClick={handlePredict}
            disabled={loading}
          >
            {loading ? <span className="spinner" /> : '🔮 Predict Market Value'}
          </button>
        </div>

        {error && (
          <div className="result-card error-card">
            <span className="error-icon">⚠️</span>
            <p>{error}</p>
            <p className="hint">Make sure the API is running: <code>uvicorn app:app --port 8000</code></p>
          </div>
        )}

        {prediction !== null && (
          <div className="result-card success-card">
            <div className="result-label">
              {playerName ? `${playerName}'s` : 'Predicted'} Market Value
            </div>
            <div className="result-value">
              €{prediction.toLocaleString(undefined, { minimumFractionDigits: 0, maximumFractionDigits: 0 })}
            </div>
            <div className="result-sub">Based on last 7 days of performance + sentiment data</div>
            <div className="model-badge">LSTM Neural Network</div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
