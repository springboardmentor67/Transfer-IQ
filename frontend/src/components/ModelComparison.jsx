import React from 'react';
import { BarChart, Bar, CartesianGrid, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts';

const DEFAULT_MODELS = [
  { name: 'LSTM', rmse: 5.9, mae: 4.2, accuracy: 0.87 },
  { name: 'XGBoost', rmse: 4.8, mae: 3.5, accuracy: 0.91 },
];

function getWinner(models) {
  return [...models].sort((a, b) => {
    const scoreA = a.accuracy * 100 - a.rmse * 4 - a.mae * 3;
    const scoreB = b.accuracy * 100 - b.rmse * 4 - b.mae * 3;
    return scoreB - scoreA;
  })[0]?.name;
}

function ModelComparison({ models = DEFAULT_MODELS }) {
  const winner = getWinner(models);

  const chartData = [
    { metric: 'RMSE', LSTM: models[0]?.rmse ?? 0, XGBoost: models[1]?.rmse ?? 0 },
    { metric: 'MAE', LSTM: models[0]?.mae ?? 0, XGBoost: models[1]?.mae ?? 0 },
    { metric: 'Accuracy', LSTM: (models[0]?.accuracy ?? 0) * 10, XGBoost: (models[1]?.accuracy ?? 0) * 10 },
  ];

  return (
    <div className="glass chart-card rounded-xl shadow-xl">
      <div className="section-head-row">
        <h4 className="chart-title">Model Comparison</h4>
        <span className="winner-tag">Best: {winner}</span>
      </div>

      <div className="model-rows">
        {models.map((m) => (
          <div key={m.name} className={`model-row ${winner === m.name ? 'best' : ''}`}>
            <strong>{m.name}</strong>
            <span>RMSE: {m.rmse.toFixed(2)}</span>
            <span>MAE: {m.mae.toFixed(2)}</span>
            <span>Accuracy: {(m.accuracy * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>

      <div className="chart-area" style={{ minHeight: 260 }}>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
            <XAxis dataKey="metric" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.25)' }} />
            <Bar dataKey="LSTM" radius={[6, 6, 0, 0]}>
              {chartData.map((entry, index) => <Cell key={`l-${index}`} fill="#a78bfa" />)}
            </Bar>
            <Bar dataKey="XGBoost" radius={[6, 6, 0, 0]}>
              {chartData.map((entry, index) => <Cell key={`x-${index}`} fill="#38bdf8" />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="tiny-note">Accuracy is scaled x10 for chart readability against error metrics.</p>
    </div>
  );
}

export default ModelComparison;
