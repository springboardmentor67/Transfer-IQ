import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { Smile, MinusCircle, Frown } from 'lucide-react';

const COLORS = {
  positive: '#34d399',
  neutral: '#fbbf24',
  negative: '#fb7185',
};

function buildSentimentData(score = 0) {
  const clamped = Math.max(-1, Math.min(1, Number(score) || 0));
  const pos = Math.max(0, 35 + clamped * 45);
  const neg = Math.max(0, 35 - clamped * 45);
  const neu = Math.max(0, 100 - pos - neg);

  return [
    { name: 'Positive', value: Number(pos.toFixed(1)), key: 'positive' },
    { name: 'Neutral', value: Number(neu.toFixed(1)), key: 'neutral' },
    { name: 'Negative', value: Number(neg.toFixed(1)), key: 'negative' },
  ];
}

function SentimentChart({ score = 0.0 }) {
  const data = buildSentimentData(score);
  const mood = score >= 0.25 ? 'Positive' : score <= -0.25 ? 'Negative' : 'Neutral';

  return (
    <div className="glass chart-card rounded-xl shadow-xl">
      <h4 className="chart-title">Sentiment Analysis</h4>
      <div className="sentiment-grid">
        <div className="sentiment-pie-wrap">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={data}
                innerRadius={55}
                outerRadius={85}
                dataKey="value"
                paddingAngle={3}
                stroke="none"
              >
                {data.map((entry) => (
                  <Cell key={entry.name} fill={COLORS[entry.key]} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{ background: '#0f172a', border: '1px solid rgba(148,163,184,0.25)' }}
                formatter={(value) => [`${value}%`, 'Share']}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="sentiment-meta">
          <div className="sentiment-score">
            <span className="metric-label">Sentiment Score</span>
            <strong className={score >= 0 ? 'sent-up' : 'sent-down'}>{score >= 0 ? '+' : ''}{score.toFixed(2)} {mood}</strong>
          </div>

          <div className="sentiment-legend">
            <div><Smile size={14} color={COLORS.positive} /> Positive</div>
            <div><MinusCircle size={14} color={COLORS.neutral} /> Neutral</div>
            <div><Frown size={14} color={COLORS.negative} /> Negative</div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SentimentChart;
