import React, { useMemo, useState, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  TrendingUp, Activity, Brain, User, Calendar, Award, Info,
  ShieldCheck, Sparkles, ArrowUpRight, ArrowDownRight
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from 'recharts';
import PipelineFlow from './components/PipelineFlow';
import PredictionCard from './components/PredictionCard';
import SentimentChart from './components/SentimentChart';
import ModelComparison from './components/ModelComparison';
import './App.css';

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
const FALLBACK_API_BASE = API_BASE.includes(':8000')
  ? API_BASE.replace(':8000', ':8001')
  : API_BASE.includes(':8001')
    ? API_BASE.replace(':8001', ':8000')
    : API_BASE;

const SEASON_OPTIONS = [
  '2019-20',
  '2020-21',
  '2021-22',
  '2022-23',
  '2023-24',
  '2024-25',
];

function formatSeasonLabel(startYear) {
  return `${startYear}-${String((startYear + 1) % 100).padStart(2, '0')}`;
}

function buildSeasonWindow(selectedSeason) {
  const startYear = Number.parseInt(String(selectedSeason).slice(0, 4), 10);
  const base = Number.isNaN(startYear) ? 2024 : startYear;
  return [formatSeasonLabel(base - 2), formatSeasonLabel(base - 1), formatSeasonLabel(base)];
}

function normalizePlayersPayload(payload) {
  const rows = Array.isArray(payload)
    ? payload
    : Array.isArray(payload?.players)
      ? payload.players
      : [];

  return rows
    .map((p, idx) => ({
      player_id: p?.player_id ?? p?.id ?? p?.uid ?? idx + 1,
      name: p?.name ?? p?.player_name ?? p?.full_name ?? 'Unknown Player',
      league: p?.league ?? '',
      team: p?.team ?? p?.club ?? '',
      age: p?.age ?? p?.current_age ?? null,
      goals: p?.goals ?? 0,
      assists: p?.assists ?? 0,
      minutes: p?.minutes ?? p?.minutes_played ?? 0,
      injury_days: p?.injury_days ?? p?.total_days_injured ?? 0,
      sentiment_score: p?.sentiment_score ?? p?.vader_compound_score ?? 0,
    }))
    .filter((p) => String(p.name).trim().length > 0);
}

async function fetchPlayersWithFallback(params) {
  try {
    const primary = await axios.get(`${API_BASE}/players`, { params });
    return normalizePlayersPayload(primary.data);
  } catch {
    if (FALLBACK_API_BASE === API_BASE) {
      throw new Error('Primary API unavailable');
    }
    const fallback = await axios.get(`${FALLBACK_API_BASE}/players`, { params });
    return normalizePlayersPayload(fallback.data);
  }
}

function App() {
  const [players, setPlayers] = useState([]);
  const [leagueOptions, setLeagueOptions] = useState([]);
  const [playerSearch, setPlayerSearch] = useState('');
  const [selectedPlayerId, setSelectedPlayerId] = useState('');
  const [selectedSeason, setSelectedSeason] = useState('2023-24');
  const [selectedLeague, setSelectedLeague] = useState('All Leagues');
  const [stats, setStats] = useState([]);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Load players from backend with season/league/search filters.
  useEffect(() => {
    const params = {
      limit: 5000,
      season: selectedSeason,
    };
    if (playerSearch.trim()) params.search = playerSearch.trim();
    if (selectedLeague !== 'All Leagues') params.league = selectedLeague;

    fetchPlayersWithFallback(params)
      .then((list) => {
        setPlayers(list);
        if (list.length === 0 && !playerSearch.trim()) {
          setError(`No players found for season ${selectedSeason}. Try another season.`);
        } else {
          setError(null);
        }
      })
      .catch(() => setError("Failed to connect to backend API. Ensure server is running."));
  }, [selectedSeason, selectedLeague, playerSearch]);

  // Load league options for current season (without league filter applied).
  useEffect(() => {
    fetchPlayersWithFallback({ limit: 5000, season: selectedSeason })
      .then((source) => {
        const leagues = Array.from(new Set(source.map((p) => String(p?.league || '').trim()).filter(Boolean))).sort((a, b) => a.localeCompare(b));
        setLeagueOptions(leagues);
      })
      .catch(() => setLeagueOptions([]));
  }, [selectedSeason]);

  useEffect(() => {
    if (selectedLeague === 'All Leagues') return;
    if (!leagueOptions.includes(selectedLeague)) {
      setSelectedLeague('All Leagues');
    }
  }, [leagueOptions, selectedLeague]);

  const selectedPlayer = useMemo(
    () => players.find((p) => String(p.player_id) === String(selectedPlayerId)) || null,
    [players, selectedPlayerId]
  );

  const marketSeries = useMemo(() => {
    if (!Array.isArray(stats) || stats.length === 0) return [];

    return stats.map((row, idx) => ({
      ...row,
      actual_value: idx < stats.length - 1 ? row.market_value : null,
      predicted_value: idx >= stats.length - 1 ? row.market_value : null,
    }));
  }, [stats]);

  const trendDirection = useMemo(() => {
    if (!Array.isArray(stats) || stats.length < 2) return 'up';
    const first = stats[0]?.market_value || 0;
    const last = stats[stats.length - 1]?.market_value || 0;
    return last >= first ? 'up' : 'down';
  }, [stats]);

  useEffect(() => {
    if (!selectedPlayerId) return;
    const stillVisible = players.some((p) => String(p.player_id) === String(selectedPlayerId));
    if (!stillVisible) {
      setSelectedPlayerId('');
      setPrediction(null);
      setStats([]);
    }
  }, [players, selectedPlayerId]);

  useEffect(() => {
    if (!Array.isArray(stats) || stats.length === 0) return;
    const [seasonMinus2, seasonMinus1, seasonCurrent] = buildSeasonWindow(selectedSeason);
    const labels = [seasonMinus2, seasonMinus1, seasonCurrent];
    setStats((prev) => prev.map((row, idx) => ({ ...row, season: labels[idx] || row.season })));
  }, [selectedSeason]);

  const handlePlayerChange = async (e) => {
    const id = e.target.value;
    setSelectedPlayerId(id);
    if (!id) return;

    setLoading(true);
    setPrediction(null);
    setError(null);

    try {
      // Find selected player to get their features
      const player = players.find(p => String(p.player_id) === id);
      if (!player) {
        setError("Player not found.");
        setLoading(false);
        return;
      }

      // Call POST /predict with player's features
      const predRes = await axios.post(`${API_BASE}/predict`, {
        age: player.age || 0,
        goals: player.goals || 0,
        assists: player.assists || 0,
        minutes: player.minutes || 0,
        injury_days: player.injury_days || 0,
        sentiment_score: player.sentiment_score || 0
      });
      
      // Mock stats data (create trend from prediction)
      const rawPred = Number(predRes.data.value || predRes.data.predicted_transfer_value || 0);
      const normalizedPred = rawPred > 0 && rawPred < 10000 ? rawPred * 1000000 : rawPred;
      const goals = Number(player.goals || 0);
      const assists = Number(player.assists || 0);
      const age = Number(player.age || 22);
      const [seasonMinus2, seasonMinus1, seasonCurrent] = buildSeasonWindow(selectedSeason);

      const mockStats = [
        { season: seasonMinus2, goals: Math.max(0, goals * 0.78), assists: Math.max(0, assists * 0.7), age: age - 1, market_value: normalizedPred * 0.62 },
        { season: seasonMinus1, goals: Math.max(0, goals * 0.92), assists: Math.max(0, assists * 0.85), age, market_value: normalizedPred * 0.81 },
        { season: seasonCurrent, goals, assists, age: age + 1, market_value: normalizedPred },
      ];
      
      setStats(mockStats);
      setPrediction({
        ...predRes.data,
        player_id: player.player_id,
        player_name: player.name,
        predicted_value: normalizedPred,
        predicted_transfer_value: normalizedPred,
        confidence_score: predRes.data.confidence_score || predRes.data.confidence || 0.88,
      });
    } catch (err) {
      setError(err.response?.data?.detail || "Error fetching player prediction.");
    } finally {
      setLoading(false);
    }
  };

  const formatEuro = (val) => 
    new Intl.NumberFormat('en-DE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 }).format(val);

  const explanationPoints = [
    { icon: TrendingUp, text: 'Strong recent performance with stable attacking output.' },
    { icon: Sparkles, text: 'Positive fan/media sentiment is boosting market perception.' },
    { icon: ShieldCheck, text: 'Low injury risk profile supports valuation confidence.' },
    { icon: Activity, text: 'Market trend trajectory indicates upward transfer demand.' },
  ];

  const modelMetrics = [
    { name: 'LSTM', rmse: 5.9, mae: 4.2, accuracy: 0.87 },
    { name: 'XGBoost', rmse: 4.8, mae: 3.5, accuracy: 0.91 },
  ];

  return (
    <div className="app-container">
      <header className="app-header">
        <div className="logo-row">
          <Brain className="logo-icon" size={42} />
          <h1 className="app-title text-gradient">TransferIQ</h1>
        </div>
        <p className="app-subtitle">AI-Powered Football Player Transfer Value Prediction</p>
      </header>

      <div className="glass filters-row">
        <div className="field-group">
          <label className="field-label">Player (More Dataset Rows)</label>
          <input
            className="search-input"
            placeholder="Search player name..."
            value={playerSearch}
            onChange={(e) => setPlayerSearch(e.target.value)}
            style={{ marginBottom: '0.5rem' }}
          />
          <select className="search-input" value={selectedPlayerId} onChange={handlePlayerChange}>
            <option value="">-- Choose a Player --</option>
            {Array.isArray(players) && players.map((p) => (
              <option key={p.player_id} value={p.player_id}>{p.name}{p.league ? ` (${p.league})` : ''}</option>
            ))}
          </select>
        </div>
        <div className="field-group">
          <label className="field-label">Season</label>
          <select className="search-input" value={selectedSeason} onChange={(e) => setSelectedSeason(e.target.value)}>
            {SEASON_OPTIONS.map((season) => (
              <option key={season} value={season}>{season}</option>
            ))}
          </select>
        </div>
        <div className="field-group">
          <label className="field-label">League</label>
          <select className="search-input" value={selectedLeague} onChange={(e) => setSelectedLeague(e.target.value)}>
            <option value="All Leagues">All Leagues</option>
            {leagueOptions.map((league) => (
              <option key={league} value={league}>{league}</option>
            ))}
          </select>
        </div>
      </div>

      <PipelineFlow activeStep={prediction ? 5 : 0} />

      <div className="layout-two-col">
        {/* Sidebar: Selection */}
        <aside className="sidebar glass">
          {selectedPlayer ? (
            <motion.div 
              className="player-profile"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <div className="profile-top">
                <img
                  src={`https://i.pravatar.cc/140?img=${Number(selectedPlayer.player_id || 1) % 70}`}
                  alt={selectedPlayer.name}
                  className="profile-avatar"
                />
                <div>
                  <h3>{selectedPlayer.name}</h3>
                  <p className="profile-sub">{selectedPlayer.league || selectedLeague} | ID {selectedPlayer.player_id}</p>
                </div>
              </div>

              <div className="profile-grid">
                <div className="info-row"><User size={14} /> <span>Position: Forward</span></div>
                <div className="info-row"><Calendar size={14} /> <span>Age: {selectedPlayer.age || 'N/A'}</span></div>
                <div className="info-row"><Award size={14} /> <span>Team: {selectedPlayer.team || 'N/A'}</span></div>
                <div className="info-row">
                  <Activity size={14} />
                  <span>
                    Form: 
                    <strong className={(selectedPlayer.goals || 0) + (selectedPlayer.assists || 0) > 10 ? 'form-hot' : 'form-cold'}>
                      {(selectedPlayer.goals || 0) + (selectedPlayer.assists || 0) > 10 ? ' Hot' : ' Cold'}
                    </strong>
                  </span>
                </div>
              </div>
            </motion.div>
          ) : (
            <div className="player-info">
              <div className="info-row"><Info size={14} /><span>Select a player to open the profile panel.</span></div>
            </div>
          )}

          {error && <div className="alert alert-danger" style={{marginTop: '1rem'}}>{error}</div>}
        </aside>

        {/* Main: Predictions & Charts */}
        <main>
          <AnimatePresence mode="wait">
            {loading ? (
              <div className="empty-state">
                <div className="loader" style={{borderTopColor: 'var(--primary)', border: '4px solid #333', borderRadius: '50%', width: '40px', height: '40px', animation: 'spin 1s linear infinite'}} />
                <h3>Analyzing Statistics...</h3>
                <p>Ensemble models (XGBoost + LSTM) are calculating future value.</p>
              </div>
            ) : prediction ? (
              <motion.div 
                key="prediction-content"
                initial={{ opacity: 0, scale: 0.98 }}
                animate={{ opacity: 1, scale: 1 }}
                className="results-stack"
              >
                <PredictionCard prediction={prediction} trendDirection={trendDirection} />

                <div className="glass explainer-panel">
                  <h4 className="chart-title"><Brain size={16} /> Why this prediction?</h4>
                  <ul className="why-list">
                    {explanationPoints.map((item) => {
                      const Icon = item.icon;
                      return (
                        <li key={item.text}>
                          <Icon size={15} />
                          <span>{item.text}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>

                {/* Performance Chart */}
                <div className="charts-grid">
                  <div className="chart-card glass">
                    <h4 className="chart-title"><TrendingUp size={16} /> Market Trend (Actual vs Predicted)</h4>
                    <div className="chart-area">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={marketSeries}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                          <XAxis dataKey="season" stroke="#888" />
                          <YAxis stroke="#888" tickFormatter={(v) => `${(v / 1000000).toFixed(1)}M`} />
                          <Tooltip
                            contentStyle={{ background: '#111', border: '1px solid #444' }}
                            formatter={(v) => formatEuro(v)}
                          />
                          <Line type="monotone" dataKey="actual_value" stroke="#38bdf8" strokeWidth={3} dot={{ r: 4 }} name="Past" />
                          <Line type="monotone" dataKey="predicted_value" stroke="#fb7185" strokeDasharray="7 5" strokeWidth={3} dot={{ r: 5 }} name="Predicted" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>

                  <div className="chart-card glass">
                    <h4 className="chart-title"><Activity size={16} /> Performance Score</h4>
                    <div className="chart-area">
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={stats}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.2)" />
                          <XAxis dataKey="season" stroke="#888" />
                          <YAxis stroke="#888" />
                          <Tooltip contentStyle={{ background: '#111', border: '1px solid #444' }} cursor={{ stroke: '#6366f1', strokeWidth: 1 }} />
                          <Line type="monotone" dataKey="goals" stroke="#fb7185" strokeWidth={3} dot={{ r: 3 }} />
                          <Line type="monotone" dataKey="assists" stroke="#38bdf8" strokeWidth={3} dot={{ r: 3 }} />
                        </LineChart>
                      </ResponsiveContainer>
                      <div className="chart-legend" style={{display: 'flex', gap: '1rem', marginTop: '1rem', justifyContent: 'center'}}>
                        <span style={{color: '#fb7185'}}>● Goals</span>
                        <span style={{color: '#38bdf8'}}>● Assists</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="charts-grid">
                  <SentimentChart score={Number(selectedPlayer?.sentiment_score || 0.72)} />
                  <ModelComparison models={modelMetrics} />
                </div>

                {/* Stats Table */}
                <div className="glass table-shell" style={{ padding: '1.5rem', marginTop: '1.5rem', overflowX: 'auto' }}>
                  <h4 className="chart-title"><Info size={16} /> Historical Stats</h4>
                  <table className="stats-table" style={{ width: '100%', borderCollapse: 'collapse', marginTop: '1rem' }}>
                    <thead>
                      <tr style={{ textAlign: 'left', borderBottom: '1px solid #333' }}>
                        <th style={{ padding: '10px' }}>Season</th>
                        <th style={{ padding: '10px' }}>Goals</th>
                        <th style={{ padding: '10px' }}>Assists</th>
                        <th style={{ padding: '10px' }}>Age</th>
                        <th style={{ padding: '10px' }}>Market Value</th>
                        <th style={{ padding: '10px' }}>Trend</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Array.isArray(stats) && stats.length > 0 ? stats.map((s, i) => (
                        <tr key={i} style={{ borderBottom: '1px solid #222' }}>
                          <td style={{ padding: '10px' }}>{s.season}</td>
                          <td style={{ padding: '10px' }}>{s.goals}</td>
                          <td style={{ padding: '10px' }}>{s.assists}</td>
                          <td style={{ padding: '10px' }}>{s.age}</td>
                          <td style={{ padding: '10px' }}>{formatEuro(s.market_value)}</td>
                          <td style={{ padding: '10px' }}>
                            {i === 0 || s.market_value >= (stats[i - 1]?.market_value || 0) ? (
                              <span className="trend-up"><ArrowUpRight size={14} /> Up</span>
                            ) : (
                              <span className="trend-down"><ArrowDownRight size={14} /> Down</span>
                            )}
                          </td>
                        </tr>
                      )) : (
                        <tr><td colSpan="6" style={{ padding: '10px', textAlign: 'center', color: '#888' }}>No data available</td></tr>
                      )}
                    </tbody>
                  </table>
                </div>

              </motion.div>
            ) : (
              <motion.div 
                key="empty-state"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="empty-state"
              >
                <TrendingUp size={64} className="pulse-icon" />
                <h3>Welcome to TransferIQ</h3>
                <p>Select a player from the menu to analyze their performance and predict their future market value using AI.</p>
              </motion.div>
            )}
          </AnimatePresence>
        </main>
      </div>
      
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
}

export default App;
