import { useState, useEffect } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from "recharts";
import "./App.css";

function App() {
  const [players, setPlayers] = useState([]);
  const [player, setPlayer] = useState(null);
  const [hybrid, setHybrid] = useState(null);
  const [season, setSeason] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [topPlayers, setTopPlayers] = useState([]);
  const [score, setScore] = useState(null);
  const [sentimentTrend, setSentimentTrend] = useState([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const [similarPlayers, setSimilarPlayers] = useState([]);
  const [playerRank, setPlayerRank] = useState(null);
  const [selectedSeason, setSelectedSeason] = useState("all");
  
  const [loading, setLoading] = useState({
    players: true,
    player: false,
    topPlayers: true,
    hybrid: false
  });
  
  const [error, setError] = useState(null);
  const [selectedPlayer, setSelectedPlayer] = useState("");

  const API_BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:5000";

  useEffect(() => {
    document.body.className = darkMode ? "dark" : "light";
  }, [darkMode]);

  useEffect(() => {
    fetchInitialData();
  }, []);

  useEffect(() => {
    if (player && topPlayers.length > 0) {
      calculateSimilarPlayers();
      calculatePlayerRank();
    }
  }, [player, topPlayers]);

  const fetchInitialData = async () => {
    try {
      const [playersRes, topPlayersRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/players`),
        axios.get(`${API_BASE_URL}/topplayers`)
      ]);
      
      setPlayers(playersRes.data);
      setTopPlayers(topPlayersRes.data);
      setLoading(prev => ({ ...prev, players: false, topPlayers: false }));
      
      const sentimentRes = await axios.get(`${API_BASE_URL}/sentiment_trend`);
      setSentimentTrend(sentimentRes.data);
    } catch (err) {
      setError("Failed to load initial data");
      setLoading(prev => ({ ...prev, players: false, topPlayers: false }));
    }
  };

  const loadPlayer = async (playerName) => {
    if (!playerName) return;
    
    setSelectedPlayer(playerName);
    setLoading(prev => ({ ...prev, player: true, hybrid: true }));
    setError(null);
    setSelectedSeason("all");
    
    try {
      const [playerRes, seasonRes, forecastRes, scoreRes, hybridRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/player/${encodeURIComponent(playerName)}`),
        axios.get(`${API_BASE_URL}/season/${encodeURIComponent(playerName)}`),
        axios.get(`${API_BASE_URL}/forecast/${encodeURIComponent(playerName)}`),
        axios.get(`${API_BASE_URL}/score/${encodeURIComponent(playerName)}`),
        axios.get(`${API_BASE_URL}/hybrid/${encodeURIComponent(playerName)}`)
      ]);
      
      setPlayer(playerRes.data);
      setSeason(seasonRes.data);
      setForecast(forecastRes.data);
      setScore(scoreRes.data.score);
      setHybrid(hybridRes.data);
    } catch (err) {
      setError(`Failed to load data for ${playerName}`);
    } finally {
      setLoading(prev => ({ ...prev, player: false, hybrid: false }));
    }
  };

  const calculateSimilarPlayers = () => {
    if (!player || topPlayers.length === 0) return;
    
    const playerValue = player.market_value;
    const tolerance = playerValue * 0.2;
    
    const similar = topPlayers
      .filter(p => p.player_name !== player.player)
      .filter(p => Math.abs(p.market_value_eur - playerValue) <= tolerance)
      .slice(0, 5)
      .map(p => ({
        name: p.player_name,
        value: p.market_value_eur,
        similarity: Math.round(100 - (Math.abs(p.market_value_eur - playerValue) / playerValue * 100))
      }));
    
    setSimilarPlayers(similar);
  };

  const calculatePlayerRank = () => {
    if (!player || topPlayers.length === 0) return;
    const rank = topPlayers.findIndex(p => p.player_name === player.player) + 1;
    setPlayerRank(rank || null);
  };

  const getTrend = () => {
    if (season.length < 2) return "stable";
    const last = season[season.length - 1].market_value_eur;
    const prev = season[season.length - 2].market_value_eur;
    if (last > prev) return "up";
    if (last < prev) return "down";
    return "stable";
  };

  const exportReport = () => {
    if (!player) return;
    const report = {
      player: player,
      prediction: hybrid?.hybrid_prediction,
      score: score,
      timestamp: new Date().toISOString()
    };
    const dataStr = JSON.stringify(report, null, 2);
    const dataUri = 'data:application/json;charset=utf-8,' + encodeURIComponent(dataStr);
    const linkElement = document.createElement('a');
    linkElement.setAttribute('href', dataUri);
    linkElement.setAttribute('download', `${player.player.replace(/\s/g, '_')}_report.json`);
    linkElement.click();
  };

  const formatCurrency = (value) => {
    if (!value) return "N/A";
    if (value >= 1000000000) return `€${(value / 1000000000).toFixed(1)}B`;
    if (value >= 1000000) return `€${(value / 1000000).toFixed(1)}M`;
    if (value >= 1000) return `€${(value / 1000).toFixed(0)}K`;
    return `€${Math.round(value)}`;
  };

  const getScoreColor = (score) => {
    if (score >= 80) return "#10b981";
    if (score >= 60) return "#f59e0b";
    if (score >= 40) return "#f97316";
    return "#ef4444";
  };

  const getSentimentColor = (score) => {
    if (score >= 0.2) return "#10b981";
    if (score >= -0.2) return "#f59e0b";
    return "#ef4444";
  };

  const getFilteredSeasonData = () => {
    if (selectedSeason === "all") return season;
    return season.filter(s => s.season === selectedSeason);
  };

  const filteredPlayers = players.filter(p => 
    p.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const trend = getTrend();
  const trendIcon = trend === "up" ? "▲" : trend === "down" ? "▼" : "●";
  const trendText = trend === "up" ? "RISING" : trend === "down" ? "FALLING" : "STABLE";

  const sentimentRadarData = player ? [
    { metric: "Sentiment Score", value: Math.max(0, Math.min(100, ((player.sentiment_score || 0) + 1) * 50)) },
    { metric: "Positive", value: Math.min(100, ((player.positive_tweets || 0) / ((player.total_tweets || 1)) * 100)) },
    { metric: "Engagement", value: Math.min(100, (player.tweet_engagement_rate || 0)) },
    { metric: "Neutral", value: Math.min(100, ((player.neutral_count || 0) / ((player.total_tweets || 1)) * 100)) },
    { metric: "Negative", value: Math.min(100, ((player.negative_tweets || 0) / ((player.total_tweets || 1)) * 100)) }
  ] : [];

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-left">
          <h1>⚽ TransferIQ</h1>
          <p>Player Market Value Prediction System</p>
        </div>
        <div className="header-right">
          <button className="theme-toggle" onClick={() => setDarkMode(!darkMode)}>
            {darkMode ? "☀️ Light" : "🌙 Dark"}
          </button>
        </div>
      </header>

      <div className="main-layout">
        <aside className="left-sidebar">
          <div className="sidebar-card">
            <h3>🔍 Search Players</h3>
            <input 
              type="text" 
              placeholder="Search by name..." 
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="search-input"
            />
          </div>

          <div className="sidebar-card">
            <h3>📋 Select Player ({filteredPlayers.length} players)</h3>
            <div className="player-list-scroll">
              <select 
                onChange={(e) => loadPlayer(e.target.value)} 
                value={selectedPlayer}
                disabled={loading.players}
                className="player-dropdown"
                size="8"
              >
                <option value="">-- Choose a player --</option>
                {filteredPlayers.map(p => (
                  <option key={p} value={p}>{p}</option>
                ))}
              </select>
            </div>
          </div>

          {player && (
            <div className="sidebar-card player-info">
              <h3>Player Details</h3>
              <div className="player-name-large">{player.player}</div>
              <div className="player-team">{player.team} | {player.position} | Age {player.age}</div>
              <div className="player-value-large">{formatCurrency(player.market_value)}</div>
              <div className={`trend-indicator ${trend}`}>
                {trendIcon} {trendText}
              </div>
              {playerRank && (
                <div className="rank-container">
                  <div className="rank-label">Market Rank</div>
                  <div className="rank-bar">
                    <div className="rank-fill" style={{ width: `${((players.length - playerRank) / players.length) * 100}%` }}></div>
                  </div>
                  <div className="rank-number">#{playerRank} out of {players.length} players</div>
                </div>
              )}
              <div className="player-stats-row">
                <div className="stat-item"><span>⚽ Goals</span><strong>{player.goals || 0}</strong></div>
                <div className="stat-item"><span>🎯 Assists</span><strong>{player.assists || 0}</strong></div>
              </div>
              <button className="export-btn" onClick={exportReport}>📄 Export Report</button>
            </div>
          )}

          {season.length > 0 && (
            <div className="sidebar-card">
              <h3>📅 Season Filter</h3>
              <select value={selectedSeason} onChange={(e) => setSelectedSeason(e.target.value)} className="player-dropdown">
                <option value="all">All Seasons</option>
                {season.map(s => (<option key={s.season} value={s.season}>{s.season}</option>))}
              </select>
            </div>
          )}

          {score !== null && (
            <div className="sidebar-card score-card">
              <h3>⭐ Transfer Score</h3>
              <div className="score-circle" style={{ background: `conic-gradient(${getScoreColor(score)} 0deg ${score * 3.6}deg, #2d2d2d ${score * 3.6}deg 360deg)` }}>
                <div className="score-inner"><span className="score-number">{score}</span></div>
              </div>
              <div className="score-label" style={{ color: getScoreColor(score) }}>
                {score >= 80 ? "Elite Player" : score >= 60 ? "Strong Player" : score >= 40 ? "Average" : "Needs Development"}
              </div>
            </div>
          )}
        </aside>

        <main className="right-content">
          {error && <div className="error-message">⚠️ {error}</div>}
          {loading.player && <div className="loading-state"><div className="spinner"></div><p>Loading player data...</p></div>}

          {!loading.player && selectedPlayer && (
            <>
              {hybrid && (
                <div className="main-prediction-card">
                  <div className="prediction-badge">AI PREDICTION</div>
                  <div className="prediction-value-main">{formatCurrency(hybrid.hybrid_prediction)}</div>
                  <div className="prediction-range">
                    <span>Best Case: {formatCurrency(hybrid.hybrid_prediction * 1.15)}</span>
                    <span>Worst Case: {formatCurrency(hybrid.hybrid_prediction * 0.85)}</span>
                  </div>
                  <div className="prediction-note">🤖 Stacking Ensemble: LSTM + XGBoost</div>
                </div>
              )}

              <div className="charts-section">
                {season.length > 0 && (
                  <div className="chart-card">
                    <h3>Historical Market Value Trend</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={getFilteredSeasonData()}>
                        <XAxis dataKey="season" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" tickFormatter={(v) => formatCurrency(v)} />
                        <Tooltip formatter={(v) => formatCurrency(v)} />
                        <Line type="monotone" dataKey="market_value_eur" stroke="#3b82f6" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {forecast.length > 0 && (
                  <div className="chart-card">
                    <h3>Future Value Forecast</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={forecast}>
                        <XAxis dataKey="season" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" tickFormatter={(v) => formatCurrency(v)} />
                        <Tooltip formatter={(v) => formatCurrency(v)} />
                        <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {player && (
                  <div className="chart-card">
                    <h3>Performance Radar</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <RadarChart data={[
                        { metric: "Goals", value: Math.min(100, (player.goals || 0) * 8) },
                        { metric: "Assists", value: Math.min(100, (player.assists || 0) * 8) },
                        { metric: "Value", value: Math.min(100, (player.market_value || 0) / 1000000) },
                        { metric: "Experience", value: player.age ? Math.min(100, player.age * 3) : 50 }
                      ]}>
                        <PolarGrid stroke="#334155" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8' }} />
                        <PolarRadiusAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
                        <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              <div className="sentiment-section">
                <h3>Social Media Sentiment Analysis</h3>
                <div className="sentiment-grid">
                  <div className="sentiment-card">
                    <h4>Overall Sentiment Score</h4>
                    <div className="sentiment-score-large" style={{ color: getSentimentColor(player?.sentiment_score || 0) }}>
                      {(player?.sentiment_score || 0).toFixed(3)}
                    </div>
                    <div className="sentiment-label">
                      {player?.sentiment_score >= 0.2 ? "Very Positive" : 
                       player?.sentiment_score >= 0.05 ? "Positive" :
                       player?.sentiment_score <= -0.2 ? "Very Negative" :
                       player?.sentiment_score <= -0.05 ? "Negative" : "Neutral"}
                    </div>
                  </div>

                  <div className="sentiment-card">
                    <h4>Sentiment Distribution</h4>
                    <div className="sentiment-bars">
                      <div className="sentiment-bar"><span>Positive</span>
                        <div className="bar-container"><div className="bar-fill positive" style={{ width: `${((player?.positive_tweets || 0) / ((player?.total_tweets || 1))) * 100}%` }}></div>
                        <span>{Math.round(((player?.positive_tweets || 0) / ((player?.total_tweets || 1))) * 100)}%</span></div>
                      </div>
                      <div className="sentiment-bar"><span>Neutral</span>
                        <div className="bar-container"><div className="bar-fill neutral" style={{ width: `${((player?.neutral_count || 0) / ((player?.total_tweets || 1))) * 100}%` }}></div>
                        <span>{Math.round(((player?.neutral_count || 0) / ((player?.total_tweets || 1))) * 100)}%</span></div>
                      </div>
                      <div className="sentiment-bar"><span>Negative</span>
                        <div className="bar-container"><div className="bar-fill negative" style={{ width: `${((player?.negative_tweets || 0) / ((player?.total_tweets || 1))) * 100}%` }}></div>
                        <span>{Math.round(((player?.negative_tweets || 0) / ((player?.total_tweets || 1))) * 100)}%</span></div>
                      </div>
                    </div>
                  </div>

                  <div className="sentiment-card">
                    <h4>Social Engagement</h4>
                    <div className="engagement-metrics">
                      <div className="metric"><span>Total Tweets</span><strong>{(player?.total_tweets || 0).toLocaleString()}</strong></div>
                      <div className="metric"><span>Total Likes</span><strong>{(player?.total_likes || 0).toLocaleString()}</strong></div>
                      <div className="metric"><span>Engagement Rate</span><strong>{(player?.tweet_engagement_rate || 0).toFixed(1)}%</strong></div>
                    </div>
                  </div>
                </div>

                {player && (
                  <div className="chart-card">
                    <h3>Sentiment Analysis Radar</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <RadarChart data={sentimentRadarData}>
                        <PolarGrid stroke="#334155" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8' }} />
                        <PolarRadiusAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
                        <Radar dataKey="value" stroke="#10b981" fill="#10b981" fillOpacity={0.5} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {sentimentTrend.length > 0 && (
                  <div className="chart-card">
                    <h3>Sentiment Trend Over Seasons</h3>
                    <ResponsiveContainer width="100%" height={280}>
                      <LineChart data={sentimentTrend}>
                        <XAxis dataKey="season" stroke="#94a3b8" />
                        <YAxis stroke="#94a3b8" domain={[-1, 1]} />
                        <Tooltip formatter={(v) => v.toFixed(3)} />
                        <Line type="monotone" dataKey="sentiment" stroke="#10b981" strokeWidth={2} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {similarPlayers.length > 0 && (
                <div className="table-container">
                  <h3>Similar Players (by Market Value)</h3>
                  <table className="players-table"><thead><tr><th>Player</th><th>Value</th><th>Similarity</th><th></th></tr></thead>
                  <tbody>{similarPlayers.map(p => (
                    <tr key={p.name} onClick={() => loadPlayer(p.name)}>
                      <td>{p.name}</td><td>{formatCurrency(p.value)}</td><td><span className="similarity-badge">{p.similarity}%</span></td>
                      <td><button className="view-button">View</button></td>
                    </tr>
                  ))}</tbody></table>
                </div>
              )}

              <div className="table-container">
                <h3>Top 10 Players by Market Value</h3>
                <table className="players-table"><thead><tr><th>#</th><th>Player</th><th>Team</th><th>Value</th><th></th></tr></thead>
                <tbody>{topPlayers.slice(0, 10).map((p, idx) => (
                  <tr key={p.player_name} onClick={() => loadPlayer(p.player_name)}>
                    <td>{idx + 1}</td><td>{p.player_name}</td><td>{p.team}</td><td>{formatCurrency(p.market_value_eur)}</td>
                    <td><button className="view-button">View</button></td>
                  </tr>
                ))}</tbody></table>
              </div>
            </>
          )}

          {!selectedPlayer && !loading.player && (
            <div className="welcome-message">
              <div className="welcome-icon">⚽</div>
              <h2>Welcome to TransferIQ</h2>
              <p>Select a player from the left sidebar to see AI-powered predictions</p>
              <div className="features-grid-small">
                <div className="feature">🤖 LSTM + XGBoost Stacking</div>
                <div className="feature">📊 Real-time Sentiment Analysis</div>
                <div className="feature">📈 5-Year Forecast</div>
                <div className="feature">⭐ Transfer Score</div>
              </div>
            </div>
          )}

          <div className="footer">
            <small>Powered by LSTM + XGBoost Stacking Ensemble</small>
            <small>Sentiment Analysis from Social Media Data</small>
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;