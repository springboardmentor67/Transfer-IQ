"""
=============================================================
LSTM Models for TransferIQ Player Market Value Prediction
=============================================================
Implements:
  1. Univariate LSTM        - Predict next-season value from value history
  2. Multivariate LSTM      - Predict value using multiple features
  3. Encoder-Decoder LSTM   - Multi-step forecasting (predict 2 future seasons)

All models built from scratch using NumPy only.
=============================================================
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings, os
warnings.filterwarnings('ignore')

np.random.seed(42)
OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# CORE LSTM CELL (pure NumPy)
# ─────────────────────────────────────────────
class LSTMCell:
    """Single LSTM cell with forget/input/output gates."""
    def __init__(self, input_size, hidden_size):
        self.H = hidden_size
        I = input_size + hidden_size
        scale = 0.1
        # Gates: forget, input, candidate cell, output
        self.Wf = np.random.randn(I, hidden_size) * scale
        self.Wi = np.random.randn(I, hidden_size) * scale
        self.Wc = np.random.randn(I, hidden_size) * scale
        self.Wo = np.random.randn(I, hidden_size) * scale
        self.bf = np.zeros((1, hidden_size))
        self.bi = np.zeros((1, hidden_size))
        self.bc = np.zeros((1, hidden_size))
        self.bo = np.zeros((1, hidden_size))

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def tanh(x):    return np.tanh(np.clip(x, -15, 15))

    def forward(self, x, h_prev, c_prev):
        combined = np.concatenate([x, h_prev], axis=1)
        f = self.sigmoid(combined @ self.Wf + self.bf)
        i = self.sigmoid(combined @ self.Wi + self.bi)
        c_tilde = self.tanh(combined @ self.Wc + self.bc)
        o = self.sigmoid(combined @ self.Wo + self.bo)
        c = f * c_prev + i * c_tilde
        h = o * self.tanh(c)
        return h, c, (combined, f, i, c_tilde, o, c, h_prev, c_prev)

    def backward(self, dh, dc, cache, lr):
        combined, f, i, c_tilde, o, c, h_prev, c_prev = cache
        tanh_c = self.tanh(c)
        do     = dh * tanh_c
        dc    += dh * o * (1 - tanh_c**2)
        df     = dc * c_prev
        dc_tilde = dc * i
        di     = dc * c_tilde
        dc_prev= dc * f

        # Gate deltas (pre-activation)
        do_pre = do * o * (1 - o)
        df_pre = df * f * (1 - f)
        di_pre = di * i * (1 - i)
        dc_pre = dc_tilde * (1 - c_tilde**2)

        # Gradients
        for gate_pre, W, b, attr_W, attr_b in [
            (df_pre, self.Wf, self.bf, 'Wf', 'bf'),
            (di_pre, self.Wi, self.bi, 'Wi', 'bi'),
            (dc_pre, self.Wc, self.bc, 'Wc', 'bc'),
            (do_pre, self.Wo, self.bo, 'Wo', 'bo'),
        ]:
            dW = combined.T @ gate_pre
            db = gate_pre.sum(axis=0, keepdims=True)
            setattr(self, attr_W, getattr(self, attr_W) - lr * np.clip(dW, -1, 1))
            setattr(self, attr_b, getattr(self, attr_b) - lr * np.clip(db, -1, 1))

        dcombined = df_pre @ self.Wf.T + di_pre @ self.Wi.T + \
                    dc_pre @ self.Wc.T + do_pre @ self.Wo.T
        # combined = [x (input_size), h_prev (H)] -> h_prev gradient starts after input
        input_size = dcombined.shape[1] - self.H
        dh_prev = dcombined[:, input_size:]
        return dh_prev, dc_prev


# ─────────────────────────────────────────────
# 1.  UNIVARIATE LSTM
# ─────────────────────────────────────────────
class UnivariateLSTM:
    """Single-feature LSTM to predict next-season market value."""
    def __init__(self, hidden_size=32, lr=0.005):
        self.cell = LSTMCell(1, hidden_size)
        self.H    = hidden_size
        self.Wy   = np.random.randn(hidden_size, 1) * 0.1
        self.by   = np.zeros((1, 1))
        self.lr   = lr
        self.losses = []

    def _forward_seq(self, X_seq):
        """X_seq: (T, 1)"""
        T = X_seq.shape[0]
        h = np.zeros((1, self.H)); c = np.zeros((1, self.H))
        caches = []
        for t in range(T):
            h, c, cache = self.cell.forward(X_seq[t:t+1], h, c)
            caches.append(cache)
        y_hat = h @ self.Wy + self.by
        return y_hat, h, caches

    def train(self, X, y, epochs=80):
        """X: (N, T, 1), y: (N, 1)"""
        N = X.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            idx = np.random.permutation(N)
            for n in idx:
                y_hat, h, caches = self._forward_seq(X[n])
                y_n = y[n].reshape(1,1)
                loss = float(np.mean((y_hat - y_n)**2))
                epoch_loss += loss
                dy = 2 * (y_hat - y_n)
                dWy = h.T @ dy
                dby = dy
                self.Wy -= self.lr * np.clip(dWy, -1, 1)
                self.by -= self.lr * np.clip(dby, -1, 1)
                dh = dy @ self.Wy.T  # (1, H)
                dc = np.zeros((1, self.H))
                for cache in reversed(caches):
                    dh, dc = self.cell.backward(dh, dc, cache, self.lr)
            self.losses.append(epoch_loss / N)
            if (epoch+1) % 20 == 0:
                print(f"  [Univariate] Epoch {epoch+1}/{epochs}  Loss={epoch_loss/N:.6f}")

    def predict(self, X):
        preds = []
        for n in range(X.shape[0]):
            y_hat, _, _ = self._forward_seq(X[n])
            preds.append(float(y_hat.flat[0]))
        return np.array(preds)


# ─────────────────────────────────────────────
# 2.  MULTIVARIATE LSTM
# ─────────────────────────────────────────────
class MultivariateLSTM:
    """Multi-feature LSTM to predict market value per player."""
    def __init__(self, input_size, hidden_size=48, lr=0.003):
        self.cell = LSTMCell(input_size, hidden_size)
        self.H    = hidden_size
        self.Wy   = np.random.randn(hidden_size, 1) * 0.1
        self.by   = np.zeros((1, 1))
        self.lr   = lr
        self.losses = []

    def _forward_seq(self, X_seq):
        T = X_seq.shape[0]
        h = np.zeros((1, self.H)); c = np.zeros((1, self.H))
        caches = []
        for t in range(T):
            h, c, cache = self.cell.forward(X_seq[t:t+1], h, c)
            caches.append(cache)
        y_hat = h @ self.Wy + self.by
        return y_hat, h, caches

    def train(self, X, y, epochs=80):
        N = X.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            idx = np.random.permutation(N)
            for n in idx:
                y_hat, h, caches = self._forward_seq(X[n])
                y_n = y[n].reshape(1,1)
                loss = float(np.mean((y_hat - y_n)**2))
                epoch_loss += loss
                dy = 2 * (y_hat - y_n)
                dWy = h.T @ dy
                dby = dy
                self.Wy -= self.lr * np.clip(dWy, -1, 1)
                self.by -= self.lr * np.clip(dby, -1, 1)
                dh = dy @ self.Wy.T   # (1, H)
                dc = np.zeros((1, self.H))
                for cache in reversed(caches):
                    dh, dc = self.cell.backward(dh, dc, cache, self.lr)
            self.losses.append(epoch_loss / N)
            if (epoch+1) % 20 == 0:
                print(f"  [Multivariate] Epoch {epoch+1}/{epochs}  Loss={epoch_loss/N:.6f}")

    def predict(self, X):
        preds = []
        for n in range(X.shape[0]):
            y_hat, _, _ = self._forward_seq(X[n])
            preds.append(float(y_hat.flat[0]))
        return np.array(preds)


# ─────────────────────────────────────────────
# 3.  ENCODER-DECODER LSTM  (multi-step)
# ─────────────────────────────────────────────
class EncoderDecoderLSTM:
    """
    Encoder reads historical sequence.
    Decoder autoregressively generates forecast_steps outputs.
    """
    def __init__(self, input_size, hidden_size=48, forecast_steps=2, lr=0.003):
        self.enc_cell = LSTMCell(input_size, hidden_size)
        self.dec_cell = LSTMCell(1, hidden_size)
        self.H    = hidden_size
        self.F    = forecast_steps
        self.Wy   = np.random.randn(hidden_size, 1) * 0.1
        self.by   = np.zeros((1, 1))
        self.lr   = lr
        self.losses = []

    def _encode(self, X_seq):
        h = np.zeros((1, self.H)); c = np.zeros((1, self.H))
        enc_caches = []
        for t in range(X_seq.shape[0]):
            h, c, cache = self.enc_cell.forward(X_seq[t:t+1], h, c)
            enc_caches.append(cache)
        return h, c, enc_caches

    def _decode(self, h, c, seed_val, steps):
        dec_caches = []; preds = []
        inp = np.array([[seed_val]])
        for _ in range(steps):
            h, c, cache = self.dec_cell.forward(inp, h, c)
            dec_caches.append(cache)
            y_hat = float((h @ self.Wy + self.by).flat[0])
            preds.append(y_hat)
            inp = np.array([[y_hat]])
        return np.array(preds), h, dec_caches

    def train(self, X, y, epochs=80):
        """X: (N, T, input_size), y: (N, forecast_steps)"""
        N = X.shape[0]
        for epoch in range(epochs):
            epoch_loss = 0
            idx = np.random.permutation(N)
            for n in idx:
                h, c, enc_caches = self._encode(X[n])
                seed = float(X[n, -1, 0])
                preds, h_last, dec_caches = self._decode(h, c, seed, self.F)
                loss = float(np.mean((preds - y[n])**2))
                epoch_loss += loss
                # Backprop decoder
                dh = np.zeros((1, self.H)); dc = np.zeros((1, self.H))
                for s in reversed(range(self.F)):
                    residual = preds[s] - y[n, s]
                    # recompute h at step s (approximate with final h_last for simplicity)
                    dy = np.array([[2 * residual / self.F]])
                    h_s = dec_caches[s][6]   # h_prev stored in cache
                    dWy = h_s.T @ dy
                    self.Wy -= self.lr * np.clip(dWy, -1, 1)
                    self.by -= self.lr * np.clip(dy, -1, 1)
                    dh += dy @ self.Wy.T
                    dh, dc = self.dec_cell.backward(dh, dc, dec_caches[s], self.lr)
                # Backprop encoder
                for cache in reversed(enc_caches):
                    dh, dc = self.enc_cell.backward(dh, dc, cache, self.lr)
            self.losses.append(epoch_loss / N)
            if (epoch+1) % 20 == 0:
                print(f"  [Enc-Dec]    Epoch {epoch+1}/{epochs}  Loss={epoch_loss/N:.6f}")

    def predict(self, X):
        all_preds = []
        for n in range(X.shape[0]):
            h, c, _ = self._encode(X[n])
            seed = float(X[n, -1, 0])
            preds, _, _ = self._decode(h, c, seed, self.F)
            all_preds.append(preds)
        return np.array(all_preds)


# ─────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────
print("=" * 60)
print("  INFOSYS PROJECT: LSTM Player Market Value Prediction")
print("=" * 60)
print("\n[1] Loading & preparing data...")

df = pd.read_csv('/mnt/user-data/uploads/player_transfer_value_with_sentiment.csv')

SEASON_ORDER = {'2019/20':0,'2020/21':1,'2021/22':2,'2022/23':3,'2023/24':4}
df['season_idx'] = df['season'].map(SEASON_ORDER)
df = df.sort_values(['player_name','season_idx'])

MULTI_FEATURES = [
    'market_value_eur','current_age','matches','goals','assists',
    'minutes_played','goals_per90','assists_per90','pass_accuracy_pct',
    'injury_burden_index','availability_rate','vader_compound_score',
    'attacking_output_index','defensive_actions_per90'
]

# Keep only players with all 5 seasons
players_full = df.groupby('player_name').filter(lambda x: len(x)==5)
print(f"  Players with full 5-season history: {players_full['player_name'].nunique()}")

# ── Scalers ──
mv_scaler   = MinMaxScaler()
multi_scaler = MinMaxScaler()

all_mv     = players_full['market_value_eur'].values.reshape(-1,1)
mv_scaler.fit(all_mv)

all_multi  = players_full[MULTI_FEATURES].values
multi_scaler.fit(all_multi)

# ── Build sequences ──
def build_sequences(group_df, seq_len=3, forecast=1):
    """Returns X (N,seq_len,F), y (N,forecast)"""
    X_list, y_list = [], []
    n = len(group_df)
    for start in range(n - seq_len - forecast + 1):
        X_list.append(start)
        y_list.append(start + seq_len)
    return X_list, y_list

UNI_SEQ   = 3  # use 3 seasons to predict next
MULTI_SEQ = 3
ENC_SEQ   = 3
FORECAST  = 2  # encoder-decoder predicts 2 future seasons

X_uni, y_uni      = [], []
X_multi, y_multi  = [], []
X_enc,  y_enc     = [], []

for name, grp in players_full.groupby('player_name'):
    grp = grp.sort_values('season_idx').reset_index(drop=True)
    mv_scaled    = mv_scaler.transform(grp[['market_value_eur']].values)
    multi_scaled = multi_scaler.transform(grp[MULTI_FEATURES].values)

    # Univariate (seq_len=3 → predict t+1)
    for i in range(len(grp) - UNI_SEQ - 1 + 1):
        if i + UNI_SEQ < len(grp):
            X_uni.append(mv_scaled[i:i+UNI_SEQ].reshape(UNI_SEQ, 1))
            y_uni.append(mv_scaled[i+UNI_SEQ, 0:1])

    # Multivariate (seq_len=3 → predict t+1 market value)
    for i in range(len(grp) - MULTI_SEQ - 1 + 1):
        if i + MULTI_SEQ < len(grp):
            X_multi.append(multi_scaled[i:i+MULTI_SEQ])
            y_multi.append(mv_scaled[i+MULTI_SEQ, 0:1])

    # Encoder-Decoder (seq_len=3 → predict t+1, t+2)
    for i in range(len(grp) - ENC_SEQ - FORECAST + 1):
        if i + ENC_SEQ + FORECAST <= len(grp):
            X_enc.append(multi_scaled[i:i+ENC_SEQ])
            y_enc.append(mv_scaled[i+ENC_SEQ:i+ENC_SEQ+FORECAST, 0])

X_uni   = np.array(X_uni);   y_uni   = np.array(y_uni)
X_multi = np.array(X_multi); y_multi = np.array(y_multi)
X_enc   = np.array(X_enc);   y_enc   = np.array(y_enc)

print(f"  Univariate   sequences: X={X_uni.shape},   y={y_uni.shape}")
print(f"  Multivariate sequences: X={X_multi.shape}, y={y_multi.shape}")
print(f"  Enc-Dec      sequences: X={X_enc.shape},   y={y_enc.shape}")

# Train/test split (80/20)
def split(X, y, ratio=0.8):
    n = int(len(X)*ratio)
    return X[:n], X[n:], y[:n], y[n:]

X_u_tr, X_u_te, y_u_tr, y_u_te = split(X_uni,   y_uni)
X_m_tr, X_m_te, y_m_tr, y_m_te = split(X_multi, y_multi)
X_e_tr, X_e_te, y_e_tr, y_e_te = split(X_enc,   y_enc)

print(f"\n  Train/Test split (80/20):")
print(f"    Univariate  : Train={len(X_u_tr)}, Test={len(X_u_te)}")
print(f"    Multivariate: Train={len(X_m_tr)}, Test={len(X_m_te)}")
print(f"    Enc-Decoder : Train={len(X_e_tr)}, Test={len(X_e_te)}")


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
print("\n[2] Training Models...")
EPOCHS = 100

print("\n--- Model 1: Univariate LSTM ---")
uni_model = UnivariateLSTM(hidden_size=32, lr=0.004)
uni_model.train(X_u_tr, y_u_tr, epochs=EPOCHS)

print("\n--- Model 2: Multivariate LSTM ---")
multi_model = MultivariateLSTM(input_size=len(MULTI_FEATURES), hidden_size=48, lr=0.003)
multi_model.train(X_m_tr, y_m_tr, epochs=EPOCHS)

print("\n--- Model 3: Encoder-Decoder LSTM ---")
enc_model = EncoderDecoderLSTM(input_size=len(MULTI_FEATURES), hidden_size=48,
                                forecast_steps=FORECAST, lr=0.003)
enc_model.train(X_e_tr, y_e_tr, epochs=EPOCHS)


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
print("\n[3] Evaluating Models...")

def rmse(true, pred): return np.sqrt(mean_squared_error(true, pred))
def mae(true, pred):  return np.mean(np.abs(true - pred))

# ─ Univariate ─
u_pred_tr = uni_model.predict(X_u_tr)
u_pred_te = uni_model.predict(X_u_te)
u_pred_tr_inv = mv_scaler.inverse_transform(u_pred_tr.reshape(-1,1)).flatten()
u_pred_te_inv = mv_scaler.inverse_transform(u_pred_te.reshape(-1,1)).flatten()
y_u_tr_inv    = mv_scaler.inverse_transform(y_u_tr).flatten()
y_u_te_inv    = mv_scaler.inverse_transform(y_u_te).flatten()

u_rmse_tr = rmse(y_u_tr_inv, u_pred_tr_inv)
u_rmse_te = rmse(y_u_te_inv, u_pred_te_inv)
u_mae_te  = mae(y_u_te_inv,  u_pred_te_inv)

# ─ Multivariate ─
m_pred_tr = multi_model.predict(X_m_tr)
m_pred_te = multi_model.predict(X_m_te)
m_pred_tr_inv = mv_scaler.inverse_transform(m_pred_tr.reshape(-1,1)).flatten()
m_pred_te_inv = mv_scaler.inverse_transform(m_pred_te.reshape(-1,1)).flatten()
y_m_tr_inv    = mv_scaler.inverse_transform(y_m_tr).flatten()
y_m_te_inv    = mv_scaler.inverse_transform(y_m_te).flatten()

m_rmse_tr = rmse(y_m_tr_inv, m_pred_tr_inv)
m_rmse_te = rmse(y_m_te_inv, m_pred_te_inv)
m_mae_te  = mae(y_m_te_inv,  m_pred_te_inv)

# ─ Encoder-Decoder ─
e_pred_tr = enc_model.predict(X_e_tr)
e_pred_te = enc_model.predict(X_e_te)

def inv_enc(pred_2d):
    flat = pred_2d.reshape(-1,1)
    return mv_scaler.inverse_transform(flat).reshape(pred_2d.shape)

e_pred_tr_inv = inv_enc(e_pred_tr)
e_pred_te_inv = inv_enc(e_pred_te)
y_e_tr_inv    = inv_enc(y_e_tr)
y_e_te_inv    = inv_enc(y_e_te)

e_rmse_tr = rmse(y_e_tr_inv.flatten(), e_pred_tr_inv.flatten())
e_rmse_te = rmse(y_e_te_inv.flatten(), e_pred_te_inv.flatten())
e_mae_te  = mae(y_e_te_inv.flatten(),  e_pred_te_inv.flatten())

print(f"\n  ┌─────────────────────────────────────────────────────┐")
print(f"  │          PERFORMANCE METRICS SUMMARY                │")
print(f"  ├──────────────────┬──────────────┬────────────────────┤")
print(f"  │ Model            │ RMSE (Test)  │  MAE (Test)        │")
print(f"  ├──────────────────┼──────────────┼────────────────────┤")
print(f"  │ Univariate LSTM  │ €{u_rmse_te:>10,.0f} │ €{u_mae_te:>15,.0f}  │")
print(f"  │ Multivariate LSTM│ €{m_rmse_te:>10,.0f} │ €{m_mae_te:>15,.0f}  │")
print(f"  │ Encoder-Decoder  │ €{e_rmse_te:>10,.0f} │ €{e_mae_te:>15,.0f}  │")
print(f"  └──────────────────┴──────────────┴────────────────────┘")

metrics = {
    'Model': ['Univariate LSTM','Multivariate LSTM','Encoder-Decoder LSTM'],
    'RMSE_Train': [u_rmse_tr, m_rmse_tr, e_rmse_tr],
    'RMSE_Test':  [u_rmse_te, m_rmse_te, e_rmse_te],
    'MAE_Test':   [u_mae_te,  m_mae_te,  e_mae_te],
}
pd.DataFrame(metrics).to_csv(f"{OUTPUT_DIR}/metrics_summary.csv", index=False)


# ─────────────────────────────────────────────
# SAVE PREDICTIONS
# ─────────────────────────────────────────────
pred_uni = pd.DataFrame({
    'actual_market_value': y_u_te_inv,
    'predicted_market_value': u_pred_te_inv,
    'error_eur': u_pred_te_inv - y_u_te_inv
})
pred_uni.to_csv(f"{OUTPUT_DIR}/predictions_univariate.csv", index=False)

pred_multi = pd.DataFrame({
    'actual_market_value': y_m_te_inv,
    'predicted_market_value': m_pred_te_inv,
    'error_eur': m_pred_te_inv - y_m_te_inv
})
pred_multi.to_csv(f"{OUTPUT_DIR}/predictions_multivariate.csv", index=False)

enc_rows = []
for i in range(len(y_e_te_inv)):
    for step in range(FORECAST):
        enc_rows.append({
            'sample': i,
            'forecast_step': step+1,
            'actual': y_e_te_inv[i, step],
            'predicted': e_pred_te_inv[i, step],
            'error_eur': e_pred_te_inv[i, step] - y_e_te_inv[i, step]
        })
pd.DataFrame(enc_rows).to_csv(f"{OUTPUT_DIR}/predictions_encoder_decoder.csv", index=False)


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────
print("\n[4] Generating visualisations...")

COLORS = {
    'uni':   '#2563EB',
    'multi': '#16A34A',
    'enc':   '#DC2626',
    'actual':'#374151',
    'train': '#F59E0B',
    'bg':    '#F9FAFB',
}

# ── Figure 1: Loss Curves (all 3 models) ──
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.patch.set_facecolor(COLORS['bg'])
for ax, losses, color, title in zip(axes,
    [uni_model.losses, multi_model.losses, enc_model.losses],
    [COLORS['uni'], COLORS['multi'], COLORS['enc']],
    ['Univariate LSTM','Multivariate LSTM','Encoder-Decoder LSTM']):
    ax.set_facecolor(COLORS['bg'])
    ax.plot(losses, color=color, lw=2)
    ax.fill_between(range(len(losses)), losses, alpha=0.12, color=color)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel('MSE Loss', fontsize=10)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.annotate(f'Final: {losses[-1]:.5f}', xy=(len(losses)-1, losses[-1]),
                fontsize=9, color=color,
                xytext=(-40, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2))

fig.suptitle('Training Loss Curves — All LSTM Models', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_loss_curves.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

# ── Figure 2: Univariate — Actual vs Predicted ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor(COLORS['bg'])
N_SHOW = min(60, len(y_u_te_inv))
for ax, (actual, pred, split_label) in zip(axes, [
    (y_u_tr_inv[:N_SHOW], u_pred_tr_inv[:N_SHOW], 'Train'),
    (y_u_te_inv[:N_SHOW], u_pred_te_inv[:N_SHOW], 'Test'),
]):
    ax.set_facecolor(COLORS['bg'])
    x = np.arange(N_SHOW)
    ax.plot(x, actual/1e6, color=COLORS['actual'], lw=2, label='Actual', zorder=3)
    ax.plot(x, pred/1e6,   color=COLORS['uni'],    lw=1.8, linestyle='--', label='Predicted', zorder=2)
    ax.fill_between(x, actual/1e6, pred/1e6, alpha=0.15, color=COLORS['uni'])
    ax.set_title(f'Univariate LSTM — {split_label} Set', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('Market Value (€M)', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    rmse_val = rmse(actual, pred)/1e6
    ax.text(0.97, 0.05, f'RMSE: €{rmse_val:.2f}M', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color=COLORS['uni'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

fig.suptitle('Univariate LSTM: Actual vs Predicted Market Values', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_univariate_predictions.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

# ── Figure 3: Multivariate — Actual vs Predicted + Scatter ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor(COLORS['bg'])

ax = axes[0]
ax.set_facecolor(COLORS['bg'])
N_SHOW = min(60, len(y_m_te_inv))
x = np.arange(N_SHOW)
ax.plot(x, y_m_te_inv[:N_SHOW]/1e6, color=COLORS['actual'], lw=2, label='Actual')
ax.plot(x, m_pred_te_inv[:N_SHOW]/1e6, color=COLORS['multi'], lw=1.8, linestyle='--', label='Predicted')
ax.fill_between(x, y_m_te_inv[:N_SHOW]/1e6, m_pred_te_inv[:N_SHOW]/1e6, alpha=0.15, color=COLORS['multi'])
ax.set_title('Multivariate LSTM — Test Set', fontsize=13, fontweight='bold')
ax.set_xlabel('Sample Index'); ax.set_ylabel('Market Value (€M)')
ax.legend(); ax.grid(True, alpha=0.3); ax.spines[['top','right']].set_visible(False)

ax = axes[1]
ax.set_facecolor(COLORS['bg'])
ax.scatter(y_m_te_inv/1e6, m_pred_te_inv/1e6, alpha=0.35, color=COLORS['multi'], s=18, edgecolors='none')
lims = [min(y_m_te_inv.min(), m_pred_te_inv.min())/1e6 - 2,
        max(y_m_te_inv.max(), m_pred_te_inv.max())/1e6 + 2]
ax.plot(lims, lims, 'k--', lw=1.5, alpha=0.6, label='Perfect prediction')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel('Actual (€M)'); ax.set_ylabel('Predicted (€M)')
ax.set_title('Predicted vs Actual Scatter — Multivariate', fontsize=13, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3); ax.spines[['top','right']].set_visible(False)

fig.suptitle('Multivariate LSTM: Market Value Predictions', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_multivariate_predictions.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

# ── Figure 4: Encoder-Decoder — Step 1 vs Step 2 ──
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor(COLORS['bg'])
for ax, step, step_label in zip(axes, [0, 1], ['Step 1 (t+1)', 'Step 2 (t+2)']):
    ax.set_facecolor(COLORS['bg'])
    N_SHOW = min(60, len(y_e_te_inv))
    x = np.arange(N_SHOW)
    ax.plot(x, y_e_te_inv[:N_SHOW, step]/1e6,   color=COLORS['actual'], lw=2, label='Actual')
    ax.plot(x, e_pred_te_inv[:N_SHOW, step]/1e6, color=COLORS['enc'],    lw=1.8, linestyle='--', label='Predicted')
    ax.fill_between(x, y_e_te_inv[:N_SHOW,step]/1e6, e_pred_te_inv[:N_SHOW,step]/1e6,
                    alpha=0.15, color=COLORS['enc'])
    ax.set_title(f'Encoder-Decoder — {step_label}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Sample Index'); ax.set_ylabel('Market Value (€M)')
    ax.legend(); ax.grid(True, alpha=0.3); ax.spines[['top','right']].set_visible(False)
    step_rmse = rmse(y_e_te_inv[:,step], e_pred_te_inv[:,step])/1e6
    ax.text(0.97, 0.05, f'RMSE: €{step_rmse:.2f}M', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=10, color=COLORS['enc'],
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

fig.suptitle('Encoder-Decoder LSTM: Multi-Step Forecasting', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_encoder_decoder_predictions.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

# ── Figure 5: Model Comparison Bar Chart ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(COLORS['bg'])

models = ['Univariate\nLSTM', 'Multivariate\nLSTM', 'Enc-Decoder\nLSTM']
colors = [COLORS['uni'], COLORS['multi'], COLORS['enc']]

for ax, (vals_train, vals_test, title) in zip(axes, [
    ([u_rmse_tr/1e6, m_rmse_tr/1e6, e_rmse_tr/1e6],
     [u_rmse_te/1e6, m_rmse_te/1e6, e_rmse_te/1e6], 'RMSE Comparison (€M)'),
    ([u_mae_te/1e6,  m_mae_te/1e6,  e_mae_te/1e6],
     [u_mae_te/1e6,  m_mae_te/1e6,  e_mae_te/1e6], 'MAE (Test Set, €M)'),
]):
    ax.set_facecolor(COLORS['bg'])
    x = np.arange(len(models))
    if 'RMSE' in title:
        w = 0.35
        bars1 = ax.bar(x - w/2, vals_train, w, color=colors, alpha=0.55, label='Train', edgecolor='white')
        bars2 = ax.bar(x + w/2, vals_test,  w, color=colors, alpha=0.95, label='Test',  edgecolor='white')
        ax.legend(fontsize=10)
        for bar in list(bars1)+list(bars2):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=8.5)
    else:
        bars = ax.bar(x, vals_test, color=colors, alpha=0.9, edgecolor='white')
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_ylabel('€ Millions', fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_model_comparison.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

# ── Figure 6: Feature Importance (proxy via correlation) ──
corr = players_full[MULTI_FEATURES].corrwith(players_full['market_value_eur']).abs().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])
bars = ax.barh(corr.index, corr.values, color=COLORS['multi'], alpha=0.85, edgecolor='white')
ax.set_xlabel('|Correlation with Market Value|', fontsize=11)
ax.set_title('Feature Correlation with Market Value\n(Used in Multivariate & Encoder-Decoder LSTM)', fontsize=13, fontweight='bold')
ax.grid(True, axis='x', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
for bar in bars:
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2,
            f'{bar.get_width():.3f}', va='center', fontsize=8.5)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_feature_importance.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

# ── Figure 7: Sample player multi-step forecast ──
# Pick a top player and show their full timeline + forecast
top_players = players_full.groupby('player_name')['market_value_eur'].mean().nlargest(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.patch.set_facecolor(COLORS['bg'])
axes = axes.flatten()

for ax, pname in zip(axes, top_players):
    ax.set_facecolor(COLORS['bg'])
    grp = players_full[players_full['player_name']==pname].sort_values('season_idx')
    seasons = grp['season'].tolist()
    values  = (grp['market_value_eur'].values / 1e6).tolist()

    multi_sc = multi_scaler.transform(grp[MULTI_FEATURES].values)
    mv_sc    = mv_scaler.transform(grp[['market_value_eur']].values)

    # Encode last 3 seasons → predict next 2
    if len(grp) >= ENC_SEQ + FORECAST:
        seq_x = multi_sc[:ENC_SEQ].reshape(1, ENC_SEQ, len(MULTI_FEATURES))
        fc = enc_model.predict(seq_x)[0]
        fc_inv = mv_scaler.inverse_transform(fc.reshape(-1,1)).flatten() / 1e6
        fc_seasons = ['2024/25 (fc)', '2025/26 (fc)']
    else:
        fc_inv = None

    ax.plot(range(len(seasons)), values, 'o-', color=COLORS['actual'], lw=2.2, ms=7, label='Historical')
    if fc_inv is not None:
        ext_x = list(range(len(seasons), len(seasons)+FORECAST))
        ax.plot([len(seasons)-1] + ext_x,
                [values[-1]] + list(fc_inv),
                's--', color=COLORS['enc'], lw=2, ms=7, label='Forecast')
        ax.fill_between([len(seasons)-1]+ext_x,
                        [values[-1]]+list(fc_inv*0.88),
                        [values[-1]]+list(fc_inv*1.12),
                        alpha=0.15, color=COLORS['enc'])
    all_seasons = seasons + (fc_seasons if fc_inv is not None else [])
    ax.set_xticks(range(len(all_seasons)))
    ax.set_xticklabels(all_seasons, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('€M', fontsize=9)
    ax.set_title(pname.split()[-1], fontsize=11, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

fig.suptitle('Player Market Value: Historical + Encoder-Decoder Forecast (2024/25 & 2025/26)',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_player_forecasts.png", dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
plt.close()

print("\n[5] All outputs saved!")
print(f"  → {OUTPUT_DIR}/metrics_summary.csv")
print(f"  → {OUTPUT_DIR}/predictions_univariate.csv")
print(f"  → {OUTPUT_DIR}/predictions_multivariate.csv")
print(f"  → {OUTPUT_DIR}/predictions_encoder_decoder.csv")
print(f"  → {OUTPUT_DIR}/fig1_loss_curves.png")
print(f"  → {OUTPUT_DIR}/fig2_univariate_predictions.png")
print(f"  → {OUTPUT_DIR}/fig3_multivariate_predictions.png")
print(f"  → {OUTPUT_DIR}/fig4_encoder_decoder_predictions.png")
print(f"  → {OUTPUT_DIR}/fig5_model_comparison.png")
print(f"  → {OUTPUT_DIR}/fig6_feature_importance.png")
print(f"  → {OUTPUT_DIR}/fig7_player_forecasts.png")
print("\n✅ DONE — All 3 LSTM models trained, evaluated, and visualised.\n")
