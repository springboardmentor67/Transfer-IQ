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

# Multivariate (seq_len=3 → predict t+1 market value)
    for i in range(len(grp) - MULTI_SEQ - 1 + 1):
        if i + MULTI_SEQ < len(grp):
            X_multi.append(multi_scaled[i:i+MULTI_SEQ])
            y_multi.append(mv_scaled[i+MULTI_SEQ, 0:1])

X_multi = np.array(X_multi); y_multi = np.array(y_multi)

print(f"  Multivariate sequences: X={X_multi.shape}, y={y_multi.shape}")


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

print("\n--- Model 2: Multivariate LSTM ---")
multi_model = MultivariateLSTM(input_size=len(MULTI_FEATURES), hidden_size=48, lr=0.003)
multi_model.train(X_m_tr, y_m_tr, epochs=EPOCHS)

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
print("\n[3] Evaluating Models...")

def rmse(true, pred): return np.sqrt(mean_squared_error(true, pred))
def mae(true, pred):  return np.mean(np.abs(true - pred))

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
