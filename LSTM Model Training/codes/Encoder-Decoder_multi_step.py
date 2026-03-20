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

     # Encoder-Decoder (seq_len=3 → predict t+1, t+2)
    for i in range(len(grp) - ENC_SEQ - FORECAST + 1):
        if i + ENC_SEQ + FORECAST <= len(grp):
            X_enc.append(multi_scaled[i:i+ENC_SEQ])
            y_enc.append(mv_scaled[i+ENC_SEQ:i+ENC_SEQ+FORECAST, 0])

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