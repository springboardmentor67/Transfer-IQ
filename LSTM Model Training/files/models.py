"""
================================================================
INFOSYS PROJECT — LSTM Player Market Value Prediction
PyTorch GPU-Accelerated Version (CUDA RTX 3050 / i5-12th Gen)
================================================================
Models:
  1. Univariate LSTM        — market value history → next season
  2. Multivariate LSTM      — 14 features → next season value
  3. Encoder-Decoder LSTM   — 14 features → 2-step forecast

Requirements:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  pip install pandas scikit-learn matplotlib numpy

Run:
  python lstm_player_value_pytorch.py

Outputs saved to ./lstm_outputs/
================================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

# ── Output directory ──────────────────────────────────────────
OUT = "./lstm_outputs"
os.makedirs(OUT, exist_ok=True)

# ── Device: use CUDA if available ────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  INFOSYS PROJECT — LSTM Market Value Prediction")
print(f"{'='*60}")
print(f"  Device : {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ================================================================
# CONFIG
# ================================================================
CSV_PATH     = r"C:\Users\bvaib\Desktop\Football player valuation analyzer\player_transfer_value_with_sentiment.csv"
SEQ_LEN      = 3          # look-back window (seasons)
FORECAST     = 2          # enc-decoder steps ahead
BATCH_SIZE   = 64         # larger batch = better GPU utilisation
EPOCHS       = 150
LR           = 1e-3
HIDDEN_UNI   = 64
HIDDEN_MULTI = 128
HIDDEN_ENC   = 128
NUM_LAYERS   = 2          # stacked LSTM layers
DROPOUT      = 0.2

MULTI_FEATURES = [
    'market_value_eur', 'current_age', 'matches', 'goals', 'assists',
    'minutes_played', 'goals_per90', 'assists_per90', 'pass_accuracy_pct',
    'injury_burden_index', 'availability_rate', 'vader_compound_score',
    'attacking_output_index', 'defensive_actions_per90'
]
N_FEAT = len(MULTI_FEATURES)

# ================================================================
# 1. DATA PREPARATION
# ================================================================
print("\n[1] Loading & preparing data ...")
df = pd.read_csv(CSV_PATH)

SEASON_ORDER = {'2019/20': 0, '2020/21': 1, '2021/22': 2,
                '2022/23': 3, '2023/24': 4}
df['season_idx'] = df['season'].map(SEASON_ORDER)
df = df.sort_values(['player_name', 'season_idx'])

# Keep only players with all 5 seasons
players_full = df.groupby('player_name').filter(lambda x: len(x) == 5)
n_players = players_full['player_name'].nunique()
print(f"  Players with full history : {n_players}")

# ── Fit scalers ──
mv_scaler    = MinMaxScaler()
multi_scaler = MinMaxScaler()
mv_scaler.fit(players_full[['market_value_eur']].values)
multi_scaler.fit(players_full[MULTI_FEATURES].values)

# ── Build sequence arrays ──
X_uni, y_uni     = [], []
X_multi, y_multi = [], []
X_enc,  y_enc    = [], []

for _, grp in players_full.groupby('player_name'):
    grp = grp.sort_values('season_idx').reset_index(drop=True)
    mv_sc    = mv_scaler.transform(grp[['market_value_eur']].values)      # (5,1)
    multi_sc = multi_scaler.transform(grp[MULTI_FEATURES].values)         # (5,14)

    # Univariate: windows of SEQ_LEN → next value
    for i in range(len(grp) - SEQ_LEN):
        X_uni.append(mv_sc[i:i+SEQ_LEN])
        y_uni.append(mv_sc[i+SEQ_LEN, 0])

    # Multivariate: windows of SEQ_LEN → next market value
    for i in range(len(grp) - SEQ_LEN):
        X_multi.append(multi_sc[i:i+SEQ_LEN])
        y_multi.append(mv_sc[i+SEQ_LEN, 0])

    # Enc-Decoder: windows of SEQ_LEN → next FORECAST values
    for i in range(len(grp) - SEQ_LEN - FORECAST + 1):
        X_enc.append(multi_sc[i:i+SEQ_LEN])
        y_enc.append(mv_sc[i+SEQ_LEN:i+SEQ_LEN+FORECAST, 0])

X_uni   = np.array(X_uni,   dtype=np.float32)
y_uni   = np.array(y_uni,   dtype=np.float32)
X_multi = np.array(X_multi, dtype=np.float32)
y_multi = np.array(y_multi, dtype=np.float32)
X_enc   = np.array(X_enc,   dtype=np.float32)
y_enc   = np.array(y_enc,   dtype=np.float32)

print(f"  Univariate   : X={X_uni.shape},   y={y_uni.shape}")
print(f"  Multivariate : X={X_multi.shape}, y={y_multi.shape}")
print(f"  Enc-Decoder  : X={X_enc.shape},   y={y_enc.shape}")

def train_test_split_ts(X, y, ratio=0.8):
    n = int(len(X) * ratio)
    return X[:n], X[n:], y[:n], y[n:]

Xu_tr, Xu_te, yu_tr, yu_te = train_test_split_ts(X_uni,   y_uni)
Xm_tr, Xm_te, ym_tr, ym_te = train_test_split_ts(X_multi, y_multi)
Xe_tr, Xe_te, ye_tr, ye_te = train_test_split_ts(X_enc,   y_enc)

def make_loader(X, y, shuffle=True):
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(Xt, yt), batch_size=BATCH_SIZE,
                      shuffle=shuffle, pin_memory=(DEVICE.type == 'cuda'),
                      num_workers=0)

uni_tr_loader   = make_loader(Xu_tr, yu_tr)
uni_te_loader   = make_loader(Xu_te, yu_te, shuffle=False)
multi_tr_loader = make_loader(Xm_tr, ym_tr)
multi_te_loader = make_loader(Xm_te, ym_te, shuffle=False)
enc_tr_loader   = make_loader(Xe_tr, ye_tr)
enc_te_loader   = make_loader(Xe_te, ye_te, shuffle=False)

print(f"  Train/Test split (80/20) done.")

# ================================================================
# 2. MODEL DEFINITIONS
# ================================================================

# ── Model 1: Univariate LSTM ──────────────────────────────────
class UnivariateLSTM(nn.Module):
    def __init__(self, input_size=1, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):                  # x: (B, T, 1)
        out, _ = self.lstm(x)              # (B, T, H)
        return self.fc(out[:, -1, :])      # (B, 1)


# ── Model 2: Multivariate LSTM ────────────────────────────────
class MultivariateLSTM(nn.Module):
    def __init__(self, input_size=N_FEAT, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                            batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.fc   = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ── Model 3: Encoder-Decoder LSTM ────────────────────────────
class EncoderDecoderLSTM(nn.Module):
    """
    Encoder  : reads SEQ_LEN input steps
    Decoder  : autoregressively generates FORECAST steps
    """
    def __init__(self, input_size=N_FEAT, hidden=128, layers=2,
                 dropout=0.2, forecast=FORECAST):
        super().__init__()
        self.hidden   = hidden
        self.layers   = layers
        self.forecast = forecast

        self.encoder = nn.LSTM(input_size, hidden, num_layers=layers,
                               batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.decoder = nn.LSTM(1, hidden, num_layers=layers,
                               batch_first=True,
                               dropout=dropout if layers > 1 else 0.0)
        self.fc = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x, teacher_forcing_ratio=0.0, targets=None):
        B = x.size(0)
        _, (h, c) = self.encoder(x)           # encode full sequence

        # Seed decoder with last known value
        dec_input = x[:, -1:, 0:1]            # (B, 1, 1)
        outputs   = []

        for t in range(self.forecast):
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred        = self.fc(out)         # (B, 1, 1)
            outputs.append(pred.squeeze(-1))   # (B, 1)

            # Teacher forcing during training
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = targets[:, t:t+1].unsqueeze(-1)
            else:
                dec_input = pred

        return torch.cat(outputs, dim=1)       # (B, FORECAST)


# ================================================================
# 3. TRAINING LOOP
# ================================================================
def train_model(model, tr_loader, te_loader, epochs=EPOCHS,
                lr=LR, model_name="Model", teacher_forcing=False):
    model = model.to(DEVICE)

    # Use AMP (Automatic Mixed Precision) for faster GPU training
    use_amp = (DEVICE.type == 'cuda')
    scaler_amp = torch.cuda.amp.GradScaler() if use_amp else None

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    criterion = nn.HuberLoss()               # robust to outliers vs MSE

    train_losses, val_losses = [], []
    best_val = float('inf')
    best_state = None
    t0 = time.time()

    print(f"\n--- {model_name} ---")
    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        epoch_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    if teacher_forcing:
                        pred = model(Xb, teacher_forcing_ratio=0.5, targets=yb)
                    else:
                        pred = model(Xb)
                    loss = criterion(pred.squeeze(-1), yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimiser)
                scaler_amp.update()
            else:
                if teacher_forcing:
                    pred = model(Xb, teacher_forcing_ratio=0.5, targets=yb)
                else:
                    pred = model(Xb)
                loss = criterion(pred.squeeze(-1), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train = epoch_loss / len(tr_loader)
        train_losses.append(avg_train)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in te_loader:
                Xb, yb = Xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
                pred = model(Xb)
                val_loss += criterion(pred.squeeze(-1), yb).item()
        avg_val = val_loss / len(te_loader)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 30 == 0 or epoch == epochs:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:>3}/{epochs}  "
                  f"Train={avg_train:.6f}  Val={avg_val:.6f}  "
                  f"LR={scheduler.get_last_lr()[0]:.2e}  "
                  f"Elapsed={elapsed:.1f}s")

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"  Best Val Loss : {best_val:.6f}")
    return model, train_losses, val_losses


# ── Train all three ───────────────────────────────────────────
uni_model   = UnivariateLSTM(hidden=HIDDEN_UNI,   layers=NUM_LAYERS, dropout=DROPOUT)
multi_model = MultivariateLSTM(hidden=HIDDEN_MULTI, layers=NUM_LAYERS, dropout=DROPOUT)
enc_model   = EncoderDecoderLSTM(hidden=HIDDEN_ENC, layers=NUM_LAYERS, dropout=DROPOUT)

print("\n[2] Training ...")
uni_model,   uni_tr_loss,   uni_val_loss   = train_model(uni_model,   uni_tr_loader,   uni_te_loader,   model_name="Univariate LSTM")
multi_model, multi_tr_loss, multi_val_loss = train_model(multi_model, multi_tr_loader, multi_te_loader, model_name="Multivariate LSTM")
enc_model,   enc_tr_loss,   enc_val_loss   = train_model(enc_model,   enc_tr_loader,   enc_te_loader,   model_name="Encoder-Decoder LSTM", teacher_forcing=True)


# ================================================================
# 4. SAVE TRAINED MODELS
# ================================================================
print("\n[3] Saving trained models ...")

def save_model(model, name, scaler_mv, scaler_multi):
    path = os.path.join(OUT, f"{name}.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class':      model.__class__.__name__,
        'config': {
            'hidden':      model.lstm.hidden_size if hasattr(model, 'lstm') else HIDDEN_ENC,
            'num_layers':  NUM_LAYERS,
            'dropout':     DROPOUT,
            'seq_len':     SEQ_LEN,
            'n_features':  N_FEAT,
            'forecast':    FORECAST,
            'features':    MULTI_FEATURES,
        }
    }, path)
    print(f"  Saved → {path}")

save_model(uni_model,   "univariate_lstm",   mv_scaler, multi_scaler)
save_model(multi_model, "multivariate_lstm", mv_scaler, multi_scaler)
save_model(enc_model,   "encoder_decoder_lstm", mv_scaler, multi_scaler)

# Also save scalers
import pickle
with open(os.path.join(OUT, "scalers.pkl"), "wb") as f:
    pickle.dump({'mv_scaler': mv_scaler, 'multi_scaler': multi_scaler}, f)
print(f"  Saved → {OUT}/scalers.pkl")


# ================================================================
# 5. EVALUATION
# ================================================================
print("\n[4] Evaluating ...")

def predict_all(model, loader):
    """For univariate / multivariate — returns 1D arrays."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            out = model(Xb).squeeze(-1).cpu().numpy()
            preds.extend(out.flatten().tolist())
            actuals.extend(yb.numpy().flatten().tolist())
    return np.array(actuals), np.array(preds)

def predict_all_multistep(model, loader):
    """For encoder-decoder — returns 2D arrays (N, FORECAST)."""
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for Xb, yb in loader:
            Xb = Xb.to(DEVICE, non_blocking=True)
            out = model(Xb).cpu().numpy()          # (B, FORECAST)
            preds.append(out)
            actuals.append(yb.numpy())             # (B, FORECAST)
    return np.vstack(actuals), np.vstack(preds)    # (N, FORECAST)

def inverse(arr):
    return mv_scaler.inverse_transform(arr.reshape(-1, 1)).flatten()

def compute_metrics(actual_sc, pred_sc, label):
    actual = inverse(actual_sc)
    pred   = inverse(pred_sc)
    rmse   = np.sqrt(mean_squared_error(actual, pred))
    mae    = mean_absolute_error(actual, pred)
    r2     = r2_score(actual, pred)
    mape   = np.mean(np.abs((actual - pred) / (np.abs(actual) + 1e-8))) * 100
    mean_a = np.mean(actual)
    dir_acc= np.mean((pred > mean_a) == (actual > mean_a)) * 100
    print(f"\n  {'─'*46}")
    print(f"  {label}")
    print(f"  {'─'*46}")
    print(f"  R²  Score (Accuracy) : {r2*100:>7.2f}%")
    print(f"  MAPE                 : {mape:>7.2f}%")
    print(f"  Direction Accuracy   : {dir_acc:>7.2f}%")
    print(f"  RMSE                 : €{rmse:>12,.0f}")
    print(f"  MAE                  : €{mae:>12,.0f}")
    return dict(Model=label, R2_pct=round(r2*100,2), MAPE_pct=round(mape,2),
                Direction_Acc_pct=round(dir_acc,2), RMSE=round(rmse), MAE=round(mae),
                actual=actual, pred=pred)

yu_a,  yu_p  = predict_all(uni_model,   uni_te_loader)
ym_a,  ym_p  = predict_all(multi_model, multi_te_loader)
ye_a,  ye_p  = predict_all_multistep(enc_model, enc_te_loader)

r_uni   = compute_metrics(yu_a,      yu_p,      "Univariate LSTM")
r_multi = compute_metrics(ym_a,      ym_p,      "Multivariate LSTM")
r_enc1  = compute_metrics(ye_a[:,0], ye_p[:,0], "Encoder-Decoder (Step t+1)")
r_enc2  = compute_metrics(ye_a[:,1], ye_p[:,1], "Encoder-Decoder (Step t+2)")

metrics_df = pd.DataFrame([
    {k: v for k, v in r.items() if k not in ('actual','pred')}
    for r in [r_uni, r_multi, r_enc1, r_enc2]
])
metrics_df.to_csv(os.path.join(OUT, "accuracy_metrics.csv"), index=False)
print(f"\n  Metrics saved → {OUT}/accuracy_metrics.csv")


# ================================================================
# 6. PLOTS
# ================================================================
print("\n[5] Generating plots ...")

BG    = '#0F1117'
GRID  = '#1E2130'
C_UNI   = '#38BDF8'
C_MULTI = '#4ADE80'
C_ENC   = '#F87171'
C_ACT   = '#FBBF24'

plt.rcParams.update({
    'figure.facecolor':  BG,
    'axes.facecolor':    BG,
    'axes.edgecolor':    GRID,
    'axes.labelcolor':   '#CBD5E1',
    'xtick.color':       '#64748B',
    'ytick.color':       '#64748B',
    'text.color':        '#E2E8F0',
    'grid.color':        GRID,
    'legend.facecolor':  '#1E2130',
    'legend.edgecolor':  GRID,
})

# ── Fig 1: Loss curves ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, tl, vl, col, title in zip(axes,
    [uni_tr_loss,   multi_tr_loss,   enc_tr_loss],
    [uni_val_loss,  multi_val_loss,  enc_val_loss],
    [C_UNI, C_MULTI, C_ENC],
    ['Univariate LSTM', 'Multivariate LSTM', 'Encoder-Decoder LSTM']):
    epochs_x = range(1, len(tl)+1)
    ax.plot(epochs_x, tl, color=col,      lw=1.8, label='Train')
    ax.plot(epochs_x, vl, color='white',  lw=1.5, linestyle='--', alpha=0.7, label='Validation')
    ax.fill_between(epochs_x, tl, alpha=0.08, color=col)
    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Huber Loss')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.annotate(f'Best val: {min(vl):.5f}', xy=(np.argmin(vl)+1, min(vl)),
                fontsize=8, color=col,
                xytext=(20, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color=col, lw=1))

fig.suptitle('Training & Validation Loss — All LSTM Models', fontsize=14,
             fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_loss_curves.png"),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 2: Actual vs Predicted (all 3) ──
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, res, col, title in zip(axes,
    [r_uni, r_multi, r_enc1],
    [C_UNI, C_MULTI, C_ENC],
    ['Univariate LSTM', 'Multivariate LSTM', 'Enc-Decoder (t+1)']):
    N  = min(80, len(res['actual']))
    x  = np.arange(N)
    ax.plot(x, res['actual'][:N]/1e6, color=C_ACT, lw=1.8, label='Actual')
    ax.plot(x, res['pred'][:N]/1e6,   color=col,   lw=1.6, linestyle='--', label='Predicted')
    ax.fill_between(x, res['actual'][:N]/1e6, res['pred'][:N]/1e6, alpha=0.1, color=col)
    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Sample'); ax.set_ylabel('Market Value (€M)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.text(0.97, 0.05, f'R²={res["R2_pct"]:.1f}%\nMAPE={res["MAPE_pct"]:.1f}%',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9, color=col,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=GRID, alpha=0.8))

fig.suptitle('Actual vs Predicted Market Values', fontsize=14,
             fontweight='bold', color='white', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_predictions.png"),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 3: Model comparison bar chart ──
fig, ax = plt.subplots(figsize=(12, 5))
models  = ['Univariate\nLSTM', 'Multivariate\nLSTM',
           'Enc-Dec\n(t+1)', 'Enc-Dec\n(t+2)']
r2_vals = [r_uni['R2_pct'], r_multi['R2_pct'], r_enc1['R2_pct'], r_enc2['R2_pct']]
colors  = [C_UNI, C_MULTI, C_ENC, C_ENC]
xp      = np.arange(len(models))
bars    = ax.bar(xp, r2_vals, color=colors, alpha=0.85,
                 width=0.5, edgecolor=BG, linewidth=1.5)
ax.set_ylim(min(r2_vals) - 1, 101)
ax.set_xticks(xp); ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('R² Score (%)', fontsize=11)
ax.set_title('Model Accuracy Comparison (R² Score)', fontsize=14,
             fontweight='bold', color='white')
ax.grid(True, axis='y', alpha=0.3)
ax.spines[['top','right']].set_visible(False)
for bar, val in zip(bars, r2_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_accuracy_comparison.png"),
            dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

print(f"\n{'='*60}")
print(f"  ✅  ALL DONE")
print(f"{'='*60}")
print(f"  Saved models  → {OUT}/univariate_lstm.pt")
print(f"                → {OUT}/multivariate_lstm.pt")
print(f"                → {OUT}/encoder_decoder_lstm.pt")
print(f"                → {OUT}/scalers.pkl")
print(f"  Metrics       → {OUT}/accuracy_metrics.csv")
print(f"  Plots         → {OUT}/fig1_loss_curves.png")
print(f"                → {OUT}/fig2_predictions.png")
print(f"                → {OUT}/fig3_accuracy_comparison.png")
print(f"{'='*60}\n")


# ================================================================
# HOW TO LOAD A SAVED MODEL LATER
# ================================================================
"""
import torch, pickle
from lstm_player_value_pytorch import UnivariateLSTM, MultivariateLSTM, EncoderDecoderLSTM

# Load scalers
with open("lstm_outputs/scalers.pkl", "rb") as f:
    scalers = pickle.load(f)
mv_scaler    = scalers['mv_scaler']
multi_scaler = scalers['multi_scaler']

# Load a model
checkpoint = torch.load("lstm_outputs/multivariate_lstm.pt", map_location='cpu')
model = MultivariateLSTM(hidden=128, layers=2, dropout=0.2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print("Model loaded:", checkpoint['model_class'])
print("Config:",       checkpoint['config'])
"""