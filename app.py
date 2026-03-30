from flask import Flask, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
import pandas as pd
import numpy as np
import os, json, pickle
import sys


# ── JSON encoder ──────────────────────────────────────────
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)


# ── LSTM classes (must match main.py for pickle to load) ──
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        scale = 0.1
        n = input_size + hidden_size
        self.Wf = np.random.randn(hidden_size, n) * scale;  self.bf = np.zeros((hidden_size, 1))
        self.Wi = np.random.randn(hidden_size, n) * scale;  self.bi = np.zeros((hidden_size, 1))
        self.Wc = np.random.randn(hidden_size, n) * scale;  self.bc = np.zeros((hidden_size, 1))
        self.Wo = np.random.randn(hidden_size, n) * scale;  self.bo = np.zeros((hidden_size, 1))
        self.hidden_size = hidden_size

    @staticmethod
    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
    @staticmethod
    def tanh(x):    return np.tanh(np.clip(x, -15, 15))

    def forward(self, x, h_prev, c_prev):
        combined = np.vstack([h_prev, x])
        f = self.sigmoid(self.Wf @ combined + self.bf)
        i = self.sigmoid(self.Wi @ combined + self.bi)
        c_tilde = self.tanh(self.Wc @ combined + self.bc)
        c = f * c_prev + i * c_tilde
        o = self.sigmoid(self.Wo @ combined + self.bo)
        h = o * self.tanh(c)
        return h, c

    def run_sequence(self, X_seq):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        for t in range(X_seq.shape[0]):
            h, c = self.forward(X_seq[t].reshape(-1, 1), h, c)
        return h, c


class SimpleLSTM:
    def __init__(self, input_size, hidden_size=32, output_size=1):
        self.cell = LSTMCell(input_size, hidden_size)
        self.Wy = np.random.randn(output_size, hidden_size) * 0.1
        self.by = np.zeros((output_size, 1))
        self.hidden_size = hidden_size
        self.input_size  = input_size
        self.train_losses = []
        self.val_losses   = []

    def predict_one(self, X_seq):
        h, _ = self.cell.run_sequence(X_seq)
        return float((self.Wy @ h + self.by).flatten()[0])

    def predict_batch(self, X):
        return np.array([self.predict_one(X[i]) for i in range(len(X))])

    def _rmse(self, X, y):
        return float(np.sqrt(np.mean((self.predict_batch(X) - y) ** 2)))

    def _get_params(self):
        return [self.cell.Wf, self.cell.bf, self.cell.Wi, self.cell.bi,
                self.cell.Wc, self.cell.bc, self.cell.Wo, self.cell.bo,
                self.Wy, self.by]

    def _set_params(self, params):
        for k, p in zip(["Wf","bf","Wi","bi","Wc","bc","Wo","bo"], params[:8]):
            setattr(self.cell, k, p.copy())
        self.Wy = params[8].copy()
        self.by = params[9].copy()

    def fit(self, X_tr, y_tr, X_vl, y_vl, epochs=80, lr=0.05):
        best = self._rmse(X_tr, y_tr)
        best_p = [p.copy() for p in self._get_params()]
        for e in range(epochs):
            cand = [p + np.random.randn(*p.shape) * lr * (0.995**e) for p in best_p]
            self._set_params(cand)
            loss = self._rmse(X_tr, y_tr)
            if loss < best:
                best   = loss
                best_p = [p.copy() for p in cand]
            else:
                self._set_params(best_p)
            self.train_losses.append(self._rmse(X_tr, y_tr))
            self.val_losses.append(self._rmse(X_vl, y_vl))
        self._set_params(best_p)


class EncoderDecoderLSTM:
    def __init__(self, input_size, hidden_size=32, dec_steps=2):
        self.encoder   = LSTMCell(input_size, hidden_size)
        self.decoder   = LSTMCell(1, hidden_size)
        self.Wy        = np.random.randn(1, hidden_size) * 0.1
        self.by        = np.zeros((1, 1))
        self.dec_steps = dec_steps
        self.hidden_size  = hidden_size
        self.train_losses = []
        self.val_losses   = []

    def predict_one(self, X_seq):
        h, c = self.encoder.run_sequence(X_seq)
        outputs   = []
        dec_input = X_seq[-1, 0:1].reshape(-1, 1)
        for _ in range(self.dec_steps):
            h, c = self.decoder.forward(dec_input, h, c)
            out  = float((self.Wy @ h + self.by).flatten()[0])
            outputs.append(out)
            dec_input = np.array([[out]])
        return np.array(outputs)

    def predict_batch(self, X):
        return np.array([self.predict_one(X[i]) for i in range(len(X))])

    def _rmse(self, X, y):
        return float(np.sqrt(np.mean((self.predict_batch(X) - y) ** 2)))

    def _get_params(self):
        keys = ["Wf","bf","Wi","bi","Wc","bc","Wo","bo"]
        return ([getattr(self.encoder, k) for k in keys] +
                [getattr(self.decoder, k) for k in keys] +
                [self.Wy, self.by])

    def _set_params(self, params):
        keys = ["Wf","bf","Wi","bi","Wc","bc","Wo","bo"]
        for k, p in zip(keys, params[:8]):   setattr(self.encoder, k, p.copy())
        for k, p in zip(keys, params[8:16]): setattr(self.decoder, k, p.copy())
        self.Wy = params[16].copy()
        self.by = params[17].copy()

    def fit(self, X_tr, y_tr, X_vl, y_vl, epochs=80, lr=0.04):
        best   = self._rmse(X_tr, y_tr)
        best_p = [p.copy() for p in self._get_params()]
        for e in range(epochs):
            cand = [p + np.random.randn(*p.shape)*lr*(0.995**e) for p in best_p]
            self._set_params(cand)
            loss = self._rmse(X_tr, y_tr)
            if loss < best:
                best   = loss
                best_p = [p.copy() for p in cand]
            else:
                self._set_params(best_p)
            self.train_losses.append(self._rmse(X_tr, y_tr))
            self.val_losses.append(self._rmse(X_vl, y_vl))
        self._set_params(best_p)


# ── App setup ─────────────────────────────────────────────
app = Flask(__name__, static_folder=".")
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        return super().default(obj)

app.json = NumpyJSONProvider(app)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(BASE_DIR, "player_transfer_value_with_sentiment.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load dataset
df = pd.read_csv(CSV_PATH)
SEASONS = sorted(df["season"].unique())
df["season_idx"] = df["season"].map({s: i for i, s in enumerate(SEASONS)})

PREDICT_FEATURES = [
    "market_value_eur",
    "goals", "assists", "minutes_played",
    "pass_accuracy_pct", "total_injuries",
    "total_days_injured", "availability_rate",
    "vader_compound_score", "social_buzz_score",
    "goal_contributions_per90", "defensive_actions_per90",
]


# ── Load saved models ─────────────────────────────────────
def load_models():
    required = [
        "model_univariate.pkl", "model_multivariate.pkl",
        "model_encoder_decoder.pkl", "scaler_univariate.pkl",
        "scalers_multivariate.pkl", "mv_features.pkl",
        "uni_data.pkl", "uni_data_scaled.pkl", "player_list.pkl",
    ]
    missing = [f for f in required
               if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None, (
            f"Models not found. Run python main.py first to train and save them. "
            f"Missing: {missing}"
        )

    def ld(name):
        return pickle.load(open(os.path.join(MODELS_DIR, name), "rb"))

    models = {
        "model_uni":        ld("model_univariate.pkl"),
        "model_mv":         ld("model_multivariate.pkl"),
        "model_ed":         ld("model_encoder_decoder.pkl"),
        "uni_scaler":       ld("scaler_univariate.pkl"),
        "mv_scalers":       ld("scalers_multivariate.pkl"),
        "mv_features":      ld("mv_features.pkl"),
        "uni_data":         ld("uni_data.pkl"),
        "uni_data_scaled":  ld("uni_data_scaled.pkl"),
        "player_list":      ld("player_list.pkl"),
        # ── Ensemble models (optional — loaded if ensemble.py has been run) ──
        "model_xgb_ensemble":    None,
        "model_lgb_ensemble":    None,
        "ensemble_feature_cols": None,
    }

    # Try loading ensemble models gracefully
    for key, fname in [
        ("model_xgb_ensemble",    "model_xgb_ensemble.pkl"),
        ("model_lgb_ensemble",    "model_lgb_ensemble.pkl"),
        ("ensemble_feature_cols", "ensemble_feature_cols.pkl"),
    ]:
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            try:
                models[key] = pickle.load(open(path, "rb"))
                print(f"✅ Loaded: {fname}")
            except Exception as e:
                print(f"⚠️  Could not load {fname}: {e}")

    return models, None

sys.modules['__main__'].SimpleLSTM = SimpleLSTM
sys.modules['__main__'].LSTMCell = LSTMCell
sys.modules['__main__'].EncoderDecoderLSTM = EncoderDecoderLSTM


MODELS, MODEL_ERROR = load_models()
if MODELS:
    has_ensemble = MODELS["model_xgb_ensemble"] is not None
    print("✅ LSTM models loaded from models/")
    print(f"{'✅' if has_ensemble else '⚠️ '} Ensemble models {'loaded' if has_ensemble else 'not found — run ensemble.py first'}")
else:
    print(f"⚠️  {MODEL_ERROR}")


# ── Build multivariate matrix (needed for XGBoost feature vector) ─────────
MV_MATRIX_SCALED = None

def build_mv_matrix():
    """Pre-compute the full scaled multivariate matrix for XGBoost inference."""
    global MV_MATRIX_SCALED
    if MODELS is None:
        return

    mv_features = MODELS["mv_features"]
    mv_scalers  = MODELS["mv_scalers"]
    player_list = MODELS["player_list"]
    n_players   = len(player_list)
    n_seasons   = 5
    n_feats     = len(mv_features)

    mv_matrix = np.zeros((n_players, n_seasons, n_feats))
    for pi, pname in enumerate(player_list):
        pdata = df[df["player_name"] == pname].sort_values("season_idx")
        for si in range(min(len(pdata), n_seasons)):
            mv_matrix[pi, si, :] = pdata.iloc[si][mv_features].values.astype(float)

    mv_matrix_scaled = mv_matrix.copy()
    for fi in range(n_feats):
        flat = mv_matrix[:, :, fi].reshape(-1, 1)
        mv_matrix_scaled[:, :, fi] = mv_scalers[fi].transform(flat).reshape(n_players, n_seasons)

    MV_MATRIX_SCALED = mv_matrix_scaled

if MODELS:
    build_mv_matrix()


# ── Predict ───────────────────────────────────────────────
def predict_player(player_name: str) -> dict:

    if MODEL_ERROR:
        return {"error": MODEL_ERROR}

    pdata = (df[df["player_name"] == player_name]
             .sort_values("season_idx")
             .reset_index(drop=True))

    if len(pdata) < 2:
        return {"error": f"Not enough seasons of data for '{player_name}'"}

    model_uni       = MODELS["model_uni"]
    model_mv        = MODELS["model_mv"]
    model_ed        = MODELS["model_ed"]
    uni_scaler      = MODELS["uni_scaler"]
    mv_scalers      = MODELS["mv_scalers"]
    mv_features     = MODELS["mv_features"]
    uni_data        = MODELS["uni_data"]
    uni_data_scaled = MODELS["uni_data_scaled"]
    player_list     = MODELS["player_list"]

    # ── Univariate model → market value prediction ──────────────────────────
    pred_mv_eur = None
    loss_curves = {"epochs": [], "train": [], "val": []}
    if player_name in player_list:
        try:
            p_idx       = player_list.index(player_name)
            uni_3d      = uni_data_scaled[:, :, np.newaxis]
            last_seq    = uni_3d[p_idx, -3:, :]
            pred_scaled = model_uni.predict_one(last_seq)
            dummy       = np.zeros((1, uni_data.shape[1]))
            dummy[0, 0] = pred_scaled
            pred_mv_eur = float(uni_scaler.inverse_transform(dummy)[0, 0])
            loss_curves = {
                "epochs": list(range(1, len(model_uni.train_losses) + 1, 5)),
                "train":  [round(float(v), 4) for v in model_uni.train_losses[::5]],
                "val":    [round(float(v), 4) for v in model_uni.val_losses[::5]],
            }
        except Exception as e:
            print(f"Univariate prediction error: {e}")

    # ── Multivariate model → all feature predictions ─────────────────────
    feats  = [f for f in mv_features if f in pdata.columns]
    raw    = pdata[feats].fillna(0).values.astype(float)
    mn, mx = raw.min(axis=0), raw.max(axis=0)
    rng    = np.where(mx - mn == 0, 1.0, mx - mn)
    scaled = (raw - mn) / rng
    seq_len = min(3, len(pdata) - 1)

    last_seq_mv    = scaled[-seq_len:, :]
    pred_scaled    = np.zeros(len(feats))
    pred_scaled[0] = np.clip(model_mv.predict_one(last_seq_mv), 0, 1)
    for fi in range(1, len(feats)):
        vals  = scaled[:, fi]
        trend = (vals[-1] - vals[0]) / max(1, len(vals) - 1)
        pred_scaled[fi] = np.clip(vals[-1] + trend * 0.6, 0, 1)

    pred_raw   = pred_scaled * rng + mn
    prediction = {feat: float(pred_raw[fi]) for fi, feat in enumerate(feats)}

    if pred_mv_eur is not None:
        prediction["market_value_eur"] = pred_mv_eur

    # ── Encoder-Decoder → 2-step forecast ───────────────────────────────
    ed_result = {}
    ed_pred_scaled_s1 = None
    if player_name in player_list:
        try:
            p_idx    = player_list.index(player_name)
            uni_3d   = uni_data_scaled[:, :, np.newaxis]
            last_seq = uni_3d[p_idx, -3:, :]
            preds_2  = model_ed.predict_one(last_seq)
            dummy    = np.zeros((2, uni_data.shape[1]))
            dummy[:, 0] = preds_2
            orig     = uni_scaler.inverse_transform(dummy)[:, 0]
            ed_result = {"step1_eur": float(orig[0]), "step2_eur": float(orig[1])}
            ed_pred_scaled_s1 = float(preds_2[0])
        except Exception as e:
            print(f"Encoder-Decoder error: {e}")

    # ── Ensemble model predictions (XGBoost + LightGBM) ─────────────────
    ensemble_predictions = {}

    ensemble_feature_cols = MODELS.get("ensemble_feature_cols")

    if player_name in player_list and ensemble_feature_cols is not None:
        try:
            # Get the last known season row for this player
            last_row = pdata.iloc[-1]

            # Build the raw feature vector (same mv_features used in training)
            raw_feats = np.array([
                float(last_row.get(f, 0)) for f in mv_features
            ])

            # LSTM prediction for this player (already computed above)
            lstm_pred_val = pred_mv_eur if pred_mv_eur is not None else float(last_row.get("market_value_eur", 0))

            # New dataset row = original features + LSTM prediction (matches mentor's description)
            feature_vector = np.append(raw_feats, lstm_pred_val).reshape(1, -1)

            p_idx_ens = player_list.index(player_name)

            # ── Multivariate LSTM ──
            mv_seq    = MV_MATRIX_SCALED[p_idx_ens, -3:, :]
            mv_scaled = float(model_mv.predict_one(mv_seq))
            mv_eur    = float(mv_scalers[0].inverse_transform([[mv_scaled]])[0][0])
            ensemble_predictions["mv_lstm"] = mv_eur

            # ── Univariate LSTM ──
            uni_3d_local = uni_data_scaled[:, :, np.newaxis]
            uni_seq      = uni_3d_local[p_idx_ens, -3:, :]
            uni_scaled   = float(model_uni.predict_one(uni_seq))
            dummy_u      = np.zeros((1, uni_data.shape[1]))
            dummy_u[0,0] = uni_scaled
            uni_eur      = float(uni_scaler.inverse_transform(dummy_u)[0, 0])

            # ── Encoder-Decoder LSTM ──
            ed_seq       = uni_3d_local[p_idx_ens, -3:, :]
            ed_preds     = model_ed.predict_one(ed_seq)
            dummy_e      = np.zeros((2, uni_data.shape[1]))
            dummy_e[:,0] = ed_preds
            ed_orig      = uni_scaler.inverse_transform(dummy_e)[:, 0]
            ed1_eur      = float(ed_orig[0])
            ed2_eur      = float(ed_orig[1])

            # ── Build feature vector: 14 raw + 4 LSTM predictions ──
            feature_vector = np.append(
                raw_feats,
                [mv_eur, uni_eur, ed1_eur, ed2_eur]
            ).reshape(1, -1)

            # XGBoost Ensemble
            if MODELS["model_xgb_ensemble"] is not None:
                xgb_pred = float(MODELS["model_xgb_ensemble"].predict(feature_vector)[0])
                ensemble_predictions["xgb_ensemble"] = xgb_pred

            # LightGBM Ensemble
            if MODELS["model_lgb_ensemble"] is not None:
                lgb_pred = float(MODELS["model_lgb_ensemble"].predict(feature_vector)[0])
                ensemble_predictions["lgb_ensemble"] = lgb_pred

        except Exception as e:
            print(f"Ensemble prediction error: {e}")

    # ── RMSE against last known season ──────────────────────────────────
    rmse_eur = float(abs(
        prediction.get("market_value_eur", 0) - float(pdata["market_value_eur"].iloc[-1])
    ))

    # ── Historical data ──────────────────────────────────────────────────
    history = {
        "seasons":      [str(s) for s in pdata["season"].values],
        "market_value": [round(float(v) / 1e6, 2) for v in pdata["market_value_eur"].fillna(0)],
        "goals":        [int(v) for v in pdata["goals"].fillna(0)],
        "assists":      [int(v) for v in pdata["assists"].fillna(0)],
    }

    return {
        "player":               player_name,
        "prediction":           prediction,
        "rmse_eur":             round(rmse_eur, 2),
        "loss_curves":          loss_curves,
        "history":              history,
        "predicted_season":     "2024/25",
        "encoder_decoder":      ed_result,
        "ensemble_predictions": ensemble_predictions,
        "model_source":         "main.py (pre-trained)",
    }


# ── Routes ────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

@app.route("/api/predict/<path:player_name>")
def predict(player_name):
    return jsonify(predict_player(player_name))

@app.route("/api/players")
def player_list_route():
    return jsonify(sorted(df["player_name"].unique().tolist()))


@app.route("/api/report")
def report_route():
    report_path = os.path.join(MODELS_DIR, "report_data.json")
    if not os.path.exists(report_path):
        return jsonify({"error": "Report data not found. Run ensemble.py first."})
    with open(report_path, "r") as f:
        return jsonify(json.load(f))


@app.route("/api/plots/<path:filename>")
def serve_plot(filename):
    """Serve matplotlib PNGs saved by ensemble.py from models/plots/."""
    plots_dir = os.path.join(MODELS_DIR, "plots")
    return send_from_directory(plots_dir, filename)

@app.route("/api/status")
def status():
    has_ensemble = MODELS is not None and MODELS.get("model_xgb_ensemble") is not None

    # Check if plots exist in report_data.json
    has_plots = False
    report_path = os.path.join(MODELS_DIR, "report_data.json")
    if os.path.exists(report_path):
        try:
            with open(report_path) as f:
                rdata = json.load(f)
            has_plots = bool(rdata.get("plots"))
        except Exception:
            pass

    if has_ensemble and has_plots:
        msg = "All models ready — ensemble + evaluation plots generated"
    elif has_ensemble:
        msg = "Ensemble models loaded. Re-run ensemble.py to regenerate plots."
    elif MODELS:
        msg = "LSTM models ready. Run ensemble.py for ensemble models + plots."
    else:
        msg = MODEL_ERROR

    return jsonify({
        "models_loaded":   MODELS is not None,
        "ensemble_loaded": has_ensemble,
        "plots_ready":     has_plots,
        "model_source":    "main.py + ensemble.py",
        "message":         msg,
    })

if __name__ == "__main__":
    print("\nTransferIQ running...")

    if MODEL_ERROR:
        print(f"{MODEL_ERROR}\n")
    else:
        has_ens = MODELS.get("model_xgb_ensemble") is not None
        print(f"LSTM models:     ✅ ready")
        print(f"Ensemble models: {'✅ ready' if has_ens else '⚠️  not found'}\n")

    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)