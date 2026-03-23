"""
Hooke's Law TensorFlow Model with Min-Max Normalization
-------------------------------------------------------
Applies the same Min-Max scaling concept from 03_data_preprocessing.py:
  (x - x_min) / (x_max - x_min)  →  0~1 range

Hooke's Law:  F = k * x   →   x = (m * g) / k
  m  = mass (kg)        [0.1 ~ 10.0 kg]
  g  = 9.8 m/s²
  k  = 10 N/m  (spring constant)
  x  = displacement (m)  [0.0098 ~ 0.98 m]

The two features have very different scales → Min-Max normalization
is essential for stable gradient descent (same lesson as salary vs age).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Physical constants ──────────────────────────────────────────────────────
G = 9.8        # m/s²
K = 10.0       # N/m  (spring constant)
N_SAMPLES = 600
NOISE_STD = 0.008

OUTPUT_DIR = Path(__file__).parent.parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Matplotlib dark style
DARK_BG    = "#0f172a"
CARD_BG    = "#1e293b"
BORDER     = "#334155"
TEXT_LIGHT = "#f1f5f9"
TEXT_MUTED = "#94a3b8"
ORANGE     = "#f97316"
BLUE       = "#3b82f6"
INDIGO     = "#6366f1"
GREEN      = "#34d399"
VIOLET     = "#a78bfa"


# ── Min-Max Scaler ───────────────────────────────────────────────────────────
class MinMaxScaler:
    """
    Identical formula to 03_data_preprocessing.py:
        normalized = (x - x_min) / (x_max - x_min)
    Stores min/max so we can inverse-transform predictions back to real units.
    """
    def __init__(self):
        self.min_: float = 0.0
        self.max_: float = 1.0

    def fit(self, x: np.ndarray) -> "MinMaxScaler":
        self.min_ = float(np.min(x))
        self.max_ = float(np.max(x))
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=np.float32) - self.min_) / (self.max_ - self.min_)

    def inverse_transform(self, x):
        return np.asarray(x, dtype=np.float64) * (self.max_ - self.min_) + self.min_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def to_dict(self) -> dict:
        return {"min": self.min_, "max": self.max_}


# ── Hooke's Law TensorFlow Model ─────────────────────────────────────────────
class HookesLawModel:
    def __init__(self):
        self.model: tf.keras.Model | None = None
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.history = None
        self.is_trained: bool = False
        self.metrics: dict = {}
        self._raw_mass: np.ndarray | None = None
        self._raw_disp: np.ndarray | None = None

    # ── data generation ────────────────────────────────────────────────────
    def generate_data(self, n: int = N_SAMPLES, seed: int = 42) -> tuple:
        np.random.seed(seed)
        mass = np.random.uniform(0.1, 10.0, n).astype(np.float32)
        disp = (mass * G / K + np.random.normal(0, NOISE_STD, n)).astype(np.float32)
        disp = np.clip(disp, 0.001, None)
        return mass, disp

    # ── training ───────────────────────────────────────────────────────────
    def train(self, epochs: int = 1500, learning_rate: float = 0.001) -> dict:
        mass, disp = self.generate_data()
        self._raw_mass = mass
        self._raw_disp = disp

        # ── Min-Max Normalization (핵심: 03_data_preprocessing.py 동일 공식) ──
        mass_n = self.scaler_x.fit_transform(mass)
        disp_n = self.scaler_y.fit_transform(disp)

        # ── Build model ───────────────────────────────────────────────────
        tf.random.set_seed(42)
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(1,)),
                tf.keras.layers.Dense(32, activation="relu",
                                      kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(8,  activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="HookesLaw_MLP",
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss", patience=80, restore_best_weights=True, verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=30, min_lr=1e-6, verbose=0
            ),
        ]

        self.history = self.model.fit(
            mass_n, disp_n,
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=0,
        )

        # ── Metrics ───────────────────────────────────────────────────────
        pred_n  = self.model.predict(mass_n.reshape(-1, 1), verbose=0).flatten()
        pred    = self.scaler_y.inverse_transform(pred_n)
        ss_res  = float(np.sum((disp - pred) ** 2))
        ss_tot  = float(np.sum((disp - np.mean(disp)) ** 2))
        r2      = 1.0 - ss_res / ss_tot
        rmse    = float(np.sqrt(np.mean((disp - pred) ** 2)))

        self.metrics = {
            "r2_score":      round(r2, 6),
            "rmse":          round(rmse, 6),
            "final_loss":    round(float(self.history.history["loss"][-1]), 8),
            "epochs_trained": len(self.history.history["loss"]),
            "accuracy_pct":  round(r2 * 100, 2),
        }
        self.is_trained = True

        # ── Generate all PNG plots ─────────────────────────────────────────
        self._plot_normalization(mass, disp, mass_n, disp_n)
        self._plot_loss_curve()
        self._plot_regression(mass, disp)

        return self.metrics

    # ── prediction ─────────────────────────────────────────────────────────
    def predict(self, mass_kg: float) -> dict:
        if not self.is_trained:
            raise RuntimeError("Model is not trained yet. Call /api/train first.")

        m_n    = self.scaler_x.transform(np.array([mass_kg], dtype=np.float32))
        d_n    = float(self.model.predict(m_n.reshape(-1, 1), verbose=0)[0][0])
        disp   = float(self.scaler_y.inverse_transform(d_n))
        ideal  = float(mass_kg * G / K)
        error  = abs(disp - ideal)
        acc    = max(0.0, 100.0 - (error / ideal * 100)) if ideal > 0 else 0.0

        self._plot_prediction(mass_kg, disp)

        return {
            "mass_kg":        mass_kg,
            "displacement_m":  round(disp, 6),
            "displacement_cm": round(disp * 100, 4),
            "force_N":         round(mass_kg * G, 4),
            "ideal_m":         round(ideal, 6),
            "error_m":         round(error, 6),
            "accuracy_pct":    round(acc, 2),
        }

    # ── Plots ──────────────────────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(True, color=BORDER, alpha=0.6, linewidth=0.8)

    def _plot_normalization(self, mass, disp, mass_n, disp_n):
        """Visualization matching 03_data_preprocessing.py concept."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.patch.set_facecolor(DARK_BG)

        # ── Left: raw scale ───────────────────────────────────────────────
        ax = axes[0]
        self._style_ax(ax)
        sc = ax.scatter(mass, disp, c=disp, cmap="YlOrRd", alpha=0.7, s=35, edgecolors="none")
        ax.set_xlabel("Mass  (kg)", color=TEXT_MUTED, fontsize=11)
        ax.set_ylabel("Displacement  (m)", color=TEXT_MUTED, fontsize=11)
        ax.set_title("Before Normalization\n(Raw Scale)", color=TEXT_LIGHT,
                     fontsize=13, fontweight="bold", pad=12)
        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.ax.tick_params(colors=TEXT_MUTED)
        cb.set_label("Displacement (m)", color=TEXT_MUTED)
        ax.text(0.03, 0.96,
                f"mass  range: {mass.min():.2f} ~ {mass.max():.2f} kg\n"
                f"disp  range: {disp.min():.4f} ~ {disp.max():.4f} m",
                transform=ax.transAxes, color=ORANGE, fontsize=9,
                va="top", family="monospace",
                bbox=dict(facecolor=DARK_BG, edgecolor=BORDER, boxstyle="round,pad=0.4"))

        # ── Right: normalized ─────────────────────────────────────────────
        ax = axes[1]
        self._style_ax(ax)
        sc2 = ax.scatter(mass_n, disp_n, c=disp_n, cmap="cool", alpha=0.7, s=35, edgecolors="none")
        ax.set_xlabel("Mass  (0 ~ 1)", color=TEXT_MUTED, fontsize=11)
        ax.set_ylabel("Displacement  (0 ~ 1)", color=TEXT_MUTED, fontsize=11)
        ax.set_title("After Min-Max Normalization\n(0 ~ 1 Range)", color=TEXT_LIGHT,
                     fontsize=13, fontweight="bold", pad=12)
        cb2 = fig.colorbar(sc2, ax=ax, pad=0.02)
        cb2.ax.tick_params(colors=TEXT_MUTED)
        cb2.set_label("Norm. Displacement", color=TEXT_MUTED)
        ax.set_aspect("equal")
        ax.text(0.03, 0.96,
                f"mass  range: {mass_n.min():.3f} ~ {mass_n.max():.3f}\n"
                f"disp  range: {disp_n.min():.3f} ~ {disp_n.max():.3f}",
                transform=ax.transAxes, color=BLUE, fontsize=9,
                va="top", family="monospace",
                bbox=dict(facecolor=DARK_BG, edgecolor=BORDER, boxstyle="round,pad=0.4"))

        formula = r"$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$"
        fig.text(0.5, -0.02, formula, ha="center", color=GREEN, fontsize=14)
        fig.suptitle("Hooke's Law  —  Min-Max Normalization Effect",
                     color=TEXT_LIGHT, fontsize=16, fontweight="bold", y=1.03)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "01_normalization_comparison.png",
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()

    def _plot_loss_curve(self):
        losses  = self.history.history["loss"]
        epochs  = list(range(1, len(losses) + 1))

        fig, ax = plt.subplots(figsize=(11, 6))
        fig.patch.set_facecolor(DARK_BG)
        self._style_ax(ax)

        ax.semilogy(epochs, losses, color=INDIGO, linewidth=2.2, label="Training MSE Loss", zorder=3)
        ax.fill_between(epochs, losses, alpha=0.12, color=INDIGO, zorder=2)

        # Annotate minimum loss
        min_loss  = min(losses)
        min_epoch = losses.index(min_loss) + 1
        ax.scatter([min_epoch], [min_loss], color=GREEN, s=100, zorder=5)
        ax.annotate(
            f" Min Loss: {min_loss:.2e}\n Epoch {min_epoch}",
            xy=(min_epoch, min_loss), xytext=(min_epoch + len(epochs)*0.05, min_loss * 3),
            color=GREEN, fontsize=10, family="monospace",
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
        )

        ax.set_xlabel("Epoch", color=TEXT_MUTED, fontsize=12)
        ax.set_ylabel("Loss  (MSE, log scale)", color=TEXT_MUTED, fontsize=12)
        ax.set_title(
            "Training Loss vs Epoch\n"
            "Min-Max Normalized Hooke's Law  •  TensorFlow",
            color=TEXT_LIGHT, fontsize=14, fontweight="bold",
        )
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_LIGHT, framealpha=0.9)

        stats_txt = (
            f"Final Loss : {losses[-1]:.2e}\n"
            f"R² Score   : {self.metrics.get('r2_score', 0):.4f}\n"
            f"Epochs     : {len(epochs)}"
        )
        ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
                color=TEXT_MUTED, fontsize=9, va="top", ha="right", family="monospace",
                bbox=dict(facecolor=DARK_BG, edgecolor=BORDER, boxstyle="round,pad=0.5"))

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "02_loss_curve.png",
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()

    def _plot_regression(self, mass, disp):
        mass_n  = self.scaler_x.transform(mass)
        pred_n  = self.model.predict(mass_n.reshape(-1, 1), verbose=0).flatten()
        pred    = self.scaler_y.inverse_transform(pred_n)
        idx     = np.argsort(mass)
        m_line  = np.linspace(0.1, 10.0, 300)
        d_ideal = m_line * G / K

        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor(DARK_BG)
        self._style_ax(ax)

        ax.scatter(mass, disp, c=ORANGE, alpha=0.4, s=22, label="Observed Data", zorder=2, edgecolors="none")
        ax.plot(mass[idx], pred[idx], color=BLUE, linewidth=2.5,
                label="TF Model (MLP)", zorder=4)
        ax.plot(m_line, d_ideal, color=GREEN, linewidth=1.8, linestyle="--",
                label="Ideal  F = kx  (k=10)", zorder=3)

        ax.set_xlabel("Mass  (kg)", color=TEXT_MUTED, fontsize=12)
        ax.set_ylabel("Displacement  (m)", color=TEXT_MUTED, fontsize=12)
        ax.set_title(
            f"Regression Fit  —  Hooke's Law\nR² = {self.metrics.get('r2_score', 0):.6f}   "
            f"RMSE = {self.metrics.get('rmse', 0)*100:.4f} cm",
            color=TEXT_LIGHT, fontsize=14, fontweight="bold",
        )
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_LIGHT, framealpha=0.9, fontsize=10)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "03_regression_fit.png",
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()

    def _plot_prediction(self, mass_kg: float, disp_m: float):
        m_range = np.linspace(0.1, 10.0, 300, dtype=np.float32)
        m_n     = self.scaler_x.transform(m_range)
        d_n     = self.model.predict(m_n.reshape(-1, 1), verbose=0).flatten()
        d_range = self.scaler_y.inverse_transform(d_n)
        d_ideal = m_range * G / K

        fig, ax = plt.subplots(figsize=(11, 7))
        fig.patch.set_facecolor(DARK_BG)
        self._style_ax(ax)

        ax.plot(m_range, d_range, color=BLUE, linewidth=2.5, label="TF Model Curve", zorder=3)
        ax.plot(m_range, d_ideal, color=GREEN, linewidth=1.5, linestyle="--",
                label="Ideal F=kx", zorder=2)

        # Crosshair
        ax.axhline(disp_m, color=ORANGE, linestyle=":", linewidth=1.2, alpha=0.6)
        ax.axvline(mass_kg, color=ORANGE, linestyle=":", linewidth=1.2, alpha=0.6)
        ax.scatter([mass_kg], [disp_m], c=ORANGE, s=250, zorder=6,
                   edgecolors="white", linewidths=1.5,
                   label=f"Prediction: {disp_m:.4f} m")

        ideal = mass_kg * G / K
        ax.scatter([mass_kg], [ideal], c=GREEN, s=100, zorder=5, marker="D",
                   edgecolors="white", linewidths=1, label=f"Ideal: {ideal:.4f} m")

        acc = max(0, 100 - abs(disp_m - ideal) / ideal * 100)
        info = (
            f" Input Mass  : {mass_kg} kg\n"
            f" Prediction  : {disp_m:.4f} m  ({disp_m*100:.2f} cm)\n"
            f" Ideal Value : {ideal:.4f} m\n"
            f" Force       : {mass_kg*G:.2f} N\n"
            f" Accuracy    : {acc:.2f} %"
        )
        ax.text(0.03, 0.97, info, transform=ax.transAxes, color=TEXT_LIGHT,
                fontsize=10, va="top", family="monospace",
                bbox=dict(facecolor=DARK_BG, edgecolor=ORANGE, boxstyle="round,pad=0.6"))

        ax.set_xlabel("Mass  (kg)", color=TEXT_MUTED, fontsize=12)
        ax.set_ylabel("Displacement  (m)", color=TEXT_MUTED, fontsize=12)
        ax.set_title(
            f"Prediction Result  —  {mass_kg} kg  →  {disp_m*100:.2f} cm",
            color=TEXT_LIGHT, fontsize=14, fontweight="bold",
        )
        ax.legend(facecolor=CARD_BG, labelcolor=TEXT_LIGHT, framealpha=0.9, fontsize=10)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "04_prediction_result.png",
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()


# ── Singleton ────────────────────────────────────────────────────────────────
_model_instance = HookesLawModel()


def get_model() -> HookesLawModel:
    return _model_instance
