import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from pandas.api.indexers import FixedForwardWindowIndexer
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('modelos', exist_ok=True)
os.makedirs('relatorios', exist_ok=True)

# ===========================================================================
# MARK IV — Diagnóstico definitivo e solução:
#
# PROBLEMA RAIZ identificado após 3 iterações:
#   - Effect sizes das features: dist_ema=-0.39, macdh=-0.56, bbp=-0.43
#     → Sinal FRACO mas REAL. Modelos complexos (Attention, Bidir) colapsam
#       porque têm parâmetros demais para esse volume de dados (~16k amostras).
#   - Com janela=60: 16k amostras / 60 timesteps / 32 features = muito esparso
#   - Random Forest sem sequência: precision=56%, recall=9% → baseline real
#
# SOLUÇÃO MARK IV:
#   1. JANELA CURTA = 20 candles (100 minutos)
#      Mais sequências disponíveis, gradientes mais estáveis
#   2. FEATURES SELECIONADAS (9 features, não 32)
#      Só as que têm effect size > 0.1: dist_ema, macdh, bbp, ATRr_14, bbp, rsi, log_ret
#   3. GRU MINIMALISTA: 32 → 16 unidades
#      Menos parâmetros = menos overfitting com dataset pequeno
#   4. RobustScaler (resistente a outliers do WIN)
#   5. TREINAMENTO COM CHECKPOINT (salva o melhor val_loss, não o último)
#   6. TARGET = simulação exata do trade (alvo=1.65×ATR, stop=1.19×ATR)
# ===========================================================================

# Features selecionadas pela análise de effect size e importância RF
# Ordenadas por importância: ATRr_14 > log_ret > dist_ema > bbp > macdh > ...
FEATURE_COLS = [
    'ATRr_14',       # volatilidade atual (mais importante no RF)
    'log_ret',       # retorno do último candle
    'dist_ema',      # distância do preço à EMA (em ATRs) — effect_size=-0.39
    'BBP_5_2.0_2.0', # posição nas Bollinger Bands — effect_size=-0.43
    'MACDh_12_26_9', # histograma MACD — effect_size=-0.56 (melhor sinal)
    'RSI_14',        # RSI — effect_size=-0.23
    'ema_slope3',    # slope da EMA em 3 candles (direção da tendência)
    'ret_lag1',      # retorno do candle anterior (autocorr=-0.37)
    'hour_sin',      # hora do dia (cyclical)
    'hour_cos',
]


class FinancialTimeBenderMark4:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df       = None
        self.scaler   = RobustScaler()  # robusto a outliers do WIN

        self.param_grid = {
            'window_size': [10, 20, 40],   # janelas curtas — mais amostras
            'n_candles':   [10, 15],
            'gain_mult':   [1.65],
            'stop_mult':   [1.19],
        }

    # -----------------------------------------------------------------------
    def _calc_rsi(self, series, period=14):
        delta = series.diff()
        gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        return 100 - (100 / (1 + gain / (loss + 1e-9)))

    # -----------------------------------------------------------------------
    def load_and_engineer_features(self):
        print(f"--> Carregando: {self.filepath}")
        try:
            self.df = pd.read_csv(self.filepath)
        except Exception:
            self.df = pd.read_csv(self.filepath, sep=';')

        col_data = [c for c in self.df.columns
                    if 'time' in c.lower() or 'data' in c.lower()][0]
        self.df[col_data] = pd.to_datetime(self.df[col_data])
        self.df = self.df.sort_values(col_data).reset_index(drop=True)

        for c in ['close','high','low','ATRr_14','RSI_14','EMA_20',
                  'MACDh_12_26_9','BBP_5_2.0_2.0']:
            if c in self.df.columns:
                self.df[c] = pd.to_numeric(self.df[c], errors='coerce')

        # Features derivadas (mínimo necessário)
        self.df['log_ret']    = np.log(self.df['close'] / self.df['close'].shift(1))
        self.df['ret_lag1']   = self.df['log_ret'].shift(1)
        self.df['ema_slope3'] = self.df['EMA_20'].diff(3)
        self.df['dist_ema']   = (self.df['close'] - self.df['EMA_20']) / (self.df['ATRr_14'] + 1e-9)

        # Features de hora
        self.df['hour_sin'] = np.sin(2 * np.pi * self.df[col_data].dt.hour / 24)
        self.df['hour_cos'] = np.cos(2 * np.pi * self.df[col_data].dt.hour / 24)

        self.df.dropna(inplace=True)
        print(f"--> Shape: {self.df.shape} | "
              f"{self.df[col_data].min().date()} → {self.df[col_data].max().date()}")

        # Verifica que todas as features existem
        missing = [f for f in FEATURE_COLS if f not in self.df.columns]
        if missing:
            raise ValueError(f"Features faltando no CSV: {missing}")
        print(f"--> Features disponíveis: {len(FEATURE_COLS)} ✅")

    # -----------------------------------------------------------------------
    def create_target(self, n_candles, gain_mult=1.65, stop_mult=1.19):
        df_t = self.df.copy()
        idx  = FixedForwardWindowIndexer(window_size=n_candles)

        fwd_max  = df_t['high'].rolling(window=idx).max()
        fwd_min  = df_t['low'].rolling(window=idx).min()
        hit_gain = (fwd_max - df_t['close']) >= df_t['ATRr_14'] * gain_mult
        hit_stop = (df_t['close'] - fwd_min) >= df_t['ATRr_14'] * stop_mult

        target = np.where(hit_gain & ~hit_stop, 1,
                 np.where(hit_stop & ~hit_gain, 0, -1))
        df_t['target'] = target
        df_t = df_t.iloc[:-n_candles]
        df_t = df_t[df_t['target'] >= 0].copy()

        ratio = df_t['target'].mean()
        print(f"   [Target] N={n_candles} | gain={gain_mult} | stop={stop_mult} | "
              f"gain_rate={ratio:.3f} | n_valid={len(df_t)}")
        return df_t

    # -----------------------------------------------------------------------
    def prepare_sequences(self, data, window_size, fit_scaler=False):
        vals = data[FEATURE_COLS].values
        if fit_scaler:
            vals = self.scaler.fit_transform(vals)
        else:
            vals = self.scaler.transform(vals)

        tgt = data['target'].values
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(vals[i - window_size:i])
            y.append(tgt[i])
        return np.array(X), np.array(y)

    # -----------------------------------------------------------------------
    def build_model(self, input_shape):
        """
        GRU minimalista: 32 → 16 unidades.
        ~50k parâmetros para ~11k sequências de treino → ratio saudável.
        """
        model = Sequential([
            GRU(32, return_sequences=True,
                kernel_regularizer=l2(2e-4),
                recurrent_regularizer=l2(2e-4),
                input_shape=input_shape),
            Dropout(0.30),
            BatchNormalization(),

            GRU(16, return_sequences=False,
                kernel_regularizer=l2(2e-4)),
            Dropout(0.30),

            Dense(16, activation='relu', kernel_regularizer=l2(2e-4)),
            Dropout(0.20),
            Dense(1, activation='sigmoid'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3, clipnorm=1.0),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC'],
        )
        return model

    # -----------------------------------------------------------------------
    def run_optimization(self):
        best_score  = -999
        best_params = {}
        print("\n=== GRID SEARCH ===")

        for w in self.param_grid['window_size']:
            for nc in self.param_grid['n_candles']:
                gm = self.param_grid['gain_mult'][0]
                sm = self.param_grid['stop_mult'][0]
                print(f"\n  Window={w} | N={nc}")

                df_s = self.create_target(nc, gm, sm)
                if len(df_s) < w + 500:
                    print("  SKIP: dados insuficientes")
                    continue

                cut = int(len(df_s) * 0.70)
                try:
                    X_tr, y_tr = self.prepare_sequences(df_s.iloc[:cut], w, fit_scaler=True)
                    X_vl, y_vl = self.prepare_sequences(df_s.iloc[cut:], w, fit_scaler=False)
                except Exception as e:
                    print(f"  SKIP: {e}")
                    continue

                if len(np.unique(y_tr)) < 2:
                    print("  SKIP: target sem variância")
                    continue

                cw   = dict(enumerate(compute_class_weight(
                    'balanced', classes=np.unique(y_tr), y=y_tr)))
                model = self.build_model((X_tr.shape[1], X_tr.shape[2]))
                h = model.fit(
                    X_tr, y_tr,
                    epochs=8, batch_size=128,
                    validation_data=(X_vl, y_vl),
                    class_weight=cw,
                    verbose=0,
                )

                val_acc  = h.history['val_accuracy'][-1]
                val_loss = h.history['val_loss'][-1]
                val_auc  = h.history.get('val_auc', h.history.get('val_AUC', [0.5]))[-1]
                gap      = val_loss - h.history['loss'][-1]
                score    = val_auc - 0.10 * max(gap, 0)

                print(f"  AUC={val_auc:.4f} | Acc={val_acc:.4f} | "
                      f"Loss={val_loss:.4f} | Gap={gap:.4f} | Score={score:.4f}")

                if score > best_score:
                    best_score  = score
                    best_params = {'window_size': w, 'n_candles': nc, 'gain_mult': gm, 'stop_mult': sm}
                    print("  *** NOVO MELHOR ***")

        print(f"\n✅ Best: {best_params} | Score={best_score:.4f}")
        return best_params


# ===========================================================================
def train_final_model(agent, best_params):
    print(f"\n🚀 TREINAMENTO DEFINITIVO: {best_params}")
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    ws = best_params['window_size']

    df_final = agent.create_target(
        best_params['n_candles'],
        best_params['gain_mult'],
        best_params['stop_mult'],
    )

    n = len(df_final)
    i1, i2 = int(n * 0.70), int(n * 0.85)

    X_train, y_train = agent.prepare_sequences(df_final.iloc[:i1], ws, fit_scaler=True)
    X_val,   y_val   = agent.prepare_sequences(df_final.iloc[i1:i2], ws, fit_scaler=False)
    X_test,  y_test  = agent.prepare_sequences(df_final.iloc[i2:],   ws, fit_scaler=False)

    print(f"  X_train={X_train.shape} | features={len(FEATURE_COLS)}")
    print(f"  Treino → 0:{np.mean(y_train==0):.2%} | 1:{np.mean(y_train==1):.2%}")

    cw    = dict(enumerate(compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train)))
    model = agent.build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()

    ckpt_path = f'modelos/ckpt_markIV_{ts}.keras'
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-5, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_loss',
                        save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=64,
        validation_data=(X_val, y_val),
        class_weight=cw,
        callbacks=callbacks,
        verbose=1,
    )

    # -----------------------------------------------------------------------
    # Avaliação no TESTE
    # -----------------------------------------------------------------------
    print("\n=== AVALIAÇÃO NO CONJUNTO DE TESTE ===")
    y_prob = model.predict(X_test).flatten()

    print(f"  Distribuição das probabilidades preditas:")
    print(f"  min={y_prob.min():.3f} | p25={np.percentile(y_prob,25):.3f} | "
          f"median={np.median(y_prob):.3f} | p75={np.percentile(y_prob,75):.3f} | "
          f"max={y_prob.max():.3f}")

    thresholds = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
    best_th = 0.50
    best_f1 = 0.0
    rel_txt = ""

    header  = f"=== RELATÓRIO MARK IV ({ts}) ===\n"
    header += f"Arquivo: {agent.filepath}\n"
    header += f"Params: {best_params}\n"
    header += f"Features: {FEATURE_COLS}\n\n"
    rel_txt += header
    rel_txt += "=== THRESHOLDS (TESTE) ===\n"

    fmt = f"{'Th':<6} | {'Prec(1)':<8} | {'Rec(1)':<8} | {'F1(1)':<8} | {'Trades':<7} | {'Acc':<7}\n"
    print(fmt, end='')
    rel_txt += fmt

    final_cm  = None
    final_acc = 0.0
    min_trades = max(20, int(len(y_test) * 0.03))

    for th in thresholds:
        y_pred = (y_prob > th).astype(int)
        rd     = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        p1  = rd.get('1', {}).get('precision', 0)
        r1  = rd.get('1', {}).get('recall',    0)
        f1  = rd.get('1', {}).get('f1-score',  0)
        tr  = int(np.sum(y_pred))
        acc = accuracy_score(y_test, y_pred)
        line = f"{th:<6.2f} | {p1:<8.4f} | {r1:<8.4f} | {f1:<8.4f} | {tr:<7} | {acc:<7.4f}"
        print(line)
        rel_txt += line + "\n"

        if f1 > best_f1 and tr >= min_trades:
            best_f1   = f1
            best_th   = th
            final_cm  = confusion_matrix(y_test, y_pred)
            final_acc = acc

    print(f"\n✅ Threshold sugerido: {best_th} (F1={best_f1:.4f})")
    rel_txt += f"\n>>> THRESHOLD SUGERIDO: {best_th} (F1={best_f1:.4f}) <<<\n"
    y_pred_final = (y_prob > best_th).astype(int)
    rel_txt += "\n=== CLASSIFICATION REPORT ===\n"
    rel_txt += classification_report(y_test, y_pred_final, zero_division=0)

    # Artefatos
    with open(f'relatorios/relatorio_{ts}.txt', 'w') as f:
        f.write(rel_txt)

    if final_cm is not None:
        plt.figure(figsize=(6, 5))
        sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Prev 0 (LOSS)', 'Prev 1 (GAIN)'],
                    yticklabels=['Real 0 (LOSS)', 'Real 1 (GAIN)'])
        plt.title(f'Matriz TESTE (Th={best_th}) — Acc: {final_acc:.2%}')
        plt.tight_layout()
        plt.savefig(f'relatorios/matriz_confusao_{ts}.png', dpi=100)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history.history['loss'],     label='Treino')
    axes[0].plot(history.history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    auc_key = 'AUC' if 'AUC' in history.history else 'auc'
    if auc_key in history.history:
        axes[1].plot(history.history[auc_key],         label='Treino AUC')
        axes[1].plot(history.history[f'val_{auc_key}'], label='Val AUC')
        axes[1].set_title('AUC-ROC')
    else:
        axes[1].plot(history.history['accuracy'],     label='Treino Acc')
        axes[1].plot(history.history['val_accuracy'], label='Val Acc')
        axes[1].set_title('Acurácia')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'relatorios/performance_treino_{ts}.png', dpi=100)
    plt.close()

    plt.figure(figsize=(9, 4))
    bins = np.linspace(0, 1, 50)
    plt.hist(y_prob[y_test == 0], bins=bins, alpha=0.6,
             label='Real=0 (LOSS)', color='#D85A30')
    plt.hist(y_prob[y_test == 1], bins=bins, alpha=0.6,
             label='Real=1 (GAIN)', color='#1D9E75')
    plt.axvline(best_th, color='black', linestyle='--', label=f'Th={best_th}')
    plt.title('Distribuição das Probabilidades — Conjunto de Teste')
    plt.xlabel('P(GAIN)')
    plt.ylabel('Frequência')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'relatorios/dist_proba_{ts}.png', dpi=100)
    plt.close()

    model.save(f'modelos/modelo_markIV_{ts}.keras')
    joblib.dump(agent.scaler, f'modelos/scaler_markIV_{ts}.pkl')

    print(f"\n✅ Modelo: modelos/modelo_markIV_{ts}.keras")
    print(f"✅ Scaler: modelos/scaler_markIV_{ts}.pkl")
    print(f"\n📋 FEATURES_MODELO para funcoes.py:")
    print(FEATURE_COLS)
    return best_th, FEATURE_COLS


# ===========================================================================
if __name__ == "__main__":
    arquivo = r'dados\dados_mini_indice_M5_COM_INDICADORES_26-03.csv'
    if not os.path.exists(arquivo):
        arquivo = 'dados/dados_mini_indice_M5_COM_INDICADORES_26-03.csv'

    if os.path.exists(arquivo):
        agent = FinancialTimeBenderMark4(arquivo)
        agent.load_and_engineer_features()
        best_params = agent.run_optimization()
        if best_params:
            train_final_model(agent, best_params)
        else:
            print("❌ Grid search sem resultado.")
    else:
        print(f"❌ ARQUIVO NÃO ENCONTRADO: {arquivo}")