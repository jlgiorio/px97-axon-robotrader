from joblib import load
import config as cfg
import MetaTrader5 as mt5

# Carrega modelo auxiliar e threshold dinâmico com fallbacks
try:
    aux_data = load(cfg.CAMINHO_MODELO_AUX)
    MODELO_AUX = aux_data.get('modelo')
    metadata = aux_data.get('metadata', {}) if isinstance(aux_data, dict) else {}

    thr = aux_data.get('threshold') if isinstance(aux_data, dict) else None
    if thr is None:
        thr = metadata.get('best_threshold')

    if thr is None:
        THRESHOLD_AUX = 0.55  # fallback seguro
        print("⚠️ Chave 'threshold' ausente no artefato; usando padrão 0.55.")
    else:
        THRESHOLD_AUX = float(thr)

    # expõe features esperadas (opcional, para validação em runtime)
    FEATURES_AUX = metadata.get('features')

    print(f"✅ Modelo auxiliar carregado! Threshold dinâmico: {THRESHOLD_AUX:.4f}")
except Exception as e:
    MODELO_AUX = None
    THRESHOLD_AUX = 0.55  # valor padrão de segurança
    FEATURES_AUX = None
    print(f"⚠️ Não foi possível carregar o modelo auxiliar: {e}")

