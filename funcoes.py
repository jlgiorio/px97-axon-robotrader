import os
import json
import time
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import numpy as np
import pytz
import modelo_auxiliar as ma
from telegram_bot import TelegramBot
import config as cfg

# Variável global para controlar o intervalo de 5 minutos do log
LAST_METRICS_LOG = cfg.LAST_METRICS_LOG

# ==============================================================================
# ⚠️ IMPORTANTE: COPIE A LISTA DO ARQUIVO .TXT GERADO NO TREINO E COLE AQUI
# A ordem deve ser EXATAMENTE a mesma para o Scaler funcionar.
# Exemplo (substitua pela sua):
FEATURES_MODELO = [
    'volume', 'SMA_10', 'EMA_20', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9',
    'MACDs_12_26_9', 'BBL_5_2.0_2.0', 'BBM_5_2.0_2.0', 'BBU_5_2.0_2.0', 'BBB_5_2.0_2.0',
    'BBP_5_2.0_2.0', 'ATRr_14', 'OBV', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'log_ret', 'rsi'
    ]
# Se der erro de "expecting X features", verifique se esta lista tem o tamanho X.
# ==============================================================================

telegram_bot = TelegramBot()

if not mt5.initialize():
    print("❌ Erro ao conectar com MetaTrader 5")
    quit()

telegram_bot.start_listener()

TRIGGER_PROTECAO = getattr(cfg, 'PROTECTION_TRIGGER', 250.0) # Acima de R$ 250, liga a proteção
PCT_DEVOLUCAO_MAX = getattr(cfg, 'PROTECTION_PCT', 0.50)     # Aceita devolver 50% do topo. Mais que isso = Stop.


# === FUNÇÕES AUXILIARES DE CÁLCULO MANUAL (Idênticas ao Treino) ===
def _calc_rsi_manual(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def _calc_macd_manual(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist

# === FUNÇÕES DO SISTEMA ===

def carregar_meta_inicial():
    try:
        with open(cfg.META_FILE, "r") as f:
            dados = json.load(f)
            cfg.META_TRADE_POINTS = dados.get("meta_ajustada", 600)
            print(f"[META] Meta carregada: {cfg.META_TRADE_POINTS} pontos")
    except FileNotFoundError:
        cfg.META_TRADE_POINTS = 600
        print("[META] Usando meta padrão: 600 pontos")

    cfg.PONTOS_ACUMULADOS = 0
    cfg.PERDA_DIARIA_ACUMULADA = 0
    cfg.CAPITAL_ATUAL = cfg.CAPITAL_INICIAL
    print(f"[META] Variáveis diárias reiniciadas")

def send_telegram(mensagem):
    telegram_bot.send_message(cfg.TELEGRAM_CHAT_ID, mensagem)

def calculate_indicators(df):
    # 1. Indicadores Base (Mantidos para compatibilidade se o modelo usar)
    df['SMA_10'] = ta.sma(df['close'], length=10)
    df['EMA_20'] = ta.ema(df['close'], length=20)
    df['RSI_14'] = ta.rsi(df['close'], length=14)
    df['volume'] = df['tick_volume']

    # MACD via Pandas TA
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df = pd.concat([df, macd], axis=1)

    # Bollinger Bands
    bb = ta.bbands(df['close'], length=5, std=2.0)
    if bb is not None:
        df = pd.concat([df, bb], axis=1)

    df['ATRr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    
    # ADX
    adx = ta.adx(df['high'], df['low'], df['close'], length=14)
    
    if adx is not None:
        df = pd.concat([df, adx], axis=1)

    df['Trend_Direction'] = np.where(df['EMA_20'] > df['EMA_20'].shift(1), 1, -1)
    df['OBV'] = ta.obv(df['close'], df['tick_volume'])

    # --- NOVIDADES DO MARK II (Engenharia de Features) ---
    
    # A. Features Temporais (Seno/Cosseno)
    df['time'] = pd.to_datetime(df['time'])
    df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)

    # B. Log Returns
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

    
    # C. Indicadores Manuais (Para garantir nome idêntico ao treino 'rsi' minusculo)
    df['rsi'] = _calc_rsi_manual(df['close'], period=14)
    
    # Recalcula MACD manual para garantir nomes se necessário, ou usa o do TA
    # O treino usou _calc_macd_manual. Vamos garantir que as colunas existam.
    m_line, m_signal, m_hist = _calc_macd_manual(df['close'])
    # Se o modelo pedir nomes especificos manuais, adicione aqui. 
    # Caso contrário, o pandas_ta já gera MACD_12_26_9, etc.
    
    df.dropna(inplace=True)
    return df

def get_mt5_data(ativo, timeframe, n=None):
    if n is None:
        n = cfg.LOOKBACK + 150 # Garante dados suficientes para janelas
    rates = mt5.copy_rates_from_pos(ativo, timeframe, 0, n)
    if rates is None or len(rates) < cfg.LOOKBACK:
        print("[ERRO] Falha ao coletar dados do MT5")
        exit()

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = calculate_indicators(df)
    return df

def prepare_input(df):
    """
    Prepara os dados EXATAMENTE como o modelo foi treinado.
    Usa a lista FEATURES_MODELO para filtrar e ordenar as colunas.
    """
    df = df.copy()
    
    # Verifica se todas as features necessárias existem no DataFrame
    missing = [col for col in FEATURES_MODELO if col not in df.columns]
    if missing:
        print(f"⚠️ ATENÇÃO: Colunas faltando no DataFrame: {missing}")
        # Tenta preencher com 0 ou ffill para não quebrar, mas é ideal corrigir o calculo
        for m in missing:
            df[m] = 0

    # Seleciona APENAS as features do modelo, na ORDEM CORRETA
    df_model = df[FEATURES_MODELO].dropna()
    
    # Pega apenas os últimos N candles necessários (Lookback)
    # Se o lookback for 60, pegamos os ultimos 60
    if len(df_model) < cfg.LOOKBACK:
        print("⚠️ Dados insuficientes para o Lookback!")
        return None, None
        
    X_raw = df_model.values
    
    # Normaliza usando o Scaler carregado (que espera 20 features)
    try:
        X_scaled = cfg.SCALER.transform(X_raw)
    except ValueError as e:
        print(f"❌ ERRO DE DIMENSÃO NO SCALER: {e}")
        print(f"O Scaler espera {cfg.SCALER.n_features_in_} features.")
        print(f"Você está enviando {X_raw.shape[1]} features.")
        print("Verifique a lista FEATURES_MODELO no início do arquivo funcoes.py")
        quit()

    # Pega a sequência final para o LSTM (ex: shape 1, 60, 20)
    X_seq = X_scaled[-cfg.LOOKBACK:]
    
    return np.expand_dims(X_seq, axis=0), df.iloc[-1]

def close_position(ativo, volume):
    """
    Fecha posição existente usando a contra-ordem com configuração EXATA (IOC + GTC).
    """
    if not mt5.initialize(): return False

    # Verifica se existe posição aberta
    posicoes = mt5.positions_get(symbol=ativo)
    if not posicoes:
        # Se não achou posição, retorna False mas sem crashar
        return False

    pos = posicoes[0] # Assume a primeira posição do ativo (hedge mode ou netting)
    
    # Define o preço de fechamento (Se é COMPRA, fecha vendendo no BID. Se é VENDA, fecha comprando no ASK)
    tick = mt5.symbol_info_tick(ativo)
    if not tick: return False
    
    if pos.type == mt5.ORDER_TYPE_BUY:
        type_close = mt5.ORDER_TYPE_SELL
        price_close = tick.bid
    else:
        type_close = mt5.ORDER_TYPE_BUY
        price_close = tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": ativo,
        "volume": float(volume), # Garante float
        "type": type_close,
        "position": pos.ticket,
        "price": price_close,
        "deviation": 20,
        "magic": pos.magic,
        "comment": "MARK II CLOSE",
        "type_time": mt5.ORDER_TIME_GTC,      # Configuração validada
        "type_filling": mt5.ORDER_FILLING_IOC # Configuração validada
    }

    resultado = mt5.order_send(request)
    
    if resultado.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Erro ao fechar: {resultado.retcode}")
        return False
        
    return True

def get_last_metrics():
    try:
        df = pd.read_csv(cfg.CAMINHO_METRICAS).tail(1)
        metricas = df.to_dict('records')[0]
        return metricas
    except:
        return None

def register_trade(timestamp, preco_entrada, preco_saida, volume, resultado, lucro, alvo, stop, pontos, proba, rsi, atr, adx, dip, dim, didif, tendencia):
    # ADICIONADO: adx nos argumentos e no dicionário
    trade = pd.DataFrame([{
        "timestamp": timestamp,
        "ativo": cfg.ATIVO,
        "contratos": cfg.CONTRATOS,
        "preco_entrada": preco_entrada,
        "preco_saida": preco_saida,
        "tendencia": tendencia,
        "proba": round(proba, 4),
        "rsi": round(rsi, 2),
        "atr": round(atr, 2),
        "adx": round(adx, 2), # NOVO: Registra a força da tendência
        "di_plus": round(dip, 2),
        "di_minus": round(dim, 2),
        "di_dif": round(didif, 2),
        "resultado": resultado,
        "lucro": lucro,
        "pontos": pontos
    }])

    try:
        historico = pd.read_csv(cfg.CAMINHO_LOG)
        historico = pd.concat([historico, trade], ignore_index=True)
        historico.to_csv(cfg.CAMINHO_LOG, index=False)
    except:
        trade.to_csv(cfg.CAMINHO_LOG, index=False)

def check_daily_limits():
    if cfg.PONTOS_ACUMULADOS >= cfg.META_TRADE_POINTS:
        return True
    perda_percentual = (cfg.PERDA_DIARIA_ACUMULADA / cfg.CAPITAL_INICIAL) * 100
    if perda_percentual >= cfg.LOSS_PERCENTUAL_DIARIO * 100:
        return True
    return False

def update_totals(resultado, pontos, lucro):
    cfg.PONTOS_ACUMULADOS += pontos
    cfg.CAPITAL_ATUAL += lucro
    if resultado == "LOSS":
        cfg.PERDA_DIARIA_ACUMULADA += abs(lucro)
    print(f"[STATUS] Pontos: {cfg.PONTOS_ACUMULADOS} | Capital: R$ {cfg.CAPITAL_ATUAL:.2f}")

def check_equity_guard():
    """ 
    Verifica se o lucro devolveu mais do que o permitido.
    Retorna True se deve PARAR de operar.
    """
    if getattr(cfg, 'EQUITY_GUARD_TRIGGERED', False):
        return True
        
    lucro_atual = cfg.CAPITAL_ATUAL - cfg.CAPITAL_INICIAL
    pico = getattr(cfg, 'DAILY_PEAK_PROFIT', 0.0)
    
    # Só ativa a proteção se já tivermos garantido um lucro mínimo (Trigger)
    if pico >= TRIGGER_PROTECAO:
        limite_devolucao = pico * (1 - PCT_DEVOLUCAO_MAX) # Ex: 300 * 0.5 = 150
        
        if lucro_atual < limite_devolucao:
            print(f"\n🛡️ EQUITY GUARD ATIVADO!")
            print(f"   Topo do Dia: R$ {pico:.2f}")
            print(f"   Saldo Atual: R$ {lucro_atual:.2f}")
            print(f"   Motivo: Devolução > {PCT_DEVOLUCAO_MAX*100}% do lucro máximo.")
            print("   ⛔ TRADING ENCERRADO POR HOJE.")
            
            cfg.EQUITY_GUARD_TRIGGERED = True
            send_telegram(f"🛡️ EQUITY GUARD: Trading encerrado.\nTopo: R$ {pico:.2f}\nAtual: R$ {lucro_atual:.2f}")
            return True
            
    return False


def check_active_tick(rsi, atr, adx, DIP, DIM, volume, tendencia):
    TZ_LOCAL = pytz.timezone("America/Sao_Paulo")
    if 'timestamp_entrada' not in cfg.TRADE_INFO:
        return

    ENTRADA_LOCAL = cfg.TRADE_INFO['timestamp_entrada'].astimezone(TZ_LOCAL)
    TEMPO_DECORRIDO = datetime.now(TZ_LOCAL) - ENTRADA_LOCAL

    TICK = mt5.symbol_info_tick(cfg.ATIVO)
    if not TICK: return
    
    PRECO_TICK = TICK.last
    PRECO_ATUAL = TICK.ask if cfg.TRADE_INFO['tipo'] == 'COMPRA' else TICK.bid

    if PRECO_TICK >= cfg.TRADE_INFO['alvo']:
        resultado, preco_saida = "GAIN", cfg.TRADE_INFO['alvo']
    elif PRECO_TICK <= cfg.TRADE_INFO['stop']:
        resultado, preco_saida = "LOSS", cfg.TRADE_INFO['stop']
    # elif TEMPO_DECORRIDO >= timedelta(minutes=cfg.STOP_TRADING) and PRECO_ATUAL > cfg.TRADE_INFO['preco']:
    elif TEMPO_DECORRIDO >= timedelta(minutes=cfg.TIME_DROP_MINUTES) and (PRECO_ATUAL - cfg.TRADE_INFO['preco']) > 50:
        close_position(cfg.ATIVO, cfg.CONTRATOS)
        resultado, preco_saida = "TIME_DROP_GAIN", PRECO_TICK
    elif TEMPO_DECORRIDO >= timedelta(minutes=cfg.TIME_DROP_MINUTES) and (PRECO_ATUAL - cfg.TRADE_INFO['preco']) < -100:
        close_position(cfg.ATIVO, cfg.CONTRATOS)
        resultado, preco_saida = "TIME_DROP_LOSS", PRECO_TICK
    else:
        return

    pontos = preco_saida - cfg.TRADE_INFO['preco']
    lucro = pontos * cfg.VALOR_PONTO * cfg.CONTRATOS
    lucro -= (cfg.TAXA_POR_CONTRATO * cfg.CONTRATOS + cfg.TAXA_FIXA_POR_TRADE)

    update_totals(resultado, pontos, lucro)
    
    # --- RECUPERA DADOS DO SNAPSHOT ---
    proba_in = cfg.TRADE_INFO.get('proba', 0.0)
    rsi_in = cfg.TRADE_INFO.get('rsi', 0.0)
    atr_in = cfg.TRADE_INFO.get('atr', 0.0)
    adx_in = cfg.TRADE_INFO.get('adx', 0.0) # NOVO: Recupera ADX da entrada
    dip_in = cfg.TRADE_INFO.get('dip', 0.0)
    dim_in = cfg.TRADE_INFO.get('dim', 0.0)
    didif_in = cfg.TRADE_INFO.get('didif', 0.0)
    tendencia_in = cfg.TRADE_INFO.get('tendencia', 'N/A')
    # ----------------------------------

    # Passa adx_in para o registro
    register_trade(datetime.now(), cfg.TRADE_INFO['preco'], preco_saida, volume, resultado, round(lucro, 2), cfg.TRADE_INFO['alvo'], cfg.TRADE_INFO['stop'], round(pontos, 2), proba_in, rsi_in, atr_in, adx_in, dip_in, dim_in, didif_in, tendencia_in)

    print(f"📉 TRADE ENCERRADO ({resultado}): R$ {lucro:.2f}")
    
    cfg.TRADE_ABERTO = False
    cfg.TRADE_INFO.clear()
    
    # Se o Equity Guard disparar, nem precisa de Cooldown, o robô trava.
    if check_equity_guard():
        return


    print(f"⏳ Robo em Cooldown de {cfg.COOLDOWN}s...") 
    cfg.IN_COOLDOWN = True
    cfg.COOLDOWN_FIM = datetime.now() + timedelta(seconds=cfg.COOLDOWN)
    
    print(f"⏳ INICIANDO COOLDOWN DE {cfg.COOLDOWN}s (Modo Monitoramento Ativo)...") 
    print(f"   -> Retorno de operações previsto para: {cfg.COOLDOWN_FIM.strftime('%H:%M:%S')}")
    # -------------------------------------------------------------

def auxiliary_predict(proba, rsi, atr, preco_entrada, alvo, stop):
    """
    BYPASS TEMPORÁRIO: Retorna 1.0 para aprovar tudo.
    Motivo: Estamos coletando dados novos do Mark II para treinar um novo Random Forest.
    """
    return 1.0

    # --- CÓDIGO ANTIGO (COMENTADO PARA FUTURO) ---
    # if ma.MODELO_AUX is None: return 1.0
    # entrada = [[proba, rsi, atr, preco_entrada, alvo, stop]]
    # try:
    #     return float(ma.MODELO_AUX.predict_proba(entrada)[0][1])
    # except:
    #     return 1.0

def decide_entry(df, proba, THRESHOLD, volume, tendencia, rsi, atr, adx, dip, dim):
    
    # Captura o último preço de fechamento e a última EMA 20
    last_close = df['close'].iloc[-1]
    last_ema = df['EMA_20'].iloc[-1]
    atr_series = df['ATRr_14'].values

    # Calcula a distância (Absoluta para pegar tanto alta quanto baixa)
    distancia_media = last_close - last_ema
    
    # Define o limite (3x o ATR atual)
    limite_exaustao = 2.5 * atr
    

    def atr_expanding(atr_series):
        if len(atr_series) < 6:
            return False

        # últimos valores
        recent = atr_series[-6:]

        # 1. slope (tendência)
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]

        # 2. aceleração recente
        delta1 = recent[-1] - recent[-2]
        delta2 = recent[-2] - recent[-3]
        acceleration = delta1 - delta2

        return (slope > 0) and (acceleration > 0)
    
    
    def reversao_precoce(df):
        if len(df) < 3:
            return False

  
        rsi_rev = df['RSI_14'].values
        dim_rev = df['DMN_14'].values

        return (
            rsi_rev[-1] > rsi_rev[-2] > rsi_rev[-3]
            and dim_rev[-1] < dim_rev[-2] < dim_rev[-3]
            and df['ADX_14'].values[-1] < 30
        )
    
    if reversao_precoce(df):
       reversao = "Possível reversão"
    else:
        reversao = "Sem reversão detectada"
    
    if atr_expanding(atr_series):
        expansao = "ATR em expansão"
    else:
        expansao = "ATR sem expansão"
    
    # Regra de Bloqueio
    if distancia_media > limite_exaustao:
        mensagem = "🚫 Entrada bloqueada: Preço distante da EMA20 (> 3x ATR)"
        return False, mensagem

    print(f"{expansao} | {reversao}")

    # if (dip - dim) < 9:
    #     cfg.GAIN = 1.1
    
    rsi_max = 78 if (dip - dim) > 15 else 75

    # Exemplo de regra simples
    # ADX no MT5 com 9 pontos a menos de diferença para o calculado pelo robo
    if THRESHOLD < proba <= cfg.THRESHOLD_LIM and tendencia == "ALTA" and (40 < rsi < rsi_max) and 280 < atr < 540 and 20 <= adx < 45 and (dip - dim) >= 8:
        return True, "TREND_FOLLOWING"


    # if (
    #     THRESHOLD < proba < cfg.THRESHOLD_LIM
    #     and not (0.77 < proba < 0.82)          # zona morta histórica
    #     and (tendencia == "ALTA" 
    #         or (dip - dim) > 8 
    #         or reversao_precoce(df)
    #         )
    #     and (35 < rsi < rsi_max)
    #     and 18 < adx < 45
    #     and atr_expanding(atr_series)
    # ):
            
    #     return True, "TREND_EARLY"
       
    return False, None

    


def calcular_alvo_stop_dinamico(modo, atr, rsi, adx, volume_ratio=1.0):
    """
    Calcula alvos e stops dinâmicos baseados no modo de entrada
    """
    
    if modo == "ALTA_PROBA":
        # Alvo maior para alta confiança
        alvo_mult = 1.8 * volume_ratio
        stop_mult = 1.0
        
    elif modo == "MOMENTUM_DIVERGENCIA":
        # Alvo médio, stop apertado
        alvo_mult = 1.5 * volume_ratio
        stop_mult = 0.8
        
    elif modo == "RETRACAO_OTIMIZADA":
        # Alvo e stop reduzidos (scalp)
        alvo_mult = 0.8
        stop_mult = 0.6
        
    elif modo == "SCALPING_INTRADAY":
        # Scalp rápido
        alvo_mult = 1.2
        stop_mult = 0.7
        
    elif modo == "VOLUME_BREAKOUT":
        # Alvo grande, stop normal
        alvo_mult = 2.0 * volume_ratio
        stop_mult = 1.2
        
    else:
        # Padrão
        alvo_mult = 1.65
        stop_mult = 1.19
    
    # Ajustes baseados em volatilidade
    if atr > 150:  # Alta volatilidade
        stop_mult *= 1.2  # Stop mais largo
    elif atr < 80:  # Baixa volatilidade
        alvo_mult *= 0.8  # Alvo mais conservador
    
    pontos_alvo = round(atr * alvo_mult)
    pontos_stop = round(atr * stop_mult)
    
    return pontos_alvo, pontos_stop


def sistema_penalidade_inteligente(resultado_ultimo_trade, modo_ultimo):
    """
    Sistema de penalidade mais inteligente baseado no resultado
    """
    global PENALIDADE_ATIVA, PENALIDADE_FIM, PENALIDADE_VALOR, THRESHOLD
    
    if resultado_ultimo_trade == "GAIN":
        # Reduz penalidade após gain
        PENALIDADE_VALOR = 0.02
        duracao_minutos = 3
    elif resultado_ultimo_trade == "LOSS":
        # Aumenta penalidade após loss
        PENALIDADE_VALOR = 0.08
        duracao_minutos = 8
    else:  # TIME_DROP
        # Penalidade moderada
        PENALIDADE_VALOR = 0.05
        duracao_minutos = 5
    
    # Ajuste adicional baseado no modo
    if modo_ultimo in ["RETRACAO_OTIMIZADA", "SCALPING_INTRADAY"]:
        duracao_minutos = max(2, duracao_minutos - 2)  # Menos penalidade para scalps
    
    PENALIDADE_ATIVA = True
    PENALIDADE_FIM = datetime.now() + timedelta(minutes=duracao_minutos)
    
    print(f"[PENALIDADE] {PENALIDADE_VALOR:.3f} por {duracao_minutos}min após {resultado_ultimo_trade}")

def send_order_real(ativo, contracts, direction, price, sl, tp, magic, comment):
    """
    Envia ordem REAL com normalização pelo TICK SIZE (Passo do Preço).
    Corrige o erro 10016 (Invalid Stops) no WIN/WDO.
    """
    if not mt5.initialize():
        print("❌ Erro MT5: Falha na inicialização")
        return None
    
    sym = mt5.symbol_info(ativo)
    if not sym:
        print(f"❌ Erro: Ativo {ativo} não encontrado")
        return None

    # --- CORREÇÃO CRÍTICA (TICK SIZE) ---
    tick_size = sym.trade_tick_size # Ex: 5.0 para WIN, 0.5 para WDO
    if tick_size == 0: tick_size = 1.0 # Proteção contra divisão por zero

    # Função lambda para arredondar para o múltiplo mais próximo do tick
    # Ex: 160313.0 virou 160315.0 (Múltiplo de 5)
    normalize = lambda x: round(x / tick_size) * tick_size

    price_norm = normalize(float(price))
    sl_norm = normalize(float(sl))
    tp_norm = normalize(float(tp))
    # ------------------------------------
    
    order_type = mt5.ORDER_TYPE_BUY if direction == 'buy' else mt5.ORDER_TYPE_SELL
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": ativo,
        "volume": float(contracts),
        "type": order_type,
        "price": price_norm,
        "sl": sl_norm,
        "tp": tp_norm,
        "deviation": 20,
        "magic": int(magic),
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,      
        "type_filling": mt5.ORDER_FILLING_IOC 
    }
    
    # Debug para conferir o arredondamento no console
    print(f"📡 Enviando Ordem REAL... {ativo} | Tick: {tick_size} | Px: {price_norm} | SL: {sl_norm} | TP: {tp_norm}")
    
    result = mt5.order_send(request)
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ FALHA ENVIO REAL: Código {result.retcode} | {result.comment}")
    else:
        print(f"✅ ORDEM ENVIADA (PI): {result.order}")
        
    return result

def collect_updated_indicators():
    df = get_mt5_data(cfg.ATIVO, cfg.TIMEFRAME)
    
    last = df.iloc[-1]
    
    # Extração segura dos valores
    dip_val = last.get('DMP_14', 0)
    dim_val = last.get('DMN_14', 0)
    
    indicadores = {
        'rsi': last.get('RSI_14', 50),
        'atr': last.get('ATRr_14', 0),
        'adx': last.get('ADX_14', 0),
        'DIP': dip_val,
        'DIM': dim_val,
        'DIdif': dip_val - dim_val,  # NOVO: Diferença entre DI+ e DI-
        'volume': last.get('tick_volume', 0),
        'volume_medio_atual': df['tick_volume'].rolling(20).mean().iloc[-1],
        'tendencia': "ALTA" if last.get('Trend_Direction', 0) == 1 else "BAIXA",
        'trend_dir': last.get('Trend_Direction', 0),
        'limite_percentil': 0 
    }
    return df, indicadores

def get_model_predict(df):
    entrada, _ = prepare_input(df)
    if entrada is None: return None, 0.0
    proba = float(cfg.MODELO.predict(entrada, verbose=0).flatten()[0])
    return entrada, proba

def log_market_metrics(timestamp, ativo, vol_medio, proba, price, rsi, atr, adx, dip, dim, trend, alvo_pts, stop_pts):
    """
    Salva métricas de mercado a cada 5 min.
    """
    arquivo_log = cfg.ARQUIVO_LOG
    
    # Monta o DataFrame com os dados atuais
    dados = pd.DataFrame([{
        "timestamp": timestamp,
        "ativo": ativo,
        "volume_medio": round(vol_medio, 2),
        "proba": round(proba, 4),
        "tick": price,
        "RSI": round(rsi, 2),
        "ATR": round(atr, 2),
        "ADX": round(adx, 2),
        "DIP": round(dip, 2),
        "DIM": round(dim, 2),
        "tendencia": trend,
        "pontos_alvo": int(alvo_pts),
        "pontos_stop": int(stop_pts)
    }])

    try:
        # Se o arquivo não existe, cria com cabeçalho.
        # Se existe, adiciona (append) sem cabeçalho.
        if not os.path.isfile(arquivo_log):
            dados.to_csv(arquivo_log, index=False, sep=',', mode='w')
        else:
            dados.to_csv(arquivo_log, index=False, sep=',', mode='a', header=False)
            
    except Exception as e:
        # Se der erro (ex: arquivo aberto no Excel), avisa no console
        print(f"⚠️ ERRO AO SALVAR LOG DE MERCADO: {e}")

    try:
        # Tenta salvar incrementando (append)
        if not os.path.isfile(arquivo_log):
            dados.to_csv(arquivo_log, index=False, sep=',')
        else:
            dados.to_csv(arquivo_log, mode='a', header=False, index=False, sep=',')
    except Exception as e:
        print(f"⚠️ Erro ao salvar métricas de mercado: {e}")
        

def trade_cycle():
    global LAST_METRICS_LOG 

    # 2. EQUITY GUARD (Proteção de Lucro)
    if check_equity_guard():
        return # Encerra o ciclo se o Guard estiver ativo
    
    df, ind = collect_updated_indicators()
    if df.empty: return
    
    if check_equity_guard():
        return

    if cfg.TRADE_ABERTO:
        check_active_tick(ind['rsi'], ind['atr'], ind['adx'], ind['DIP'], ind['DIM'], ind['volume'], ind['tendencia'])
        return

    entrada, proba = get_model_predict(df)
    
    tick = mt5.symbol_info_tick(cfg.ATIVO)
    preco_atual = tick.last if tick else 0.0
    agora_dt = datetime.now()
    
    # LOG 5 MIN
    if (agora_dt - LAST_METRICS_LOG).total_seconds() >= 300: 
        pts_alvo = ind['atr'] * cfg.GAIN
        pts_stop = ind['atr'] * cfg.STOP
        log_market_metrics(agora_dt, cfg.ATIVO, ind['volume_medio_atual'], proba, preco_atual,
            ind['rsi'], ind['atr'], ind['adx'], ind['DIP'], ind['DIM'], ind['tendencia'], pts_alvo, pts_stop)
        LAST_METRICS_LOG = agora_dt
        print(f"📝 [LOG 5min] Dados de Mercado Coletados.")

    if entrada is None: return 

    os.system('cls' if os.name == 'nt' else 'clear')
    
    if cfg.REAL_TRADE:
        modo_operacao = "REAL"
    else:
        modo_operacao = "SIMULAÇÃO"
    
    if cfg.IN_COOLDOWN:
        agora = datetime.now()
        if agora >= cfg.COOLDOWN_FIM:
            cfg.IN_COOLDOWN = False
            print("✅ COOLDOWN FINALIZADO. Voltando a buscar entradas.")
        else:
            # Mostra status e SAI da função (impede entrada), mas o robô continua rodando
            tempo_restante = (cfg.COOLDOWN_FIM - agora).seconds
            os.system('cls' if os.name == 'nt' else 'clear')
            print(f"⏳ EM COOLDOWN (Restam {tempo_restante}s) | Monitorando mercado...")
            return
        
    entrada_ok, modo = decide_entry(df, proba, cfg.THRESHOLD, ind['volume'], ind['tendencia'], ind['rsi'], ind['atr'], ind['adx'], ind['DIP'], ind['DIM'])

    # Display Limpo
    print(f"🔎 MARK II ({modo_operacao})\n {modo} \n Preço: {preco_atual} | Proba: {proba:.4f} | Threshold: {cfg.THRESHOLD} - {cfg.THRESHOLD_LIM} \n RSI: {ind['rsi']:.1f} | ATR: {ind['atr']:.1f} | ADX: {ind['adx']:.1f} \n DI+: {ind['DIP']:.1f} | DI-: {ind['DIM']:.1f} |Diff: {ind['DIdif']:.1f} \n Trend: {ind['tendencia']}")



    if entrada_ok:
        tick = mt5.symbol_info_tick(cfg.ATIVO)
        if not tick: return
        
        preco_execucao = tick.ask
        if ind['atr'] >= 400:
            cfg.GAIN = 1.15

        stop = preco_execucao - (ind['atr'] * cfg.STOP)
        alvo = preco_execucao + (ind['atr'] * cfg.GAIN)
        
        print(f"🚀 COMPRA! {modo} | Preço: {preco_execucao} | Alvo: {alvo} | Stop: {stop}")
        
        telegram_msg = f"🚀 ENTRADA MARK II ({modo})\nPreço: {preco_execucao}\nProba: {proba:.2f}\nADX: {ind['adx']:.1f}\nDiff DI: {ind['DIdif']:.2f}"
        send_telegram(telegram_msg)

        cfg.TRADE_INFO = {
            'timestamp_entrada': datetime.now(),
            'preco': preco_execucao,
            'alvo': alvo,
            'stop': stop,
            'tipo': 'COMPRA',
            'modo': modo,
            'proba': proba,
            'rsi': ind['rsi'],
            'atr': ind['atr'],
            'adx': ind['adx'], # SALVANDO ADX AQUI
            'dip': ind['DIP'],
            'dim': ind['DIM'],
            'didif': ind['DIdif'],
            'tendencia': ind['tendencia']
        }
        
        cfg.TRADE_ABERTO = True
        
        if cfg.REAL_TRADE:
            send_order_real(cfg.ATIVO, cfg.CONTRATOS, 'buy', preco_execucao, stop, alvo, 31415926535897, "MARK II")

    # return modo_operacao, proba,  ind['tendencia'], ind['rsi'], ind['atr'], ind['adx'], ind['DIP'], ind['DIM']   
            
def save_goal(pts):
    pass # Simplificado