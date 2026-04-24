from datetime import datetime
import MetaTrader5 as mt5
from keras.models import load_model
from joblib import load
import time
import sentment_analysis as sa

results = sa.get_resultados()


# # === Análise de sentimento inicial === #
HUMOR_MERCADO = results['previsao_mercado']                                              # Pega a previsão do mercado do módulo de análise de sentimento
COEFICIENTE_SENTIMENTO = results['resultado_ponderado']                                  # Pega o coeficiente ponderado do sentimento

if HUMOR_MERCADO == "positivo":
    COEFICIENTE_SENTIMENTO = COEFICIENTE_SENTIMENTO + 1
elif HUMOR_MERCADO == "neutro":
    COEFICIENTE_SENTIMENTO = 1

# Parâmetros de configuração
DATA_LOG = datetime.now().strftime("%Y-%m-%d")                                           # Data atual para logs
ULTIMA_EXECUCAO = datetime.now()                                                         # Variável para controlar a última execução e definir momento atual
DIA_MES = datetime.now().strftime("%d_%m")                                               # Dia e mês atual para nomear arquivos

ATIVO = "WINM26"                                                                         # Ativo a ser negociado
TIMEFRAME = mt5.TIMEFRAME_M5                                                             # Timeframe de 5 minutos
LOOKBACK = 60                                                                            # Período de lookback para indicadores técnicos
THRESHOLD = 0.6900                                                                       # atenção a valores entre 0.38 e 0.45 - valor encontrado de acordo com o F1 do novo modelo com atenção(DeepSeek).
THRESHOLD_LIM = 0.8250                                                                   # Threshold para o corrigir overconfidence
THRESHOLD_AUX = 0.30                                                                     # Threshold para o modelo auxiliar
GAIN = 1.65 # * COEFICIENTE_SENTIMENTO                                                   # Alvo de 1.65 vezes o ATR e ajuste de sentimento
STOP = 1.20 # * COEFICIENTE_SENTIMENTO                                                   # Stop de 1.17 vezes o ATR e ajuste de sentimento
VOLUME_MINIMO_MEDIA = 30000                                                              # Volume mínimo médio para considerar entrada (ajustável)
REAL_TRADE = False                                                                        # Definir como True para operações reais, False para simulação
CONTRATOS = 1                                                                            # Número de contratos a serem negociados (ajustável)
VALOR_PONTO = 0.20                                                                       # Valor por ponto do ativo (ajustável)
TAXA_POR_CONTRATO = 0.0001                                                               # Taxa por contrato (ajustável)
TAXA_FIXA_POR_TRADE = 0.0001                                                             # Taxa fixa por trade (ajustável)

LAST_METRICS_LOG = datetime.min                                                          # Variável global para controlar o intervalo de 5 minutos do log

# === Variáveis de controle de saída === #
TRADE_ABERTO = False                                                                     # Indica se há um trade aberto
TRADE_INFO = {}                                                                          # Armazena informações do trade atual
CAPITAL_INICIAL = 5                                                                      # Capital inicial para simulação
START_TRADING_HOUR = 9                                                                   # Horario para iniciar operações (9 = 09:00)
START_TRADING_MINUTE = 15                                                                # Minuto para iniciar operações (20 = 20 minutos)
STOP_TRADING = 17                                                                        # Horario limite para operar (18 = 18:00)
TIME_DROP_MINUTES = 106                                                                  # Tempo máximo em minutos para manter a posição aberta
WARNED = False                                                                           # Indica se o aviso de real trading foi enviado
DEEP_DISTANCE = 950                                                                      # Distância em pontos para considerar alerta de distância do fundo diário    
DEEP_ALERT = "✅"                                                                       # Alerta de distância do fundo diário
IN_COOLDOWN = False                                                                      # Indica se está em período de cooldown
COOLDOWN = 500                                                                           # Tempo em segundos para cooldown entre trades
COOLDOWN_FIM = datetime.min                                                              # Momento em que o cooldown termina
# === Variáveis de controle de perda === #
LOSS_PERCENTUAL_DIARIO = 0.30                                                            # Percentual de perda diária para encerrar operações
PERDA_DIARIA = 0                                                                         # Perda diária acumulada

# === Variáveis de controle de penalidade === #
# define penalidade para evitar entradas em sequência
PENALIDADE_ATIVA = False                                                                 # Indica se a penalidade está ativa
PENALIDADE_FIM = datetime.now()                                                          # Momento em que a penalidade termina
PENALIDADE_VALOR = 0.05                                                                  # Valor da penalidade a ser somado ao threshold


# === Dados para conexão do telegram === #
TELEGRAM_TOKEN = [SEU_TOKEN_AQUI]                                                        # Token do bot do Telegram
TELEGRAM_CHAT_ID = [SEU_CHAT_ID_AQUI]                                                    # Chat ID do usuário no Telegram

# === Variáveis de controle de pontos e metas === #
META_FILE = "dados\\meta_trade.json"                                                     # Arquivo para armazenar a meta diária
META_TRADE_POINTS = 800                                                                  # Meta diária inicial de pontos (ajustável)
PONTOS_ACUMULADOS = 0                                                                    # Pontos acumulados no dia
PERDA_DIARIA_ACUMULADA = 0                                                               # ✅ Nova variável para perdas
CAPITAL_ATUAL = CAPITAL_INICIAL                                                          # ✅ Controlar capital em tempo real

# === Caminho para carregamento do modelo LSTM treinado === #
CAMINHO_MODELO   = "[CAMINHO_PARA_MODELO_TREINADO"                                       # Caminho para o modelo LSTM
CAMINHO_SCALER   = "CAMINHO_SCALER_AQUI"                                                 # Caminho para o scaler

CAMINHO_REALTIME = "dados/parametros_realtime.json"

# === Caminho para o modelo auxiliar ===
CAMINHO_MODELO_AUX = "[CAMINHO_MODELO_AUX_AQUI"                                          # carrega modelo auxiliar
MODELO = load_model(CAMINHO_MODELO)                                                      # Carrega o modelo LSTM
SCALER = load(CAMINHO_SCALER)                                                            # Carrega o scaler

# === Arquivos de logs e registros === #
CAMINHO_LOG = f"dados/px97_axon_operacoes_log_{DIA_MES}.csv"                            # Caminho para o arquivo de log de operações
ARQUIVO_LOG = f"dados/monitoramento_mercado_5min_log_{DIA_MES}.csv"                     # Caminho para o arquivo de log de probabilidades
CAMINHO_METRICAS = f"dados/px97_axon_metricas_entrada_{DIA_MES}.csv"
ARQUIVO_OPERACOES = f"dados/operacoes.json"                                             # Caminho para o arquivo de operações

# Configurações de parametros adicionais podem ser adicionadas aqui conforme necessário
proba = 0.0
rsi = 0.0
atr = 0.0
adx = 0.0
dip = 0.0
dim = 0.0
diferenca_DI = 0.0
tendencia = ""
distancia_fundo = 0.0
