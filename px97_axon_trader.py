# ====================================================== PX97-Axon ============================================================ #
# Nome do Modelo: PX97_Axon                                                                                                     #
# Versão: MARK II                                                                                                               #
# Atualizado em: 14/09/2025                                                                                                     #
# Versão funciona com Paper trading                                                                                             #
# Foi implementada a entrada real pelo preço de ASK e verificação contínua por tick via symbol_info_tick()                      #
# Foi implementada a saida do trade real por TIME_DROP                                                                          #
# Foi implementado sistema de penalidade que faz com que o modelo aumente o critério de entrada para não entrar em operações em #
# seguência,  o Modelo aguarda 8 minutos antes de permitir uma nova operação.                                                   #
# Implementada a função de analise de sentimento de notícias do mercado para calibrar os valores dos parâmentros de negociação  #
# Implementada a função de carregar a meta inicial                                                                              #  
# Implementada a função de enviar mensagens para o Telegram                                                                     #
# Implementada a função de verificação de limites diários                                                                       #
# Implementada a função de salvar a meta diária                                                                                 #
# ============================================================================================================================= #

import time
import logging
import warnings
import funcoes as fn
from datetime import datetime
import config as cfg

# import analise_sentimento as ans

# === Configurações iniciais (Warnings & Logs) === #

# Ignorar warnings desnecessários
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuração do logging
logging.basicConfig(
    filename="logs/px97_axon_trader.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# === Mensagem de inicialização do bot === #

mensagem_inicial = f"""
🖥️ [PX-97 SYSTEM BOOT SEQUENCE OMU]
└─ Codename: AXON (MARK V.1 - PAPER TRADING)
└─ Core Link Established
└─ Predictive Engine: Synchronized
└─ Order Management Unit: ONLINE
└─ Timeframe: M5
└─ Target Asset: {cfg.ATIVO}
└─ Execution Protocol: Armed
🤖 Iniciando operação PX-97 – Axon ativado.
⏱️ Modo de escuta em tempo real iniciado.
🚷 Nenhum operador manual detectado...
"""


# ✅ CARREGAR META INICIAL (ADICIONAR AQUI)
# fn.carregar_meta_inicial()

# ✅ Adicionar à mensagem inicial a meta carregada
mensagem_inicial += f"\n🎯 Meta Diária: {cfg.META_TRADE_POINTS} pontos"
mensagem_inicial += f"\n💰 Capital Inicial: R$ {cfg.CAPITAL_INICIAL:.2f}"

# Enviar mensagem inicial para o Telegram
fn.send_telegram(mensagem_inicial)


# === Loop Contínuo de Operação === #
while True:
      
       
    if ((datetime.now().hour >= cfg.STOP_TRADING and not cfg.TRADE_ABERTO) or (cfg.PONTOS_ACUMULADOS >= cfg.META_TRADE_POINTS)):
        print("Horário limite atingido ou meta batida. PX-97 entrando em modo simlação.")
        fn.send_telegram(f"🔻PX-97 entrando em modo simulação às 17:00. Pontos de {cfg.DIA_MES}: {cfg.PONTOS_ACUMULADOS}")
        fn.save_goal(cfg.PONTOS_ACUMULADOS)
        break

    fn.trade_cycle()                                                                 # função principal de trading
    time.sleep(0.15)                                                                  # Espera 0.15 segundos antes da próxima iteração



