# telegram_bot.py
import requests
import threading
import time
from datetime import datetime
import config as cfg
import funcoes as fn
import json


class TelegramBot:
    """
    Classe para gerenciar comunicação bidirecional com Telegram
    Pode RECEBER comandos e ENVIAR mensagens
    """
    
    def __init__(self):
        """
        Método construtor - inicializa o objeto quando criado com 'TelegramBot()'
        É chamado automaticamente quando você faz: bot = TelegramBot()
        """
        # Token do seu bot (você pega com @BotFather no Telegram)
        self.token = cfg.TELEGRAM_TOKEN
        
        # ID do chat onde o bot vai operar (seu ID ou de um grupo)
        self.chat_id = cfg.TELEGRAM_CHAT_ID
        
        # Controla qual foi a última mensagem processada (evita repetições)
        self.last_update_id = 0
        
        # Flag para controlar se o bot está rodando ou não
        self.running = False
        
        print("[BOT] Bot Telegram inicializado - pronto para receber comandos")
    
    def get_updates(self):
        """
        Busca novas mensagens no Telegram
        Retorna uma lista de mensagens não processadas
        """
        # URL da API oficial do Telegram para buscar mensagens
        url = f'https://api.telegram.org/bot{self.token}/getUpdates'
        
        # Parâmetros da requisição:
        # - offset: last_update_id + 1 → busca apenas mensagens NOVAS
        # - timeout: 30 segundos → a requisição fica "esperando" novas mensagens
        params = {'offset': self.last_update_id + 1, 'timeout': 30}
        
        try:
            # Faz a requisição HTTP para a API do Telegram
            response = requests.get(url, params=params, timeout=35)
            
            # Converte resposta JSON e pega a lista de resultados (mensagens)
            # Se não houver mensagens, retorna lista vazia []
            return response.json().get('result', [])
            
        except Exception as e:
            # Se der erro (sem internet, etc), mostra mas não quebra o programa
            print(f"[BOT] Erro ao buscar mensagens: {e}")
            return []  # Retorna lista vazia para continuar funcionando
    
    def process_command(self, message):
        """
        Analisa uma mensagem e executa o comando correspondente
        Retorna True se era um comando válido, False se não era comando
        """
        # Pega o texto da mensagem e remove espaços extras
        text = message.get('text', '').strip().lower()
        
        # Pega o ID de quem enviou a mensagem (para responder)
        chat_id = message['chat']['id']
        
        # Verifica se a mensagem começa com '/' → é um comando?
        if not text.startswith('/'):
            return False  # Não é comando, ignora
            
        # Dicionário que mapeia comandos para funções
        # '/threshold' → chama self.set_threshold
        # '/status' → chama self.get_status
        commands = {
            '/threshold': self.set_threshold,
            '/meta': self.set_meta,
            '/DeepDistance': self.set_deep_distance,
            '/real': self.set_real_trade,
            '/simulate': self.set_simulation_mode,
            '/parametros': self.get_parametros,
            '/status': self.get_status,
            '/help': self.show_help
        }
        
        # Percorre todos os comandos disponíveis
        for cmd, handler in commands.items():
            # Verifica se a mensagem começa com este comando
            if text.startswith(cmd):
                # Se sim, chama a função correspondente
                handler(text, chat_id)
                return True  # Comando processado com sucesso
                
        # Se chegou aqui, é um comando desconhecido
        self.send_message(chat_id, "❌ Comando desconhecido. Use /help para ver opções.")
        return False
    
    def set_threshold(self, text, chat_id):
        """
        Comando: /threshold 0.75
        Altera o limite de probabilidade para entrar em trades
        """
        try:
            # Divide o texto: "/threshold 0.75" → ['/threshold', '0.75']
            # Pega o segundo elemento [1] que é o valor
            value = float(text.split()[1])
            
            # Valida se o valor está entre 0 e 1 (probabilidade)
            if 0 <= value <= 1:
                # ATUALIZA A CONFIGURAÇÃO GLOBAL
                cfg.THRESHOLD = value
                
                # Confirma para o usuário
                self.send_message(chat_id, f"✅ Threshold atualizado para: {value}")
                print(f"[BOT BETA] Threshold alterado para: {value}")
            else:
                self.send_message(chat_id, "❌ Valor deve estar entre 0 e 1")
                
        except (IndexError, ValueError):
            # Se o usuário não passou valor ou passou texto inválido
            self.send_message(chat_id, "❌ Uso correto: /threshold 0.75")
    
    
    def set_meta(self, text, chat_id):
        """
        Comando: /meta 800
        Altera a meta diária de pontos
        """
        try:
            value = int(text.split()[1])
            
            if value > 0:
                cfg.META_TRADE_POINTS = value
                self.send_message(chat_id, f"✅ Meta diária atualizada para: {value} pontos")
                print(f"[BOT BETA] Meta alterada para: {value}")
            else:
                self.send_message(chat_id, "❌ Meta deve ser positiva")
                
        except (IndexError, ValueError):
            self.send_message(chat_id, "❌ Uso correto: /meta 800")
    
    def set_deep_distance(self, text, chat_id):
        """
        Comando: /DeepDistance 700
        Altera a distância do fundo diário para alertas
        """
        try:
            value = int(text.split()[1])
            
            if value > 0:
                cfg.DEEP_DISTANCE = value
                self.send_message(chat_id, f"✅ Distância do fundo diário atualizada para: {value} pontos")
                print(f"[BOT BETA] Distância do fundo alterada para: {value}")
            else:
                self.send_message(chat_id, "❌ Distância deve ser positiva")
                
        except (IndexError, ValueError):
            self.send_message(chat_id, "❌ Uso correto: /DeepDistance 700")     
            
             
    def set_real_trade(self, chat_id):
        """
        Comando: /real
        Ativa o modo Real Trade
        """
        fn.REAL_TRADE = True
        self.send_message(chat_id, "✅ Modo Real Trade ATIVADO")
        print("[BOT BETA] Modo Real Trade ativado pelo usuário")
    
    def set_simulation_mode(self, chat_id):
        """
        Comando: /simulate
        Ativa o modo Simulação
        """
        fn.REAL_TRADE = False
        self.send_message(chat_id, "✅ Modo Simulação ATIVADO")
        print("[BOT BETA] Modo Simulação ativado pelo usuário")
                 
    def get_parametros(self, text, chat_id):
        
        """
        Comando: /parametros
        Mostra os parâmetros atuais do robô
        """

        parametros_msg = f"""
🤖 **PARÂMETROS ATUAIS DO MERCADO (ROBÔ BETA)**
• **Probabilidade do Modelo**: {cfg.proba:.4f}
• **Tendência do Ativo**: {cfg.tendencia}
• **RSI Atual**: {cfg.rsi:.2f}
• **ATR Atual**: {cfg.atr:.2f}
• **ADX Atual**: {cfg.adx:.2f}
• **DI+**: {cfg.dip:.2f}
• **DI-**: {cfg.dim:.2f}
• **Diferença DI**: {cfg.diferenca_DI:.2f}
• **Distância do Fundo (20 candles)**: {cfg.distancia_fundo} points
        """
        self.send_message(chat_id, parametros_msg)
        
    def get_status(self, text, chat_id):
        """
        Comando: /status
        Mostra situação atual do robô
        """

        status_msg = f"""
🤖 **STATUS DO ROBÔ (ROBÔ BETA)**

• **Pontos Acumulados**: {cfg.PONTOS_ACUMULADOS}
• **Meta Diária**: {cfg.META_TRADE_POINTS}
• **Threshold**: {cfg.THRESHOLD}
• **Contratos**: {cfg.CONTRATOS}
• **Trade Aberto**: {'✅ SIM' if cfg.TRADE_ABERTO else '❌ NÃO'}
• **Em Cooldown**: {'✅ SIM' if cfg.IN_COOLDOWN else '❌ NÃO'}
• **Capital**: R$ {cfg.CAPITAL_ATUAL:.2f}
• **Última Operação**: {datetime.now().strftime('%H:%M:%S')}
📊 **Configurações Ativas**:
- Ativo: {cfg.ATIVO}
- Timeframe: M5
- Real Trade: {'✅ LIGADO' if cfg.REAL_TRADE else '❌ SIMULAÇÃO'}
        """
        self.send_message(chat_id, status_msg)
    
    def show_help(self, text, chat_id):
        """
        Comando: /help
        Mostra todos os comandos disponíveis
        """
        help_text = """
🆘 **COMANDOS DISPONÍVEIS**

📊 **Configurações**:
`/threshold 0.75` - Altera probabilidade mínima (0-1)
`/meta 800` - Altera meta diária de pontos
`/contratos N` - Altera quantidade de contratos (desativado)
`/DeepDistance 700` - Altera distância do fundo diário para alertas

⚙️ **Controle**:
`/status` - Mostra status atual
`/parametros` - Mostra parâmetros atuais do mercado
`/real` - Ativa modo Real Trade
`/simulate` - Ativa modo Simulação


❓ **Ajuda**:
`/help` - Mostra esta mensagem

💡 **Exemplos**:
`/threshold 0.68`
`/meta 650`
`/deepdistance 800`
`/real`
`/simulate`
`/status`
        """
        self.send_message(chat_id, help_text)
    
    def send_message(self, chat_id, text):
        """
        Envia uma mensagem para um chat específico
        Usado tanto para respostas de comandos quanto para alertas
        """
        # URL da API para enviar mensagens
        url = f'https://api.telegram.org/bot{self.token}/sendMessage'
        
        # Dados da mensagem:
        # - chat_id: para quem enviar
        # - text: conteúdo da mensagem  
        # - parse_mode: Markdown para formatação (negrito, etc)
        payload = {
            'chat_id': chat_id, 
            'text': text, 
            'parse_mode': 'Markdown'
        }
        
        try:
            # Envia a mensagem via POST
            requests.post(url, data=payload)
        except Exception as e:
            print(f"[BOT BETA] Erro ao enviar mensagem: {e}")
    
    def start_listener(self):
        """
        Inicia o listener em uma thread separada
        O bot começa a "escutar" comandos sem bloquear o programa principal
        """
        self.running = True
        
        # Mensagem de inicialização
        print("[BOT BETA] 🤖 Listener do Telegram INICIADO")
        print("[BOT BETA] 📱 Pronto para receber comandos via Telegram")
        
        def listener_loop():
            """
            Função interna que roda em loop verificando mensagens
            Esta função roda em thread separada
            """
            print("[BOT BETA] 🔄 Loop de mensagens iniciado...")
            
            while self.running:
                
                # Grava JSON com parâmetros
                dados_meta = {
                "proba": round(cfg.proba,2),
                "rsi": round(cfg.rsi,2),
                "atr": round(cfg.atr,2),
                "adx": round(cfg.adx,2),
                "dip": round(cfg.dip,2),
                "dim": round(cfg.dim,2),
                "Diferenca_DI": round(cfg.diferenca_DI,2),
                "tendencia": cfg.tendencia,
                "distancia_fundo": cfg.distancia_fundo,
                "data": datetime.now().strftime("%Y-%m-%d"),
                "Trade_Real": cfg.REAL_TRADE,
                "Trade_aberto": cfg.TRADE_ABERTO,
                "Cooldown": cfg.IN_COOLDOWN,
                "capital_final": cfg.CAPITAL_ATUAL
            }
            
                with open(cfg.CAMINHO_REALTIME, "w") as f:
                    json.dump(dados_meta, f, indent=2)
                    
                    # Busca novas mensagens
                updates = self.get_updates()
                
                # Processa cada mensagem nova
                for update in updates:
                    # Atualiza o ID da última mensagem processada
                    self.last_update_id = update['update_id']
                    
                    # Verifica se a mensagem tem conteúdo
                    if 'message' in update:
                        print(f"[BOT BETA] 📨 Mensagem recebida: {update['message'].get('text', '')}")
                        
                        # Processa o comando da mensagem
                        self.process_command(update['message'])
                
                # Espera 2 segundos antes de verificar novamente
                # Isso evita sobrecarregar a API do Telegram
                time.sleep(2)
            
            print("[BOT] 🛑 Loop de mensagens parado")
        
        # Cria e inicia a thread
        # daemon=True → thread para quando o programa principal parar
        thread = threading.Thread(target=listener_loop, daemon=True)
        thread.start()
        
        # Mensagem de confirmação
        self.send_message(self.chat_id, "🤖 **Bot conectado e ouvindo comandos!**\nUse /help para ver opções.")
    
    def stop_listener(self):
        """
        Para o listener de mensagens
        """
        self.running = False
        print("[BOT] 🛑 Parando listener do Telegram...")
        self.send_message(self.chat_id, "🛑 **Bot desconectado.**")


# # ⚡ EXEMPLO DE USO - TESTE RÁPIDO
# if __name__ == "__main__":
#     """
#     Se executar este arquivo diretamente, faz um teste
#     python telegram_bot.py
#     """
#     print("🧪 Testando Bot Telegram...")
    
#     # Cria o bot
#     bot = TelegramBot()
    
#     # Inicia o listener
#     bot.start_listener()
    
#     # Mantém o programa rodando por 30 segundos para testar
#     print("⏳ Teste rodando por 60 segundos...")
#     time.sleep(60)
    
#     # Para o bot
#     bot.stop_listener()
#     print("✅ Teste concluído!")
    