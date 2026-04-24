# 🤖 PX97-Axon — Modelo Preditivo para Mini-índice Bovespa (WIN)

Sistema automatizado de análise técnica e decisão de entrada para o Mini-índice Bovespa (WIN),
desenvolvido integralmente em Python. O modelo combina uma rede GRU (Gated Recurrent Unit) treinada com indicadores
técnicos selecionados por análise de effect size para gerar sinais de compra com critérios de validação em múltiplas camadas.

Por se tratar de um sistema desenvolvido com intuito de pesquisa e aprendizado, algumas das 
funcionalidades citadas podem estar comentadas ou com algum tipo de Bupass. Caso queira 
reativa-las é só descomentar o trecho do código.



```Markdown
> **AVISO:** PROJETO DE PESQUISA E APRENDIZADO. NÃO CONSTITUI RECOMENDAÇÃO DE INVESTIMENTO.
> USO EM OPERAÇÕES REAIS É DE INTEIRA RESPONSABILIDADE DO OPERADOR.
```
---

## :bulb: Por que este sistema foi criado

:cold_sweat: Operar no mercado financeiro exige frieza em momentos de pressão — algo difícil para qualquer pessoa e particularmente desafiador para quem, como eu, tende a encerrar operações prematuramente por ansiedade ou medo.

O PX97-Axon nasceu em julho de 2025 não como produto comercial, mas como solução pessoal: um sistema que toma decisões baseadas em dados, mantém critérios rígidos de entrada e saída, e elimina o fator emocional do processo. O resultado foi um modelo que opera com consistência onde eu não conseguia.

Espero que este projeto possa ajudar outras pessoas ou ao menos ser um ponto de partida.


```Markdown
## 🔨 O que o sistema faz

1. **Análise de sentimento no início do dia:** coleta e analisa notícias financeiras brasileiras
via LLM (OpenRouter), gera um score fuzzy entre 0 e 1 representando o humor do mercado e usa esse
valor para calibrar automaticamente três parâmetros operacionais: threshold de entrada, multiplica
dor de alvo e multiplicador de stop
2. Conecta ao MetaTrader 5 e coleta candles M5 do WIN em tempo real
3. Calcula 20 indicadores técnicos (RSI, MACD, Bollinger Bands, ATR, ADX, OBV, entre outros)
4. Aplica engenharia de features: log returns, codificação temporal seno/cosseno
5. Normaliza os dados com MinMaxScaler e alimenta o modelo LSTM principal
6. Aplica segundo filtro via modelo auxiliar (Random Forest) treinado nos logs de operações reais do
próprio sistema — aprende com o histórico de trades executados e valida ou rejeita o sinal do LSTM antes da entrada
7. Avalia a probabilidade de alta contra múltiplos critérios de entrada (threshold, RSI, ADX, DI+/DI-, ATR)
8. Gerencia o trade aberto: alvo e stop dinâmicos baseados no ATR, saída por tempo (TIME_DROP)
9. Controla risco diário: meta de pontos, perda máxima percentual, cooldown entre trades, Equity Guard
10. Comunica bidirecionalmente via Telegram: envia alertas de entrada, saída e status, e recebe comandos remotos do
operador em tempo real
```

## 📌 Arquitetura

```python
px97_axon_trader.py              # Entry point — loop principal de operação
├── config.py                    # Parâmetros globais, carregamento do modelo GRU e scaler
├── funcoes.py                   # Toda a lógica operacional:
│   ├── calculate_indicators()   # Cálculo dos 20 indicadores
│   ├── prepare_input()          # Preparação e normalização para o LSTM
│   ├── decide_entry()           # Lógica de decisão de entrada
│   ├── check_active_tick()      # Monitoramento e saída do trade
│   └── trade_cycle()            # Ciclo completo por iteração
├── telegram_bot.py              # Interface bidirecional com Telegram (thread separada):
│   ├── Envio → alertas de entrada, saída, Equity Guard, status a cada ciclo
│   └── Recepção → comandos remotos do operador via chat
│       ├── /threshold 0.75  — ajusta critério de entrada em tempo real
│       ├── /meta 800        — altera meta diária de pontos
│       ├── /status          — retorna situação atual do robô
│       ├── /parametros      — retorna indicadores do último ciclo
│       ├── /real            — ativa modo real trading
│       └── /simulate        — volta ao modo paper trading
├── modelo_auxiliar.py           # Random Forest treinado nos logs de operações reais:
│   ├── Carrega modelo .joblib com threshold dinâmico otimizado
│   ├── Atua como segundo filtro após o GRU — aprova ou rejeita o sinal
│   └── Retreinado periodicamente com novos logs (bypass ativo durante acumulação)
├── analise_sentimento.py        # Executado no início do dia:
│   ├── Score fuzzy 0–1 (0 = mercado extremamente negativo, 1 = positivo)
│   └── Calibra threshold de entrada, multiplicador de alvo e de stop
└── sentment_analysis.py         # Coleta notícias via RSS + análise por LLM (OpenRouter)
```

**Scripts de treino e coleta (executados offline):**

```
obter_historico_mini_indice.py   # Coleta histórico M5 via MT5 e gera CSV com indicadores
treino_px97_axon_markI_5min.py   # Treina o modelo GRU principal (grid search + checkpoint)
treino_auxiliary_model.py        # Treina o Random Forest auxiliar com logs de operações reais
```

---

## 📌 Features do modelo

O modelo de treino usa **10 features** selecionadas por análise de effect size. O pipeline de produção (`funcoes.py`) calcula um conjunto maior de indicadores, dos quais o scaler filtra apenas as features relevantes.

| Categoria | Feature | Effect size |
|---|---|---|
| Volatilidade | `ATRr_14` | mais importante (RF) |
| Retorno | `log_ret` | — |
| Retorno defasado | `ret_lag1` | autocorr = -0.37 |
| Tendência | `dist_ema` | -0.39 |
| Posição Bollinger | `BBP_5_2.0_2.0` | -0.43 |
| Momentum | `MACDh_12_26_9` | -0.56 (melhor sinal) |
| Momentum | `RSI_14` | -0.23 |
| Tendência | `ema_slope3` | direção EMA em 3 candles |
| Temporal | `hour_sin` / `hour_cos` | codificação cíclica da hora |



## 🏋️ Treinamento do modelo principal

Script: `treino_px97_axon_markI_5min.py`

Arquitetura **GRU minimalista** escolhida após diagnóstico de effect size das features — modelos mais complexos (LSTM com atenção, bidirecional) colapsaram por excesso de parâmetros para o volume disponível de dados (~16k amostras).

| Componente | Detalhe |
|---|---|
| Arquitetura | GRU 32 → GRU 16 → Dense 16 → sigmoid |
| Scaler | RobustScaler (resistente a outliers do WIN) |
| Janela | Grid search: 10, 20 ou 40 candles |
| Features | 10 selecionadas por effect size (> 0.1) |
| Target | Simulação real do trade: alvo 1.65× ATR, stop 1.19× ATR |
| Regularização | L2 (2e-4) + Dropout 0.30 + BatchNormalization |
| Otimizador | Adam (lr=1e-3, clipnorm=1.0) |
| Callbacks | EarlyStopping + ReduceLROnPlateau + ModelCheckpoint |

O target é gerado simulando o trade real sobre o histórico: para cada candle, o algoritmo verifica se nos próximos N candles o preço teria atingido o alvo antes do stop. Casos ambíguos (ambos ou nenhum atingido) são descartados do treino.

---

## 🏋️ Treinamento do modelo auxiliar (2º filtro)

Script: `treino_auxiliary_model.py`

Random Forest treinado exclusivamente sobre os **logs de operações reais** gerados pelo sistema em produção — não sobre dados históricos sintéticos.

| Componente | Detalhe |
|---|---|
| Algoritmo | RandomForestClassifier (100 estimadores) |
| Features | `proba`, `rsi`, `preco_entrada`, `alvo`, `stop` |
| Target | GAIN=1 / LOSS=0 (TIME_DROP excluído) |
| Threshold | Otimizado via curva precision-recall (F1 máximo) |
| Output | `.joblib` com modelo + threshold dinâmico juntos |

O threshold é recalculado a cada retreino com base no histórico real acumulado. Quanto mais operações registradas, mais preciso o filtro.

---

## 📥 Coleta de dados históricos

Script: `obter_historico_mini_indice.py`

Coleta candles M5 do WIN diretamente do MetaTrader 5, calcula os indicadores técnicos e salva CSV pronto para treino.

**Estratégia de coleta:**
1. Tenta o símbolo contínuo `WIN$` (histórico unificado, quando disponível na corretora)
2. Fallback automático para contratos individuais (`WINF19`, `WING19`...) com janela de ±90 dias por vencimento
3. Remove duplicatas de timestamp e linhas com NaN após cálculo de indicadores

```bash
python obter_historico_mini_indice.py
# Output: dados/dados_mini_indice_M5_COM_INDICADORES_<data>.csv
```

---

```Markdown

> Estes parâmetros podem ser alterados de acordo com o tipo de negociação que o usuário deseja(Conservador/mediano/agressivo).

THRESHOLD < proba <= THRESHOLD_LIM   # Confiança do LSTM dentro da faixa calibrada
tendencia == "ALTA"                   # EMA_20 ascendente
40 < RSI < 75~78                      # Momentum sem sobrecompra
280 < ATR < 540                       # Volatilidade operável
20 <= ADX < 45                        # Tendência presente, sem exaustão
(DI+ - DI-) >= 8                      # Direcionalidade confirmada
distancia_ema <= 2.5 * ATR            # Preço não distante demais da média
```

---

## 🛑 Gestão de risco

| Parâmetro | Valor padrão |
|-----------|-------------|
| Alvo | 1.65× ATR |
| Stop | 1.20× ATR |
| Cooldown entre trades | 500 segundos |
| Perda diária máxima | 30% do capital inicial |
| Proteção de lucro (Equity Guard) | Trava ao devolver >50% do topo do dia |
| Tempo máximo de posição | 106 minutos |

---

## 🖥️ Tecnologias

- **Python 3.10+**
- `MetaTrader5` — coleta de dados e execução de ordens
- `Keras / TensorFlow` — modelo LSTM
- `pandas-ta` — indicadores técnicos
- `scikit-learn` — normalização (MinMaxScaler)
- `pandas`, `numpy` — manipulação de dados
- `requests` — comunicação com API do Telegram (implementação própria, sem biblioteca externa)
- `pytz`, `threading` — controle de fuso horário e listener do Telegram em thread paralela

---

## 📌 Como executar

```bash
> MetaTrader 5 deve estar aberto e conectado à corretora antes de executar.
> O bot do Telegram inicia automaticamente em thread separada e confirma conexão via mensagem.

# 1. Instalar dependências
pip install MetaTrader5 keras tensorflow pandas pandas-ta scikit-learn numpy pytz joblib requests
# 2. Configurar config.py (usar config.exemplo.py como base)
# - ATIVO: símbolo do contrato vigente (ex: "WINM26")
# - REAL_TRADE: False para paper trading
# - CAMINHO_MODELO / CAMINHO_SCALER: paths para os arquivos .keras e .pkl
# - TELEGRAM_TOKEN: token do bot gerado via @BotFather
# - TELEGRAM_CHAT_ID: seu chat ID no Telegram

# 3. Executar
python px97_axon_trader.py
```



---

## 🚧 Status do projeto

| Componente | Status |
|------------|--------|
| Coleta de dados MT5 | ✅ Operacional |
| LSTM Mark II (modelo principal) | ✅ Treinado e em uso |
| Paper trading | ✅ Funcional |
| Interface Telegram (alertas + comandos) | ✅ Operacional |
| Modelo auxiliar (Random Forest — 2º filtro) | 🔄 Bypass temporário — acumulando logs para retreino |
| Análise de sentimento de notícias | 🔄 Implementado — requer chave OpenRouter e módulo `sentimento_noticias` |
| Real trading | ⚠️ Disponível — usar com cautela |


```markdown
## Pré-requisitos

- **Sistema**: Windows (MetaTrader 5 requirement)
- **Python**: 3.10 ou superior
- **MetaTrader 5**: instalado e conectado à corretora
- **Telegram**: bot criado via @BotFather

```

## 🤔 Troubleshooting

| Problema | Solução |
|----------|---------|
| "MT5 connection failed" | Abra MT5 antes de rodar o script |
| "Telegram não conecta" | Verifique TELEGRAM_TOKEN e TELEGRAM_CHAT_ID em config.py |
| "LSTM model not found" | Confirme CAMINHO_MODELO aponta para arquivo .keras válido |

---

## 📌 Autor

**José Landy Giorio do Vale**
Desenvolvedor Python | IA & Machine Learning | Mestrando PROARQ/FAU-UFRJ
[linkedin.com/in/joselandy](https://linkedin.com/in/joselandy)

---

## 📌 Licença

GPL-3.0 — veja [LICENSE](LICENSE)

Uso permitido com obrigatoriedade de manter o código aberto em derivados. Autoria registrada desde julho/2025.
