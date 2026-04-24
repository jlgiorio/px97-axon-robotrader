import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pandas_ta as ta # Importar pandas_ta

# --- Parte 1: Geração da lista de contratos (ampliada) ---
# ... (Seu código da Parte 1, sem alterações) ...
meses = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}
letras = list(meses.keys())

ano_inicio_historico = 2019 # Ou o que você definiu
ano_atual = datetime.today().year
anos = list(range(ano_inicio_historico, ano_atual))

contratos = []
for ano in anos:
    for letra in letras:
        cod = f"WIN{letras[letras.index(letra)]}{str(ano)[2:]}"
        contratos.append(cod)

df_contratos = pd.DataFrame({'contrato': contratos})
df_contratos['ano'] = df_contratos['contrato'].str.extract(r'(\d{2})$').astype(int) + 2000
df_contratos['mes'] = df_contratos['contrato'].str.extract(r'WIN([A-Z])')[0].map(meses)

df_contratos['data_vencimento_estimada'] = pd.to_datetime(
    df_contratos[['ano', 'mes']].rename(columns={'ano': 'year', 'mes': 'month'}).assign(day=1)
)

contratos_finais = df_contratos['contrato'].tolist()

print(f"🧾 Contratos gerados para tentativa de download (total: {len(contratos_finais)}): {contratos_finais[:10]} ... {contratos_finais[-10:]}")


# --- Parte 2: Coleta dos dados históricos via MetaTrader 5 ---
if not mt5.initialize():
    print("❌ Erro ao conectar ao MetaTrader 5. Certifique-se de que o MT5 está aberto e configurado.")
    exit()

all_dfs = []

continuous_symbol = "WIN$" # Verifique e ajuste este símbolo para sua corretora!
timeframe = mt5.TIMEFRAME_M5 # Ou mt5.TIMEFRAME_M5

print(f"Attempting to collect data from continuous symbol: {continuous_symbol} at {timeframe} timeframe")
if mt5.symbol_select(continuous_symbol, True):
    earliest_date_to_try = datetime(2005, 1, 1) # Ajuste este ano conforme o histórico que você precisa/espera.

    rates_continuous = mt5.copy_rates_range(continuous_symbol, timeframe, earliest_date_to_try, datetime.now())

    if rates_continuous is not None and len(rates_continuous) > 0:
        df_continuous = pd.DataFrame(rates_continuous)
        df_continuous['symbol'] = continuous_symbol
        df_continuous['time'] = pd.to_datetime(df_continuous['time'], unit='s')
        all_dfs.append(df_continuous)
        print(f"✅ Coletados {len(df_continuous)} candles do símbolo contínuo '{continuous_symbol}'.")
        # Diagnóstico de volume
        if 'real_volume' in df_continuous.columns and df_continuous['real_volume'].sum() > 0:
            print(f"   📊 Volume Real (real_volume) encontrado e com valores. Exemplo: {df_continuous['real_volume'].head().tolist()}")
        else:
            print(f"   ⚠️ Coluna 'real_volume' não disponível ou vazia. O volume pode ser de tick. Valores: {df_continuous['tick_volume'].head().tolist()}")
    else:
        print(f"⚠️ Sem dados disponíveis para o símbolo contínuo '{continuous_symbol}' ou período especificado.")
else:
    print(f"⚠️ Símbolo contínuo '{continuous_symbol}' não encontrado ou não selecionável no terminal MT5. Tentando contratos individuais.")

# Fallback para contratos individuais (se necessário)
if not all_dfs or len(all_dfs[0]) < 10000:
    print("\n--- Iniciando coleta de dados de contratos individuais ---")
    if all_dfs:
        all_dfs = [] # Limpa se o contínuo não trouxe dados suficientes

    for symbol in contratos_finais:
        print(f"📥 Tentando coletar: {symbol}")
        if not mt5.symbol_select(symbol, True):
            print(f"⚠️ Símbolo '{symbol}' não encontrado ou não selecionável no terminal MT5. Pulando.")
            continue

        linha = df_contratos[df_contratos['contrato'] == symbol]
        if linha.empty:
            print(f"❌ Erro: Informações de data não encontradas para o contrato {symbol}.")
            continue

        linha = linha.iloc[0]
        start_date_individual = datetime(linha['ano'], linha['mes'], 1) - timedelta(days=90)
        end_date_individual = min(datetime.now(), start_date_individual + timedelta(days=180))

        rates = mt5.copy_rates_range(symbol, timeframe, start_date_individual, end_date_individual)

        if rates is not None and len(rates) > 0:
            df_temp = pd.DataFrame(rates)
            df_temp['symbol'] = symbol
            df_temp['time'] = pd.to_datetime(df_temp['time'], unit='s')
            all_dfs.append(df_temp)
            print(f"✅ Coletados {len(df_temp)} candles para {symbol}.")
            # Diagnóstico de volume
            if 'real_volume' in df_temp.columns and df_temp['real_volume'].sum() > 0:
                print(f"   📊 Volume Real (real_volume) encontrado e com valores. Exemplo: {df_temp['real_volume'].head().tolist()}")
            else:
                print(f"   ⚠️ Coluna 'real_volume' não disponível ou vazia. O volume pode ser de tick. Valores: {df_temp['tick_volume'].head().tolist()}")
        else:
            print(f"⚠️ Sem dados disponíveis para {symbol} no período especificado.")

mt5.shutdown()
print("🔌 Conexão com MetaTrader 5 encerrada.")


# --- Parte 3: Consolidação, Cálculo de Indicadores e Salvamento do Dataset ---

if all_dfs:
    df_geral = pd.concat(all_dfs, ignore_index=True)

    # --- NOVO TRECHO DE CÓDIGO PARA TRATAR DUPLICATAS NO TEMPO ---
    # Primeiro, garanta que a coluna 'time' esteja ordenada antes de remover duplicatas
    df_geral.sort_values(by='time', inplace=True)
    # Remova duplicatas na coluna 'time', mantendo a primeira ocorrência
    # Isso garante que cada timestamp seja único no índice
    rows_before_dedupe = len(df_geral)
    df_geral.drop_duplicates(subset=['time'], keep='first', inplace=True)
    rows_after_dedupe = len(df_geral)
    if rows_before_dedupe > rows_after_dedupe:
        print(f"⚠️ {rows_before_dedupe - rows_after_dedupe} linhas com timestamps duplicados removidas.")
    # --- FIM DO NOVO TRECHO ---

    df_geral.set_index('time', inplace=True)
    df_geral.sort_index(inplace=True) # Sort again by index in case drop_duplicates changed order

    print("\n--- Calculando Indicadores Técnicos ---")
    # Garante que as colunas esperadas por pandas_ta existem e estão com nomes corretos
    if 'real_volume' in df_geral.columns and df_geral['real_volume'].sum() > 0:
        df_geral.rename(columns={'real_volume': 'volume'}, inplace=True)
    elif 'tick_volume' in df_geral.columns:
        df_geral.rename(columns={'tick_volume': 'volume'}, inplace=True)
        print("❗ Usando 'tick_volume' como 'volume' para cálculo de indicadores.")
    else:
        df_geral['volume'] = 0 # Cria uma coluna de volume vazia se não houver
        print("❗ Nenhuma coluna de volume encontrada para cálculo de indicadores. 'volume' setado para 0.")

    # Exclui a coluna 'spread' se ela existir e não for necessária
    if 'spread' in df_geral.columns:
        df_geral.drop(columns=['spread'], errors='ignore', inplace=True)

    # Renomeia as colunas OHLC para minúsculas, se necessário
    # Mantém a coluna 'symbol' intacta, se existir
    new_columns = {}
    for col in df_geral.columns:
        if col.lower() in ['open', 'high', 'low', 'close', 'volume', 'symbol']:
            new_columns[col] = col.lower()
        else:
            new_columns[col] = col # Mantenha outros nomes de coluna como estão
    df_geral.rename(columns=new_columns, inplace=True)


    # Cálculo dos Indicadores usando pandas_ta
    df_geral.ta.sma(length=10, append=True)
    df_geral.ta.ema(length=20, append=True)
    df_geral.ta.rsi(length=14, append=True)
    df_geral.ta.macd(append=True)
    df_geral.ta.bbands(append=True)
    df_geral.ta.atr(length=14, append=True)
    df_geral.ta.obv(append=True)
    

    print(f"✅ Indicadores calculados. Total de colunas: {len(df_geral.columns)}")

    # Tratamento de NaN após o cálculo dos indicadores
    original_rows = len(df_geral)
    df_geral.dropna(inplace=True)
    print(f"⚠️ {original_rows - len(df_geral)} linhas com NaN removidas após cálculo de indicadores.")

    output_filename = "dados\\dados_mini_indice_M5_COM_INDICADORES_07-04.csv"

    # Garante que 'symbol' seja mantido se existir e 'volume' seja o volume escolhido
    cols_to_save = ['open', 'high', 'low', 'close']
    if 'volume' in df_geral.columns: # Agora 'volume' já está renomeado
        cols_to_save.append('volume')
    if 'symbol' in df_geral.columns:
        cols_to_save.append('symbol')

    # Adiciona todas as colunas de indicadores
    for col in df_geral.columns:
        if col not in cols_to_save:
            cols_to_save.append(col)

    df_geral[cols_to_save].to_csv(output_filename)
    print(f"✅ Dados consolidados com indicadores salvos em: {output_filename}")
    print(f"Total de linhas no arquivo consolidado: {len(df_geral)}")
    print(f"Período dos dados: De {df_geral.index.min()} a {df_geral.index.max()}")
    print(f"Colunas salvas: {df_geral[cols_to_save].columns.tolist()}")

else:
    print("❌ Nenhum dado foi coletado. O arquivo CSV não será gerado.")