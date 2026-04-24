# === Módulo de Análise de Sentimento Avançado para o Robô PX-97 ===
# OBJETIVO: Coletar notícias financeiras, aplicar modelo especializado e gerar scores
# de sentimento para otimizar decisões do robô de trading

import torch
import os
import numpy as np
import pandas as pd
import feedparser
import requests
from datetime import datetime, timedelta
import pytz
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from bs4 import BeautifulSoup
import re
import joblib
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURAÇÕES GERAIS ===
RSS_URLS = [
    "https://www.infomoney.com.br/feed/",
    "https://www.infomoney.com.br/economia/feed/",
    "https://www.infomoney.com.br/mercados/feed/",
    "https://br.investing.com/rss/news_285.rss",  # Notícias gerais
    "https://br.investing.com/rss/news_14.rss",   # Análise
    "https://br.investing.com/rss/news_289.rss",  # Economia brasileira
    "https://valorinveste.globo.com/rss/noticias/",
    "https://g1.globo.com/rss/g1/economia/",
    "https://www.cnnbrasil.com.br/economia/feed/"
]

MAX_NOTICIAS = 100
TZ = pytz.timezone('America/Sao_Paulo')
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'


# === MODELO DE SENTIMENTO OTIMIZADO ===
# Modelos públicos e estáveis (ordem de preferência). Mantidos mínimos para reduzir falhas de carregamento.
MODELOS_DISPONIVEIS = [
    # Multilíngue robusto (3 classes: negative/neutral/positive). Requer tokenizer "slow".
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    # Multilíngue (5 estrelas). Útil como fallback amplo e estável no Hub.
    "nlptown/bert-base-multilingual-uncased-sentiment",
]

# Palavras-chave para filtrar notícias relevantes
PALAVRAS_CHAVE_FINANCEIRAS = [
    'bolsa', 'ibovespa', 'dólar', 'juros', 'selic', 'inflação', 'bcb', 'banco central',
    'ações', 'mercado', 'investimento', 'economia', 'financeiro', 'taxa', 'cdi',
    'win', 'dólar', 'juro', 'ipca', 'pib', 'commodities', 'petróleo', 'minério',
    'banco', 'bancário', 'renda fixa', 'renda variável', 'fundo', 'investidor',
    'b3', 'bolsa de valores', 'index', 'futuro', 'opção', 'trade', 'trading',
    'lucro', 'prejuízo', 'resultado', 'balanço', 'dividendo', 'empresa', 'empresarial',
    'empresas', 'empresário', 'empresária', 'empresariais', 'empresários', 'empresárias'
]

# === CLASSE PRINCIPAL DE ANÁLISE DE SENTIMENTO ===
class AnalisadorSentimentoTrading:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.sentiment_pipeline = None
        self.modelo_carregado = None
        self.carregar_modelo()
        self.historico = pd.DataFrame(columns=['timestamp', 'score', 'n_noticias', 'parametros'])

        # Flag para habilitar/desabilitar a análise de notícias sem alterar outros módulos.
        # Defina a variável de ambiente SENTIMENTO_NOTICIAS_ENABLED=1 para reativar.
        self.enabled = str(os.getenv("SENTIMENTO_NOTICIAS_ENABLED", "1")).lower() in ("1", "true", "yes", "on")

        # LIMIARES OTIMIZADOS - mais sensíveis
        self.limiares = {
            'muito_negativo': -0.25,  # Ajustado para ser mais sensível
            'negativo': -0.10,
            'neutro': 0.10,
            'positivo': 0.25,
            'muito_positivo': 0.45
        }

        # AJUSTES DE PARÂMETROS MAIS EFETIVOS
        self.ajustes_parametros = {
            'muito_negativo': {'THRESHOLD': +0.8, 'ALVO_MULT': 0.6, 'STOP_MULT': 1.4},
            'negativo': {'THRESHOLD': +0.06, 'ALVO_MULT': 0.8, 'STOP_MULT': 1.2},
            'neutro': {'THRESHOLD': 0.0, 'ALVO_MULT': 1.0, 'STOP_MULT': 1.0},
            'positivo': {'THRESHOLD': -0.08, 'ALVO_MULT': 1.2, 'STOP_MULT': 0.85},
            'muito_positivo': {'THRESHOLD': -0.15, 'ALVO_MULT': 1.5, 'STOP_MULT': 0.7}
        }

    def carregar_modelo(self):
        """Carrega modelo otimizado para análise de sentimento financeiro"""
        print("Carregando modelo de análise de sentimento...")
        
        # Tentar carregar modelos na ordem de preferência
        for modelo in MODELOS_DISPONIVEIS:
            try:
                print(f"Tentando carregar: {modelo}")
                # Para modelos baseados em RoBERTa/XLM-R, usar tokenizer "slow" para evitar erros de conversão (SentencePiece/Tiktoken).
                lower = modelo.lower()
                if ("roberta" in lower) or ("xlm" in lower):
                    tok = AutoTokenizer.from_pretrained(modelo, use_fast=False)
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=modelo,
                        tokenizer=tok
                    )
                else:
                    self.sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=modelo,
                        tokenizer=modelo
                    )
                self.modelo_carregado = modelo
                print(f"✅ Modelo {modelo} carregado com sucesso!")
                return
            except Exception as e:
                print(f"❌ Erro ao carregar {modelo}: {e}")
                continue
        
        # Fallback para modelo básico
        try:
            print("Tentando carregar modelo básico...")
            self.sentiment_pipeline = pipeline("sentiment-analysis")
            self.modelo_carregado = "modelo-basico"
            print("✅ Modelo básico carregado")
        except Exception as e:
            print(f"❌ Erro crítico: Não foi possível carregar nenhum modelo: {e}")
            # Criar um pipeline mock para evitar quebrar o código
            self.sentiment_pipeline = None
            self.modelo_carregado = "mock"

    def analisar_sentimento_mock(self, texto):
        """Análise de sentimento mock para quando o modelo não está disponível"""
        # Simples análise baseada em palavras-chave
        texto_lower = texto.lower()
        palavras_positivas = ['alta', 'lucro', 'crescimento', 'positivo', 'ganho', 'valorização']
        palavras_negativas = ['queda', 'perda', 'negativo', 'baixa', 'prejuízo', 'queda']
        
        positivas = sum(1 for palavra in palavras_positivas if palavra in texto_lower)
        negativas = sum(1 for palavra in palavras_negativas if palavra in texto_lower)
        
        if positivas > negativas:
            return [{'label': 'POSITIVE', 'score': 0.7}]
        elif negativas > positivas:
            return [{'label': 'NEGATIVE', 'score': 0.7}]
        else:
            return [{'label': 'NEUTRAL', 'score': 0.5}]

    def eh_noticia_financeira(self, texto):
        """Verifica se a notícia é relevante para o mercado financeiro"""
        if not texto:
            return False
            
        texto_lower = texto.lower()
        
        # Verificar palavras-chave
        palavras_encontradas = [palavra for palavra in PALAVRAS_CHAVE_FINANCEIRAS 
                               if palavra in texto_lower]
        
        # Considerar relevante se encontrar pelo menos 2 palavras-chave
        return len(palavras_encontradas) >= 2

    def coletar_noticias(self):
        """Coleta e filtra notícias financeiras relevantes"""
        noticias = []
        agora = datetime.now(TZ)
        limite_tempo = agora - timedelta(hours=12)  # Ampliado para 12 horas

        headers = {'User-Agent': USER_AGENT}

        for url in RSS_URLS:
            try:
                print(f"📰 Coletando de: {url.split('/')[-2]}")
                feed = feedparser.parse(url)

                for entry in feed.entries[:20]:  # Aumentado o limite
                    try:
                        # Tratamento de data
                        if hasattr(entry, 'published_parsed'):
                            publicado_dt = datetime(*entry.published_parsed[:6], tzinfo=TZ)
                        elif hasattr(entry, 'updated_parsed'):
                            publicado_dt = datetime(*entry.updated_parsed[:6], tzinfo=TZ)
                        else:
                            publicado_dt = agora

                        titulo = entry.title if hasattr(entry, 'title') else "Sem título"
                        conteudo = ""

                        if hasattr(entry, 'summary'):
                            conteudo = entry.summary
                        elif hasattr(entry, 'description'):
                            conteudo = entry.description

                        conteudo = self.limpar_texto(conteudo)
                        
                        # FILTRAR APENAS NOTÍCIAS FINANCEIRAS RELEVANTES
                        texto_completo = f"{titulo} {conteudo}"
                        if not self.eh_noticia_financeira(texto_completo):
                            continue

                        if publicado_dt > limite_tempo:
                            noticias.append({
                                'titulo': titulo,
                                'conteudo': conteudo,
                                'data': publicado_dt,
                                'fonte': url,
                                'relevancia': len([p for p in PALAVRAS_CHAVE_FINANCEIRAS 
                                                 if p in texto_completo.lower()])
                            })

                    except Exception as e:
                        continue

            except Exception as e:
                print(f"❌ Erro no feed {url}: {e}")
                continue

            if len(noticias) >= MAX_NOTICIAS:
                break

        # Ordenar por relevância e data
        noticias.sort(key=lambda x: (x['relevancia'], x['data']), reverse=True)
        return noticias[:MAX_NOTICIAS]  # Manter apenas as mais relevantes

    def limpar_texto(self, texto):
        """Limpa e prepara o texto para análise"""
        if not texto:
            return ""

        texto = BeautifulSoup(texto, 'html.parser').get_text()
        texto = re.sub(r'[^\w\s]', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        
        return texto.strip()

    def analisar_sentimento_noticias(self, noticias):
        """Analisa sentimento com modelo otimizado"""
        if not noticias:
            return 0.0, [], "Nenhuma notícia financeira relevante"

        textos = []
        for n in noticias:
            # Usar título + início do conteúdo (limitado a 300 chars)
            texto = f"{n['titulo']} {n['conteudo'][:300]}"
            textos.append(texto)

        scores = []
        resultados_detalhados = []
        pesos = []

        print(f"🔍 Analisando {len(textos)} notícias financeiras...")

        try:
            for i, texto in enumerate(textos):
                try:
                    # Usar análise mock se o pipeline não estiver disponível
                    if self.sentiment_pipeline is None:
                        resultado = self.analisar_sentimento_mock(texto)
                    else:
                        resultado = self.sentiment_pipeline(texto[:512])  # Limitar tamanho
                    
                    score = self.calcular_score_sentimento(resultado)
                    # Extrair rótulo/score crus do modelo (útil para auditoria)
                    try:
                        if isinstance(resultado, list) and len(resultado) > 0 and isinstance(resultado[0], dict):
                            rotulo_cru = str(resultado[0].get('label', ''))
                            confianca_crua = float(resultado[0].get('score', 0.0))
                        elif isinstance(resultado, dict):
                            rotulo_cru = str(resultado.get('label', ''))
                            confianca_crua = float(resultado.get('score', 0.0))
                        else:
                            rotulo_cru = ''
                            confianca_crua = 0.0
                    except Exception:
                        rotulo_cru = ''
                        confianca_crua = 0.0
                    
                    # Ponderar por relevância e recenticidade
                    relevancia = noticias[i]['relevancia']
                    horas_atras = (datetime.now(TZ) - noticias[i]['data']).total_seconds() / 3600
                    peso = relevancia * (1 / (1 + horas_atras))  # Notícias recentes e relevantes têm mais peso
                    
                    scores.append(score)
                    pesos.append(peso)

                    resultado_detalhado = {
                        'titulo': noticias[i]['titulo'][:100],
                        'data': noticias[i]['data'],
                        'fonte': noticias[i]['fonte'],
                        'score': score,
                        'peso': peso,
                        'relevancia': relevancia,
                        'label_cru': rotulo_cru,
                        'confianca_crua': confianca_crua,
                        'analise_crua': str(resultado)
                    }
                    resultados_detalhados.append(resultado_detalhado)
                    
                except Exception as e:
                    print(f"❌ Erro na análise da notícia {i}: {e}")
                    continue

            if scores and pesos:
                # Normalizar pesos
                pesos_array = np.array(pesos)
                if pesos_array.sum() > 0:
                    pesos_normalizados = pesos_array / pesos_array.sum()
                    score_final = np.average(scores, weights=pesos_normalizados)
                else:
                    score_final = np.mean(scores) if scores else 0.0
            else:
                score_final = 0.0

            return score_final, resultados_detalhados, "Análise concluída"

        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return 0.0, [], f"Erro: {str(e)}"

    def salvar_analise_csv(self, resultados_detalhados, score_final, threshold, alvo_mult, stop_mult):
        """Salva a análise detalhada das notícias em CSV para auditoria/validação."""
        try:
            if not resultados_detalhados:
                return None

            os.makedirs('noticias', exist_ok=True)
            ts = datetime.now(TZ).strftime('%Y%m%d_%H%M%S')
            caminho = os.path.join('analises_sentimento', f'detalhes_sentimento_analysis_{ts}.csv')

            df = pd.DataFrame(resultados_detalhados)
            # Anexar metadados da execução para cada linha (facilita filtro posterior)
            df['modelo'] = self.modelo_carregado
            df['score_final'] = float(score_final)
            df['threshold'] = float(threshold)
            df['alvo_mult'] = float(alvo_mult)
            df['stop_mult'] = float(stop_mult)
            df['timestamp_exec'] = datetime.now(TZ)

            df.to_csv(caminho, index=False, encoding='utf-8')
            print(f"📝 Resultado detalhado salvo em: {caminho}")
            return caminho
        except Exception as e:
            print(f"⚠️  Não foi possível salvar CSV de análise: {e}")
            return None

    def calcular_score_sentimento(self, resultado_analise):
        """Calcula score de sentimento otimizado"""
        try:
            if isinstance(resultado_analise, list) and len(resultado_analise) > 0:
                resultado = resultado_analise[0]
            else:
                resultado = resultado_analise

            label = resultado['label'].lower()
            score = resultado['score']

            # Mapeamento de labels para scores numéricos
            # 1) Mapeamento direto (3 classes)
            if any(word in label for word in ['positive', 'positivo', 'bullish', 'alta', 'pos']):
                return score  # Positivo
            elif any(word in label for word in ['negative', 'negativo', 'bearish', 'baixa', 'neg']):
                return -score  # Negativo
            elif any(word in label for word in ['neutral', 'neutro', 'neu']):
                return score * 0.1  # Neutro com peso baixo
            # 2) Modelos de "estrelas" (ex.: nlptown -> "1 star" .. "5 stars")
            elif 'star' in label:
                try:
                    n = int(label.split()[0])  # captura o número de estrelas inicial
                    # 1-2 estrelas: negativo | 3: neutro | 4-5: positivo
                    if n <= 2:
                        return -max(0.5, score)  # força sinal negativo moderado
                    elif n == 3:
                        return 0.0  # neutro
                    else:
                        return max(0.5, score)  # positivo moderado
                except Exception:
                    pass
            else:
                # Fallback baseado no score
                if score > 0.6:
                    return score * 0.5  # Levemente positivo
                else:
                    return 0.0  # Neutro

        except Exception as e:
            print(f"❌ Erro no cálculo do score: {e}")
            return 0.0

    def determinar_parametros_trading(self, score_sentimento, contexto_mercado=None):
        """Determina parâmetros com base no sentimento"""
        # Categorizar sentimento
        if score_sentimento <= self.limiares['muito_negativo']:
            categoria = 'muito_negativo'
            sentimento_desc = "⚠️  SENTIMENTO MUITO NEGATIVO - Reduzir exposição"
        elif score_sentimento <= self.limiares['negativo']:
            categoria = 'negativo'
            sentimento_desc = "⚠️  SENTIMENTO NEGATIVO - Operar com cautela"
        elif score_sentimento <= self.limiares['neutro']:
            categoria = 'neutro'
            sentimento_desc = "📊 SENTIMENTO NEUTRO - Mercado lateralizado"
        elif score_sentimento <= self.limiares['positivo']:
            categoria = 'positivo'
            sentimento_desc = "📈 SENTIMENTO POSITIVO - Oportunidades moderadas"
        else:
            categoria = 'muito_positivo'
            sentimento_desc = "🚀 SENTIMENTO MUITO POSITIVO - Boas oportunidades"

        ajustes = self.ajustes_parametros[categoria].copy()

        # Ajustes adicionais baseados no contexto
        if contexto_mercado:
            volatilidade = contexto_mercado.get('volatilidade', 0.5)
            if volatilidade > 0.7:
                ajustes['STOP_MULT'] *= 1.2
            elif volatilidade < 0.3:
                ajustes['ALVO_MULT'] *= 1.1

        return ajustes['THRESHOLD'], ajustes['ALVO_MULT'], ajustes['STOP_MULT'], sentimento_desc

    def executar_analise_completa(self, contexto_mercado=None):
        """Executa análise completa"""
        print("\n" + "="*50)
        print("📊 ANÁLISE DE SENTIMENTO FINANCEIRO")
        print("="*50)

        # Atalho: se desabilitado, retorna neutro sem coletar notícias
        if not self.enabled:
            print("ℹ️  Análise de notícias está desativada (SENTIMENTO_NOTICIAS_ENABLED=0). Retornando neutro.")
            resultados = {
                'score_sentimento': 0.0,
                'n_noticias': 0,
                'THRESHOLD': 0.0,
                'ALVO_MULT': 1.0,
                'STOP_MULT': 1.0,
                'mensagem': 'Análise de notícias desativada',
                'status': 'DESATIVADO'
            }
            return 0.0, resultados, [], [], 'Análise de notícias desativada'

        # Coletar notícias filtradas
        noticias = self.coletar_noticias()
        print(f"✅ {len(noticias)} notícias financeiras relevantes")

        if len(noticias) < 3:  # Mínimo reduzido de notícias para análise
            print("⚠️  Poucas notícias relevantes - usando análise neutra")
            return 0.0, {
                'score_sentimento': 0.0,
                'n_noticias': len(noticias),
                'THRESHOLD': 0.0,
                'ALVO_MULT': 1.0,
                'STOP_MULT': 1.0,
                'mensagem': 'Poucas notícias relevantes - mercado neutro',
                'status': 'NEUTRO_POR_FALTA_DE_DADOS'
            }, [], noticias, "Mercado neutro por falta de dados"

        # Analisar sentimento
        score, resultados_detalhados, status = self.analisar_sentimento_noticias(noticias)

        # Determinar parâmetros
        THRESHOLD, ALVO_MULT, STOP_MULT, sentimento_desc = self.determinar_parametros_trading(
            score, contexto_mercado
        )

        # Registrar análise
        self.registrar_analise(score, len(noticias), {
            'THRESHOLD': THRESHOLD,
            'ALVO_MULT': ALVO_MULT,
            'STOP_MULT': STOP_MULT
        })

        # Preparar resultados
        resultados = {
            'score_sentimento': float(score),
            'n_noticias': len(noticias),
            'THRESHOLD': float(THRESHOLD),
            'ALVO_MULT': float(ALVO_MULT),
            'STOP_MULT': float(STOP_MULT),
            'mensagem': sentimento_desc,
            'status': status
        }

        print(f"\n📈 RESULTADO DA ANÁLISE:")
        print(f"   Score: {score:.3f}")
        print(f"   Threshold: {THRESHOLD:+.3f}")
        print(f"   Alvo Multiplicador: {ALVO_MULT:.2f}x")
        print(f"   Stop Multiplicador: {STOP_MULT:.2f}x")
        print(f"   Recomendação: {sentimento_desc}")

        # Salvar auditoria das notícias analisadas
        try:
            self.salvar_analise_csv(resultados_detalhados, score, THRESHOLD, ALVO_MULT, STOP_MULT)
        except Exception:
            pass

        return score, resultados, resultados_detalhados, noticias, sentimento_desc

    def registrar_analise(self, score, n_noticias, parametros):
        """Registra análise no histórico"""
        registro = pd.DataFrame([{
            'timestamp': datetime.now(TZ),
            'score': score,
            'n_noticias': n_noticias,
            'parametros': str(parametros)
        }])

        self.historico = pd.concat([self.historico, registro], ignore_index=True)

        if len(self.historico) > 1000:
            self.historico = self.historico.iloc[-1000:]

# === FUNÇÃO PRINCIPAL ===
def main():
    """Função principal para análise de sentimento"""
    print("=== SISTEMA DE ANÁLISE DE SENTIMENTO PARA TRADING ===\n")

    analisador = AnalisadorSentimentoTrading()

    contexto_exemplo = {
        'volatilidade': 0.5,
        'tendencia': 0.2
    }

    try:
        score, resultados, resultados_detalhados, noticias, sentimento_desc = analisador.executar_analise_completa(contexto_exemplo)

        # Exibir notícias mais relevantes
        if resultados_detalhados and len(resultados_detalhados) > 0:
            print(f"\n📋 NOTÍCIAS MAIS RELEVANTES:")
            resultados_ordenados = sorted(resultados_detalhados, 
                                        key=lambda x: x['peso'], 
                                        reverse=True)[:3]
            
            for i, detalhe in enumerate(resultados_ordenados):
                if isinstance(detalhe, dict):
                    data_str = detalhe['data'].strftime('%H:%M') if hasattr(detalhe['data'], 'strftime') else 'N/A'
                    print(f"   {i+1}. [{data_str}] {detalhe['titulo']}")
                    print(f"      Score: {detalhe.get('score', 0):.3f} | Relevância: {detalhe.get('relevancia', 0)}")

        # Determinar decisão final
        score_final = resultados['score_sentimento']
        
        if score_final <= -0.15:
            decisao = "TENDÊNCIA DE VENDA"
            peso = -0.6
        elif score_final <= -0.05:
            decisao = "LEVE TENDÊNCIA DE VENDA" 
            peso = -0.3
        elif score_final <= 0.05:
            decisao = "MERCADO NEUTRO"
            peso = 0.0
        elif score_final <= 0.15:
            decisao = "LEVE TENDÊNCIA DE COMPRA"
            peso = 0.3
        else:
            decisao = "TENDÊNCIA DE COMPRA"
            peso = 0.6

        print(f"\n🎯 DECISÃO FINAL: {decisao} (Score: {score_final:.3f})")

        return {
            'score_final': score_final,
            'decisao': decisao,
            'peso': peso,
            'parametros': {
                'THRESHOLD': resultados.get('THRESHOLD', 0.0),
                'ALVO_MULT': resultados.get('ALVO_MULT', 1.0),
                'STOP_MULT': resultados.get('STOP_MULT', 1.0)
            },
            'n_noticias_analisadas': resultados.get('n_noticias', 0)
        }

    except Exception as e:
        print(f"❌ Erro durante a execução: {e}")
        return {
            'score_final': 0.0,
            'decisao': "ERRO NA ANÁLISE",
            'peso': 0.0,
            'parametros': {'THRESHOLD': 0.0, 'ALVO_MULT': 1.0, 'STOP_MULT': 1.0},
            'n_noticias_analisadas': 0
        }

if __name__ == "__main__":
    resultado_final = main()