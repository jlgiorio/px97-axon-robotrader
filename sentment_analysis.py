import time
import requests
import json
import os
import json
import random
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv pode não estar instalado; seguimos sem erro
    pass

    
def call_openrouter(messages: list[dict], model: str | None = None, api_key: str | None = None, retries: int = 3, timeout: int = 45) -> dict | None:
  api_key = api_key or os.getenv("SUA_OPENROUTER_API_KEY")
  model = model or os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free")

  if not api_key:
    print("Erro: SUA_OPENROUTER_API_KEY não definido. Configure a variável de ambiente.")
    return None

  headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-Title": "finacebot-form-bot",
    "HTTP-Referer": "http://localhost:8501/",
  }

  payload = {"model": model, "messages": messages}

  for attempt in range(1, retries + 1):
    try:
      resp = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout,
      )
    except requests.RequestException as e:
      print(f"[Tentativa {attempt}] Erro de rede ao chamar OpenRouter: {e}")
      # backoff e tenta novamente
      if attempt < retries:
        time.sleep(min(2 ** attempt + random.random(), 10))
        continue
      return None

    # Tenta sempre decodificar
    try:
      body = resp.json()
    except ValueError:
      print(f"[Tentativa {attempt}] Resposta não-JSON (status {resp.status_code}): {resp.text[:500]}")
      if attempt < retries and resp.status_code >= 500:
        time.sleep(min(2 ** attempt + random.random(), 10))
        continue
      return None

    if resp.status_code == 200:
      return body

    # Trata 429/5xx com retry
    if resp.status_code in (429, 500, 502, 503, 504):
      err_msg = body.get("error", {}).get("message") or body.get("message") or str(body)
      print(f"[Tentativa {attempt}] Erro transitório do OpenRouter ({resp.status_code}): {err_msg}")
      if attempt < retries:
        time.sleep(min(2 ** attempt + random.random(), 10))
        continue
      return body

    # Outros erros (4xx) — não adianta tentar novamente
    err_msg = body.get("error", {}).get("message") or body.get("message") or str(body)
    print(f"Erro do OpenRouter (status {resp.status_code}): {err_msg}")
    return body

  return None

def get_resultados():
    """
    Executa a análise e retorna um dicionário com os campos:
    'resultado_ponderado' (float) e 'previsao_mercado' (str).
    Sempre retorna valores padrão se houver erro.
    """
    model = os.getenv("OPENROUTER_MODEL", "openai/gpt-oss-20b:free")
    result = call_openrouter(messages=messages, model=model)
    if result and result.get("choices"):
        content = result["choices"][0]["message"]["content"]
        if not content:
            print("Conteúdo vazio do modelo.")
            return {"resultado_ponderado": 0.5, "previsao_mercado": "neutro"}
        # Remove blocos markdown se existirem
        if content.strip().startswith("```"):
            content = content.strip().strip("`")
            # Remove o marcador de linguagem se houver
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        if not content.startswith("{"):
            print("Conteúdo ainda não é JSON puro:", content)
            return {"resultado_ponderado": 0.5, "previsao_mercado": "neutro"}
        try:
            parsed = json.loads(content)
            return {
                "resultado_ponderado": parsed.get("resultado_ponderado", 1),
                "previsao_mercado": parsed.get("previsao_mercado", "neutro")
            }
        except Exception as e:
            print("Erro ao decodificar JSON:", e)
            print("Conteúdo retornado:", content)
    return {"resultado_ponderado": 0.5, "previsao_mercado": "neutro"}

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

system_prompt = (
  "Você é um analista financeiro experiente especializado no mercado brasileiro. "
  "Sua tarefa é buscar notícias financeiras, políticas, sociais e econômicas publicadas no Brasil, "
  "e Analisar os mercados internacionais, políticas governamentais, decisões econômicas, eventos sociais e tendências de mercado que possam influenciar o cenário financeiro brasileiro. "
  "Use exclusivamente os rótulos fornecidos e mantenha consistência e objetividade nas classificações."
  
)

user_prompt = (
    "busque as noticias nos meios de comunicação brasileiros relativos a mercado financeiro, economia, politica, internacional, juros, fiscal, tecnologia, saúde e outros.\n"
    "Considere apenas noticias publicadas nos últimos 7 dias e com potencial de dar direção ao mercado financeiro.\n"
    f"mas considere apenas as que possam impactar o humor do mercado nacional (bolsa de valores) de forma positiva ou megativa\n"
    f"tendo as noticias defina um peso para cada uma dependendo do impacto (Negativo,neutro ou positivo) e o quanto essa noticia pode impactar o mercado (peso)\n"
    f"tendo esses valores, preciso que gere um coeficiente entre 0 e 1, utilizando fuzzy, onde 0 é um mercado extremamente negativo e 1 extremamente positivo\n"
    "Responda no seguinte formato JSON, que deve obrigatoriamente conter os campos: noticias (noticias coletadas), 'resultado_ponderado' (float entre 0 e 1) e 'previsao_mercado' (string: negativo, neutro ou positivo).\n"
    "Para o campo previsao_mercado, utilize 'negativo' se o resultado_ponderado for menor que 0.45, 'neutro' se estiver entre 0.45 e 0.65, e 'positivo' se for maior que 0.65.\n"
)

messages = [
  {"role": "system", "content": system_prompt},
  {"role": "user", "content": user_prompt},
]

resultado, previsao = get_resultados().values()

print("Resultado Ponderado:", resultado)
print("Previsão de Mercado:", previsao)