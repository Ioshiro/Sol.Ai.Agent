# LiveKit Local Voice Demo

Demo di voice agent per una stack ibrida, con il massimo possibile tenuto vicino al repository:

- STT e TTS via API OpenAI
- LLM servito dal progetto `.NET` locale, che a sua volta inoltra a Runpod
- LiveKit, Redis e SIP tenuti in Docker Compose
- gli agent Python restano processi separati perché dipendono dall'host e dalle credenziali cloud

## Architettura

### Dentro `docker compose`

| Servizio | Scopo | Porta |
| --- | --- | --- |
| `redis` | backend Redis per LiveKit | `6379/tcp` |
| `livekit` | server LiveKit in modalità dev | `7880/tcp`, `7881/tcp`, `50000-50020/udp` |
| `sip` | LiveKit SIP | `5060/udp`, `10000-10100/udp` |
| `lk-bootstrap` | inizializza trunk e dispatch rule SIP | one-shot |
| `llm-service` | orchestratore OpenAI-compatible locale | `8080/tcp` nel container, `8081/tcp` sull'host |

### Fuori dal Compose

- `agent.py` — agent principale
- `agent_SIP.py` — worker SIP separato
- OpenAI API — STT/TTS
- Runpod — upstream del servizio `.NET`
- LiveKit turn detector non è usato nel baseline attuale: l'endpointing avviene senza `MultilingualModel`

## Setup

```bash
cp .env.example .env
```

Compila almeno questi valori:

- `LANGFUSE_BASE_URL`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_SECRET_KEY`
- `OPENAI_API_KEY`
- `LLM_SERVICE_BASE_URL`
- `LLM_SERVICE_API_KEY`
- `LLM_SERVICE_MODEL`
- `RUNPOD_LLM_BASE_URL`
- `RUNPOD_LLM_API_KEY`
- `RUNPOD_LLM_MODEL`

## Installare `uv`

Serve solo se vuoi avviare gli agent Python fuori da Docker.

### Windows

Con WinGet:

```powershell
winget install --id=astral-sh.uv -e
```

In alternativa, con PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Se il binario non entra automaticamente nel `PATH`, riapri il terminale oppure aggiungi la directory di installazione al `PATH`.

## Avvio infrastruttura locale

```bash
docker compose up --build
```

Questo avvia Redis, LiveKit, SIP, bootstrap SIP e il servizio `.NET` di orchestrazione LLM.

## Osservabilità Langfuse

Langfuse riceve tre livelli di osservabilità:
- il trace root della chiamata/sessione generato dagli agent Python
- gli span OpenAI automatici per STT, LLM e TTS
- gli span del servizio `.NET` LLM, esportati via OTLP verso Langfuse

Per leggerli in UI:
1. compila `LANGFUSE_BASE_URL`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
2. rilancia gli agent
3. usa `session_id` e `trace_name` per raggruppare e filtrare la chiamata

## Avvio agent

### Agent principale

```bash
uv sync
uv run agent.py console
```

`console` usa audio locale (microfono e speaker dell'host). Su una macchina remota o headless senza device audio disponibili fallisce all'avvio; in quel caso usa una postazione con audio reale oppure verifica i device con `lk-agents console --list-devices`.

### Worker SIP

```bash
uv run agent_SIP.py dev
```

## Variabili d'ambiente principali

```env
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_TRACING_ENVIRONMENT=dev
LANGFUSE_TRACING_RELEASE=initial-baseline
LANGFUSE_DEBUG=False

OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=...
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=alloy

LLM_SERVICE_BASE_URL=http://localhost:8081/v1
LLM_SERVICE_API_KEY=lm-studio
LLM_SERVICE_MODEL=qwen3.5-4b

RUNPOD_LLM_BASE_URL=https://your-runpod-endpoint/v1
RUNPOD_LLM_API_KEY=...
RUNPOD_LLM_MODEL=qwen3.5-4b
```

## Note sul servizio `.NET`

Il servizio `SolAI.Pipecat.LLMService` espone una API OpenAI-compatible locale e inoltra le richieste al modello ospitato su Runpod.
Per avviare tutto dal repository basta `docker compose up --build`.

## Note operative

- `agent.py` e `agent_SIP.py` restano esterni al compose per via del microfono/speaker dell'host e dei flussi di esecuzione interattivi.
- `LANGFUSE_*` accendono l'osservabilità: il root trace degli agent Python e gli span OpenAI automatici per STT/LLM/TTS finiscono in Langfuse.
- `LLM_SERVICE_*` servono agli agent Python per parlare con il servizio `.NET` locale.
- `RUNPOD_LLM_*` sono le variabili “esterne” della demo: `docker compose` le mappa in `LlmService__UpstreamEndpoint`, `LlmService__ApiKey` e `LlmService__DefaultModel` per il servizio `.NET`.
- `LIVEKIT_SIP_OUTBOUND_TRUNK` e `SIP_OUTBOUND_TARGET` servono solo per il flusso outbound.
- LM Studio non è più parte di questo stack.

## Troubleshooting rapido

- Se `llm-service` non risponde, controlla `docker compose logs llm-service`.
- Se `lk-bootstrap` fallisce, controlla `docker compose logs lk-bootstrap`.
- Se Langfuse non mostra tracce, verifica `LANGFUSE_*` nel `.env` e abilita `LANGFUSE_DEBUG=True`.
- Se il worker non parte, verifica `OPENAI_*`, `LLM_SERVICE_*` e `RUNPOD_LLM_*` nel `.env`.
