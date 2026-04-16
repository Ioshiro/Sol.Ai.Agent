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

- `OPENAI_API_KEY`
- `LlmService__UpstreamEndpoint`
- `LlmService__ApiKey`
- `LlmService__DefaultModel`

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

## Avvio agent

### Agent principale

```bash
uv sync
uv run agent.py console
```

### Worker SIP

```bash
uv run agent_SIP.py dev
```

## Variabili d'ambiente principali

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=...
OPENAI_STT_MODEL=gpt-4o-mini-transcribe
OPENAI_TTS_MODEL=gpt-4o-mini-tts
OPENAI_TTS_VOICE=alloy
LlmService__UpstreamEndpoint=https://your-runpod-endpoint/v1
LlmService__ApiKey=...
LlmService__DefaultModel=qwen3.5-4b
```

## Note sul servizio `.NET`

Il servizio `SolAI.Pipecat.LLMService` espone una API OpenAI-compatible locale e inoltra le richieste al modello ospitato su Runpod.
Per avviare tutto dal repository basta `docker compose up --build`.

## Note operative

- `agent.py` e `agent_SIP.py` restano esterni al compose per via del microfono/speaker dell'host e dei flussi di esecuzione interattivi.
- `LIVEKIT_SIP_OUTBOUND_TRUNK` e `SIP_OUTBOUND_TARGET` servono solo per il flusso outbound.
- LM Studio non è più parte di questo stack.

## Troubleshooting rapido

- Se `llm-service` non risponde, controlla `docker compose logs llm-service`.
- Se `lk-bootstrap` fallisce, controlla `docker compose logs lk-bootstrap`.
- Se il worker non parte, verifica le credenziali OpenAI e Runpod nel `.env`.
