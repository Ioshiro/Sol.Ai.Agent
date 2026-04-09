# Report stato lavori

Data: 2026-04-09

## Obiettivo

Portare il repository a un baseline pulito con questa architettura:

- STT e TTS via API OpenAI
- LLM servito dal progetto `.NET` locale, che inoltra a Runpod
- infrastruttura locale tenuta in Docker Compose per quanto possibile
- agent Python lasciati esterni al Compose

## Stato attuale

### Completato

- Refactor della configurazione per separare:
  - OpenAI per STT/TTS
  - servizio `.NET` locale per LLM
  - Runpod come upstream del servizio `.NET`
- Aggiornati `agent.py` e `agent_SIP.py` per usare i plugin OpenAI di LiveKit e il backend LLM locale.
- Integrato `SolAI.Pipecat.LLMService` nello stack Docker Compose.
- Aggiornati `.env.example`, `README.md` e il report stesso.

### Fuori perimetro volontario

- `agent.py` e `agent_SIP.py` restano worker separati, non containerizzati nel compose.
- OpenAI resta servizio cloud esterno.
- Runpod resta servizio cloud esterno dietro l'orchestratore `.NET`.
- LM Studio non fa più parte dello stack.

## Topologia finale

### Locale, dentro Compose

- `redis`
- `livekit`
- `sip`
- `lk-bootstrap`
- `llm-service`

### Cloud / esterni

- OpenAI per STT e TTS
- Runpod come upstream LLM

### Worker esterni al Compose

- `agent.py`
- `agent_SIP.py`

## Osservazioni tecniche

- Il compose tiene vicino al repo tutto ciò che può essere eseguito in modo stabile come infrastruttura.
- Il servizio `.NET` espone un'interfaccia OpenAI-compatible locale, così gli agent non parlano direttamente con Runpod.
- La qualità e latenza del modello dipendono dal backend Runpod scelto.

## Rischi e punti da verificare

- Il servizio `.NET` richiede che `RUNPOD_LLM_BASE_URL`, `RUNPOD_LLM_MODEL` e `RUNPOD_LLM_API_KEY` siano impostati correttamente.
- Le API OpenAI usate per STT/TTS devono essere raggiungibili dal runtime degli agent.
- Il daemon Docker locale deve essere attivo per build/run del compose.

## Prossimo passo consigliato

Validare `docker compose config` e poi avviare lo stack con `docker compose up --build`.
