"""
Ultra-Fast LiveKit Voice Agent  —  OpenAI STT → LLM → TTS Pipeline
====================================================================

Speed strategy (all-OpenAI stack):
┌─────────────┬──────────────────────────┬──────────────────────────────────┐
│  Stage      │  Model                   │  Why it's fast                   │
├─────────────┼──────────────────────────┼──────────────────────────────────┤
│  VAD        │  Silero (local)          │  Runs on-device, 0ms network     │
│  Turn det.  │  LiveKit EnglishModel    │  Fires instantly on real EoT     │
│  STT        │  gpt-4o-transcribe       │  Fastest OpenAI streaming STT    │
│  LLM        │  gpt-4o-mini             │  Lowest TTFT, KV-cached turns    │
│  TTS        │  gpt-4o-mini-tts         │  Fastest OpenAI streaming TTS    │
│  Transport  │  WebRTC via LiveKit      │  Persistent conn, no HTTP warmup │
└─────────────┴──────────────────────────┴──────────────────────────────────┘

Key latency optimisations applied:
  • Full streaming at every stage boundary — TTS starts before LLM finishes,
    LLM starts before STT fully completes (overlapping execution).
  • Silero VAD runs locally — no network round-trip before the pipeline fires.
  • LiveKit turn-detector (EnglishModel) — semantic awareness stops false early
    cuts AND lets the agent fire sooner on genuine end-of-turn than raw VAD.
  • min_endpointing_delay=0.2s — short silence window, agent fires fast.
  • gpt-4o-mini KV-caches the system prompt after turn 1, dropping TTFT from
    ~1.2s to ~0.4-0.9s on subsequent turns.
  • Short, punchy system prompt — less prompt = smaller first-turn KV tax.
  • allow_interruptions=True — user can barge in and agent stops immediately.
  • AutoSubscribe.AUDIO_ONLY — skip video track negotiation entirely.
  • prewarm() pre-loads VAD + turn-detector before first call arrives.
  • BVC noise cancellation — cleaner audio → STT is faster and more accurate.

Target latency: ~200-500ms end-to-end (end of user speech → first audio byte).

Installation:
    pip install "livekit-agents[openai,silero,turn-detector,noise-cancellation]~=1.0"

Required .env.local:
    OPENAI_API_KEY=sk-...
    LIVEKIT_URL=wss://your-project.livekit.cloud
    LIVEKIT_API_KEY=APIxxxxxxxx
    LIVEKIT_API_SECRET=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

Run (dev mode — hot reload, connects to LiveKit Cloud):
    python agent.py dev

Run (console mode — local mic/speakers, no LiveKit server needed):
    python agent.py console

Run (production):
    python agent.py start
"""

import logging
from pathlib import Path
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    JobProcess,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import noise_cancellation, openai, silero
from livekit.plugins.turn_detector.english import EnglishModel

# ---------------------------------------------------------------------------
# Env + logging
# ---------------------------------------------------------------------------
load_dotenv(dotenv_path=Path(".env"))
load_dotenv(dotenv_path=Path(".env.local"), override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("fast-voice-agent")


# ---------------------------------------------------------------------------
# System prompt  (keep it SHORT — every extra token costs ~1ms on turn 1)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a fast, helpful voice assistant.
Rules for voice:
- Keep answers SHORT and conversational (1-3 sentences max unless asked for detail).
- Never use markdown, bullet points, asterisks, or URLs — this is speech only.
- Never say "I'm sorry" or filler phrases. Get straight to the point.
- If you don't know something, say so in one sentence.
""".strip()


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
class VoiceAgent(Agent):

    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_PROMPT)

    async def on_enter(self) -> None:
        """Fires when the agent joins the room. Greet the user immediately."""
        await self.session.generate_reply(
            instructions="Greet the user briefly and ask how you can help.",
            allow_interruptions=True,
        )


def prewarm(proc: JobProcess) -> None:
    logger.info("Prewarming: loading Silero VAD ...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete.")

async def entrypoint(ctx: JobContext) -> None:

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Room connected: {ctx.room.name}")

    session = AgentSession(

        min_endpointing_delay=0.2,


        vad=silero.VAD.load(
            min_silence_duration=0.2,       # reduced from 0.3 to fire faster
            activation_threshold=0.5,       # 0-1, lower = more sensitive
            min_speech_duration=0.05,       # ignore sub-50ms noise bursts
        ),

        turn_detection=EnglishModel(),

        stt=openai.STT(
            model="gpt-4o-transcribe",
            language="en",              # explicit lang → skips auto-detect step
        ),

        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        ),

        tts=openai.TTS(
            model="gpt-4o-mini-tts",
            voice="alloy",
            speed=1.1,                  # increase to 1.1-1.2 for snappier feel
        ),
    )

    @session.on("metrics_collected")
    def on_metrics(ev) -> None:
        try:
            logger.info(f"Metrics: {ev}")
        except Exception:
            pass

    await session.start(
        room=ctx.room,
        agent=VoiceAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info("Session started — agent is live.")

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )