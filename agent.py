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
# Hardcoded ultra-fast defaults
# ---------------------------------------------------------------------------
STT_MODEL = "gpt-4o-mini-transcribe"
LLM_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "alloy"
TTS_SPEED = 1.2
MIN_ENDPOINTING_DELAY = 0.08
MIN_SILENCE_DURATION = 0.10
VAD_ACTIVATION_THRESHOLD = 0.40


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

        min_endpointing_delay=MIN_ENDPOINTING_DELAY,


        vad=silero.VAD.load(
            min_silence_duration=MIN_SILENCE_DURATION,
            activation_threshold=VAD_ACTIVATION_THRESHOLD,
            min_speech_duration=0.05,       # ignore sub-50ms noise bursts
        ),

        stt=openai.STT(
            model=STT_MODEL,
            language="en",              # explicit lang → skips auto-detect step
        ),

        llm=openai.LLM(
            model=LLM_MODEL,
            temperature=0.3,
        ),

        tts=openai.TTS(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            speed=TTS_SPEED,
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