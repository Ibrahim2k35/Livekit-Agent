import logging
from pathlib import Path

from dotenv import load_dotenv
from openai.types import realtime

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
logger = logging.getLogger("realtime-voice-agent")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """
You are a fast, helpful voice assistant.
Rules for voice:
- Always respond in English only, even if the user speaks another language.
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
        super().__init__(
            instructions=SYSTEM_PROMPT,
            # Override the session-level RealtimeModel per-agent if needed.
            # Left unset here so it inherits from AgentSession below.
        )

    async def on_enter(self) -> None:
        """Greet the user as soon as the agent joins."""
        await self.session.generate_reply(
            instructions="Greet the user briefly in English and ask how you can help.",
            allow_interruptions=True,
        )


# ---------------------------------------------------------------------------
# Prewarm — pre-load Silero VAD before first call arrives.
# Even though the Realtime API has its own built-in VAD, Silero is still
# used by LiveKit as a local gating layer to avoid sending silence to the
# WebSocket, saving bandwidth and reducing spurious turn triggers.
# ---------------------------------------------------------------------------
def prewarm(proc: JobProcess) -> None:
    logger.info("Prewarming: loading Silero VAD ...")
    proc.userdata["vad"] = silero.VAD.load()
    logger.info("Prewarm complete.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Room connected: {ctx.room.name}")

    # ------------------------------------------------------------------
    # OpenAI Realtime model configuration
    # ------------------------------------------------------------------
    realtime_model = openai.realtime.RealtimeModel(
        # Model: gpt-4o-realtime-preview is the standard Realtime model.
        model="gpt-4o-realtime-preview",

        # Voice options: alloy | ash | ballad | coral | echo |
        #                fable | onyx | nova | sage | shimmer | verse
        voice="alloy",
        speed=1.1, 

        # Modalities: "audio" only = pure speech-to-speech (fastest).
        # Use ["text", "audio"] if you also want text transcripts.
        modalities=["audio"],

        # Temperature: lower = more deterministic, slightly faster.
        temperature=0.5,

        # Input audio transcription — set a model here if you want
        # the user's speech transcribed to text in your logs.
        # Comment out to skip transcription entirely (marginally faster).
        input_audio_transcription=realtime.AudioTranscription(
            model="gpt-4o-transcribe",
            language="en",
        ),

        # Noise reduction for input audio before it hits the model.
        # "near_field" = headset/close mic. Use "far_field" for speaker setups.
        input_audio_noise_reduction="near_field",

        # Turn detection: semantic_vad is OpenAI's smarter end-of-turn model.
        # It understands conversational context — far fewer false triggers than
        # raw silence detection.
        #   eagerness="high"       → agent fires quickly after user stops
        #   interrupt_response=True → user can barge in at any time
        #   create_response=True   → model auto-generates reply on turn end
        turn_detection=realtime.realtime_audio_input_turn_detection.SemanticVad(
            type="semantic_vad",
            eagerness="high",
            create_response=True,
            interrupt_response=True,
        ),
    )

    # ------------------------------------------------------------------
    # Session — pass the RealtimeModel as the llm parameter.
    # No separate stt= or tts= needed — the Realtime API handles all three.
    # Silero VAD is still used locally as a lightweight gating layer.
    # ------------------------------------------------------------------
    session = AgentSession(
        llm=realtime_model,
        vad=silero.VAD.load(
            min_silence_duration=0.2,
            activation_threshold=0.5,
            min_speech_duration=0.05,
        ),
    )

    # ------------------------------------------------------------------
    # Metrics logger (safe version — avoids 1.5.x pydantic crash)
    # ------------------------------------------------------------------
    @session.on("metrics_collected")
    def on_metrics(ev) -> None:
        try:
            logger.info(f"Metrics: {ev}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Start the session with BVC noise cancellation
    # ------------------------------------------------------------------
    await session.start(
        room=ctx.room,
        agent=VoiceAgent(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info("Session started — Realtime agent is live.")


# ---------------------------------------------------------------------------
# Worker entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )