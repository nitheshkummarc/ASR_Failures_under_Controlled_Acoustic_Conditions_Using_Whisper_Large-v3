"""
Whisper inference wrapper.

Wraps faster-whisper (CTranslate2 backend) with the VRAM-failure resilience
needed to run large-v3 on a 4 GB RTX 3050.

IMPORTANT: `config` must be imported before `faster_whisper`, because config.py
injects the pip-installed nvidia/cublas and nvidia/cudnn bin directories into
PATH. Without that, CTranslate2 cannot load cublas64_12.dll / cudnn64_9.dll and
CUDA initialisation fails. The import order below is load-bearing.
"""

import gc

from config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    WHISPER_LANGUAGE,
)

from faster_whisper import WhisperModel

from src.logger import setup_logger

logger = setup_logger(__name__)

# Substrings that identify a genuine out-of-memory condition.
#
# This deliberately does NOT include the bare word "cuda". Matching "cuda" swept
# up every unrelated CUDA failure ("CUDA driver version is insufficient",
# "CUDA device not found", missing DLLs) and silently demoted the run to CPU or
# — worse — turned it into an empty transcript. Those failures must surface.
_OOM_MARKERS = ("out of memory", "oom", "cublas_status_alloc_failed")


def _is_oom(exc: Exception) -> bool:
    """True only for errors that genuinely indicate exhausted device memory."""
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MARKERS)


class TranscriptionOOMError(RuntimeError):
    """
    Raised when a single transcription runs out of device memory.

    This exists so callers can SKIP the affected row rather than record a result
    for it. The previous behaviour — returning "" — produced a row with
    deletions == len(reference) and wer == 1.0, which is indistinguishable from a
    genuine total recognition failure. Because OOM is most likely at the highest
    degradation levels (beam expansion on 5 dB cafe noise), those fabricated
    100%-WER rows landed precisely in the experimental cells the headline result
    depends on.
    """


class WhisperEngine:
    """Loads the model once and transcribes WAV files by path."""

    def __init__(self):
        self.model = self._load_whisper_model()

    def _load_whisper_model(self):
        """
        Initialise faster-whisper with an OOM fallback chain:
          1. config's device / compute_type  (currently cuda / int8)
          2. CUDA / int8                     (skipped when identical to step 1)
          3. CPU  / int8                     (last resort — correct but very slow)

        Any non-OOM failure is re-raised: a missing driver or a missing DLL is a
        setup error that should stop the run, not silently cost days of CPU
        inference.
        """
        logger.info(f"Loading faster-whisper model: {WHISPER_MODEL}")
        logger.info(f"Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE}")

        try:
            model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            logger.info("Model loaded successfully on primary configuration")
            return model

        except Exception as e:
            if not _is_oom(e):
                raise

            logger.warning(f"Primary configuration ran out of memory: {e}")

            # Only worth retrying if the primary configuration was not already
            # cuda/int8 — otherwise this is an identical call that will fail
            # the same way.
            if not (WHISPER_DEVICE == "cuda" and WHISPER_COMPUTE_TYPE == "int8"):
                logger.info("Retrying with device='cuda', compute_type='int8'...")
                try:
                    model = WhisperModel(
                        WHISPER_MODEL, device="cuda", compute_type="int8"
                    )
                    logger.info("Model loaded on CUDA/int8")
                    return model
                except Exception as e2:
                    if not _is_oom(e2):
                        raise
                    logger.warning(f"CUDA/int8 also ran out of memory: {e2}")

            logger.warning(
                "Falling back to CPU/int8. Inference will be MUCH slower — "
                "expect days rather than hours for a full pipeline run."
            )
            model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
            logger.info("Model loaded on CPU/int8")
            return model

    def transcribe(self, wav_path: str) -> str:
        """
        Transcribe a single WAV file.

        Inference parameters are fixed experimental controls:
          beam_size=5                    — matches the published Whisper setup
          language="en"                  — removes language-ID as a confound
          condition_on_previous_text=False
              Deliberate. With it enabled, a hallucination in one segment is fed
              into the next segment's prompt and cascades. Disabling it isolates
              per-segment failure, which is what a controlled degradation study
              needs to measure.

        Returns:
            The transcript, stripped.

        Raises:
            TranscriptionOOMError: device memory was exhausted. The caller must
                skip this row — never record a result for it.
            Exception: any other failure propagates unchanged.
        """
        try:
            # `segments` is a generator; decoding actually happens during the
            # join below, which is why it sits inside the try.
            segments, _info = self.model.transcribe(
                wav_path,
                beam_size=WHISPER_BEAM_SIZE,
                language=WHISPER_LANGUAGE,
                condition_on_previous_text=False,
            )
            predicted = " ".join(seg.text for seg in segments)
            return predicted.strip()

        except Exception as e:
            if not _is_oom(e):
                raise

            logger.warning(f"CUDA OOM during transcription of {wav_path}. Flushing cache.")
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                # torch is not a declared dependency; ctranslate2 manages its own
                # allocator. gc.collect() is still worth doing.
                pass
            gc.collect()

            raise TranscriptionOOMError(f"Out of memory transcribing {wav_path}") from e