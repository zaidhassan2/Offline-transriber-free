from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import shutil
import logging
import subprocess
import sys
from typing import Callable, Optional

try:
    import torch  # type: ignore
except Exception:
    torch = None  # type: ignore

# Simple settings
class Settings:
    whisper_model_default = "base"
    transcription_device = "auto"

settings = Settings()

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A single timed segment from Whisper."""
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Structured result returned by transcribe_file()."""
    text: str
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: Optional[str] = None


def _check_ffmpeg_available() -> bool:
    """Check whether ffmpeg is available on this system."""
    if shutil.which("ffmpeg"):
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            logger.info("ffmpeg found in global PATH")
            return True
        except Exception:
            pass

    logger.warning("ffmpeg not found. Please install FFmpeg manually.")
    return False


def _probe_audio_stream(media_path: Path) -> bool:
    """Check if a media file contains at least one audio stream."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", str(media_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        has_audio = "Audio:" in combined
        if not has_audio:
            logger.warning(f"No audio stream found (ffmpeg -i): {media_path}")
        return has_audio
    except Exception as e:
        logger.debug(f"Audio probe failed: {e}")
        return True  # Assume audio present


def _is_audio_file(media_path: Path) -> bool:
    """Check if a file is already an audio format."""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    return media_path.suffix.lower() in audio_extensions


def _extract_audio_to_wav(media_path: Path) -> Path:
    """Extract audio from a media file to 16 kHz mono WAV."""
    # If it's already an audio file, just convert to WAV format
    if _is_audio_file(media_path):
        wav_path = media_path.with_suffix(".wav")
        if wav_path.exists():
            wav_path.unlink()
        
        logger.info(f"Converting audio file to WAV: {media_path} → {wav_path}")
        
        common_tail = [
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(wav_path),
        ]
        
        cmd = ["ffmpeg", "-i", str(media_path)] + common_tail
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                logger.info(f"Audio converted successfully: {wav_path}")
                return wav_path
            raise RuntimeError(f"Audio conversion failed: {result.stderr[-400:] if result.stderr else 'unknown error'}")
        except Exception as e:
            raise RuntimeError(f"Audio conversion failed: {str(e)}")
    
    # For video files, extract audio
    wav_path = media_path.with_suffix(".wav")

    if wav_path.exists():
        wav_path.unlink()

    logger.info(f"Extracting audio from {media_path} → {wav_path}")

    common_tail = [
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        "-y",
        str(wav_path),
    ]

    # Try different extraction strategies
    strategies = [
        (["ffmpeg", "-i", str(media_path), "-map", "0:a:0"] + common_tail, "Explicit audio map"),
        (["ffmpeg", "-i", str(media_path), "-vn"] + common_tail, "Strip video"),
    ]

    last_error = None
    for cmd, strategy_name in strategies:
        try:
            logger.info(f"Audio extraction strategy: {strategy_name}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0 and wav_path.exists() and wav_path.stat().st_size > 44:
                logger.info(f"Audio extracted successfully: {wav_path}")
                return wav_path

            err_msg = result.stderr[-400:] if result.stderr else "(no stderr)"
            logger.warning(f"Strategy '{strategy_name}' failed: {err_msg}")
            last_error = err_msg

            if wav_path.exists():
                wav_path.unlink()

        except subprocess.TimeoutExpired:
            logger.warning(f"Strategy '{strategy_name}' timed out")
            last_error = "timeout"
            if wav_path.exists():
                wav_path.unlink()
        except Exception as exc:
            logger.warning(f"Strategy '{strategy_name}' raised exception: {exc}")
            last_error = str(exc)
            if wav_path.exists():
                wav_path.unlink()

    raise RuntimeError(f"Audio extraction failed. Last error: {last_error}")



def _resolve_faster_whisper_model_path(model_name: str) -> str:
    """Resolve a faster-whisper model name to its local path."""
    if Path(model_name).exists():
        return model_name
    return model_name  # Let faster-whisper handle the resolution


def transcribe_file(
    media_path: Path,
    model_name: str | None = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    language: str | None = None,
) -> TranscriptionResult:
    """Transcribe a media file to text using faster-whisper backend with ASR fixes.

    Args:
        media_path: Path to the media file
        model_name: Whisper model size (base, small, medium)
        progress_callback: Optional function for progress updates
        language: Explicit language code (e.g., 'en', 'es', 'fr') or None for auto-detect

    Returns:
        TranscriptionResult with text, segments, and detected language
    """
    if model_name is None:
        model_name = settings.whisper_model_default

    if progress_callback:
        progress_callback(0, "Starting transcription...")

    ffmpeg_available = _check_ffmpeg_available()

    if not ffmpeg_available:
        raise RuntimeError("FFmpeg is required for audio extraction. Please install FFmpeg.")

    # Determine device
    force_device = settings.transcription_device
    if force_device == "cuda":
        gpu_available = bool(torch is not None and getattr(torch.cuda, "is_available", lambda: False)())
        device = "cuda" if gpu_available else "cpu"
    elif force_device == "cpu":
        device = "cpu"
    else:  # auto
        gpu_available = bool(torch is not None and getattr(torch.cuda, "is_available", lambda: False)())
        device = "cuda" if gpu_available else "cpu"

    try:
        from faster_whisper import WhisperModel
        logger.info("Using faster-whisper backend with ASR optimizations...")
        if progress_callback:
            progress_callback(10, "Loading AI model...")

        model = None
        fw_model_path = _resolve_faster_whisper_model_path(model_name)

        if gpu_available:
            try:
                model = WhisperModel(fw_model_path, device="cuda", compute_type="float16")
                logger.info("faster-whisper initialized with CUDA (compute_type=float16)")
            except Exception as cuda_init_err:
                logger.warning(
                    "faster-whisper CUDA unavailable/unsupported; falling back to CPU",
                    exc_info=cuda_init_err,
                )

        if model is None:
            model = WhisperModel(fw_model_path, device="cpu", compute_type="int8")
            logger.info("faster-whisper initialized with CPU (compute_type=int8)")

        if progress_callback:
            progress_callback(20, "Extracting audio from media...")

        wav_path = _extract_audio_to_wav(media_path)

        if progress_callback:
            progress_callback(40, "Processing audio with ASR optimizations...")

        transcribe_input = wav_path
        logger.info(f"Transcribing from: {transcribe_input}")

        # ASR Optimization: Explicit language constraint to avoid code-switching issues
        language_param = language if language else None

        # ASR Optimization: Repetition penalty to prevent hallucination loops
        repetition_penalty = 1.2

        # ASR Optimization: Temperature for better handling of uncertain segments
        temperature = 0.0

        # ASR Optimization: No speech threshold to handle low-confidence audio
        no_speech_threshold = 0.6

        # ASR Optimization: Prompt biasing for common English terms if language is English
        initial_prompt = None
        if language == "en" or language is None:
            # Common words to bias towards for better accuracy
            initial_prompt = "This is a transcription of spoken English language content."

        raw_segments, info = model.transcribe(
            str(transcribe_input),
            language=language_param,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            initial_prompt=initial_prompt,
            word_timestamps=True  # Better for chunk alignment
        )

        total_duration = info.duration
        text_parts: list[str] = []
        captured_segments: list[TranscriptionSegment] = []

        for seg in raw_segments:
            seg_text = seg.text.strip()
            
            # ASR Fix: Filter out very short segments that might be hallucinations
            if len(seg_text) < 2:
                continue
                
            # ASR Fix: Skip segments that are just repeating punctuation
            if seg_text in [".", ",", "!", "?", "...", "...."]:
                continue
                
            text_parts.append(seg_text)
            captured_segments.append(
                TranscriptionSegment(start=float(seg.start), end=float(seg.end), text=seg_text)
            )
            
            if progress_callback and total_duration > 0:
                current_percent = 40 + int((seg.end / total_duration) * 55)
                current_percent = min(95, current_percent)
                progress_callback(
                    current_percent,
                    f"Transcribing: {int(seg.end)}s / {int(total_duration)}s",
                )

        text = " ".join(t for t in text_parts if t).strip()

        if progress_callback:
            progress_callback(100, "Transcription complete!")

        if wav_path.exists():
            wav_path.unlink()

        detected_language = getattr(info, "language", language)
        logger.info(
            f"backend=faster-whisper "
            f"device={'cuda' if (gpu_available and getattr(model, 'device', 'cpu') == 'cuda') else 'cpu'} "
            f"compute_type={'float16' if gpu_available else 'int8'} "
            f"model={model_name} "
            f"language={detected_language}"
        )
        return TranscriptionResult(
            text=text,
            segments=captured_segments,
            language=detected_language,
        )

    except ImportError:
        raise RuntimeError("faster-whisper not installed. Please install it with: pip install faster-whisper")
    except Exception as e:
        logger.exception("Transcription failed")
        raise RuntimeError(f"Transcription failed: {str(e)}")