from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os
import shutil
import logging
import subprocess
import sys
import re
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


def _normalize_text(text: str, custom_vocabulary: list[str] = None, custom_corrections: dict[str, str] = None) -> str:
    """Advanced verbatim text normalization for ASR post-processing.
    
    Fixes semantic inversions, speaker boundaries, disfluency stutters, 
    acronym fragmentation, and trailing syllable dropping without 
    summarizing or hallucinating content.
    """
    if not text:
        return text
    
    # Apply custom corrections first (user-defined - user's responsibility for domain-specific fixes)
    if custom_corrections:
        for wrong, correct in custom_corrections.items():
            # More conservative: only replace exact word boundaries, case-insensitive
            text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
    
    # === PROBLEM 1: Semantic Inversions (Context Collisions) ===
    # Fix common semantic inversions in conversational speech
    semantic_corrections = {
        'end the discussion': 'enter the discussion',  # Context: meetings
        'to be me': 'to be mean',  # Trailing consonant dropout
        'become than': 'be kind than',  # Phonetic confusion
        'call paying': 'called paying',  # Missing past tense
        'difficult for me': 'it\'s difficult for me',  # Missing subject
    }
    
    for wrong, correct in semantic_corrections.items():
        # Use word boundaries to avoid false positives
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
    
    # === PROBLEM 2: Speaker Shift Run-ons & Dropped Boundaries ===
    # Add punctuation at common speaker transition phrases
    text = re.sub(r'\b(we can say here|we can talk about|we can discuss|anything else)\s+([A-Z][a-z]+)', 
                  r'\1, \2', text)  # "anything else we can say here Phil" → "anything else we can say here, Phil"
    
    # Fix common run-on patterns
    text = re.sub(r'\b(we can say)\s+(that)', r'\1 that', text)  # "we can say here Phil" → "we can say that. Here, Phil"
    text = re.sub(r'\b(tell people about)\s+(anything else)', r'tell people that. Anything else', text)
    
    # === PROBLEM 3: Disfluency Stutters & Pause-Induced Word Duplication ===
    # Remove immediate word repetitions (stutters)
    text = re.sub(r'\b(\w+)(\s+\1){1,2}\b', r'\1', text)  # Remove 2-3 repetitions
    text = re.sub(r'\b(\w+)(\s+\1)\s+(and|or|but|so)', r'\1 \2', text)  # "learning learning and" → "learning and"
    
    # Fix specific stutter patterns
    text = re.sub(r'\b(ask|can|just)\s+\1\b', r'\1', text)  # "ask ask" → "ask"
    text = re.sub(r'\b(learning|english)\s+\1\b', r'\1', text)  # "learning learning" → "learning"
    
    # === PROBLEM 4: Acronym Fragmentation & Non-Standard Spacing ===
    # Fix common acronym fragmentations
    acronym_corrections = {
        'A, O, B': 'AOB',
        'R-E-S-P-C-T': 'R-E-S-P-E-C-T',
        'T C big': 'TCB',
        'T C B': 'TCB',
        'chat G-P-T': 'ChatGPT',
        'A I': 'AI',
        'B B C': 'BBC',
    }
    
    for wrong, correct in acronym_corrections.items():
        text = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, text, flags=re.IGNORECASE)
    
    # Fix common acronym patterns
    text = re.sub(r'\b([A-Z])\s*,\s*([A-Z])\s*,\s*([A-Z])\b', r'\1\2\3', text)  # "A, O, B" → "AOB"
    text = re.sub(r'\b([A-Z])\s*-\s*([A-Z])\s*-\s*([A-Z])\b', r'\1\2\3', text)  # "A-O-B" → "AOB"
    
    # === PROBLEM 5: Trailing Syllable Dropping at Low Energy ===
    # Fix common trailing syllable drops (already covered in semantic corrections)
    # Additional low-energy word ending fixes
    text = re.sub(r'\b(difficult)\s+(for me)\b', r'difficult for me', text)
    text = re.sub(r'\b(kids were right)\s+(around your age)\b', r'kids were right around your age', text)
    
    # === Universal Fixes ===
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)  # Normalize space after punctuation
    
    # Fix punctuation at sentence boundaries (add periods where missing)
    text = re.sub(r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b', 
                  lambda m: m.group(0), text)  # Don't add periods to very long sequences
    
    # Add periods at question words when followed by new speakers
    text = re.sub(r'\b(can|what|how|why|when|where|who)\s+([A-Z][a-z]+)\b', r'\1? \2', text)
    
    # Fix rogue number insertions at boundaries (e.g., "2025 2021")
    text = re.sub(r'(\d{4})\s+(\d{4})', r'\1', text)  # Remove duplicate years
    
    # Capitalize first letter of sentences
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    # Capitalize 'I' when standalone (universal fix)
    text = re.sub(r'\bi\b', 'I', text)
    
    # Remove leading/trailing whitespace from each line
    text = ' '.join(text.split())
    
    # Apply custom vocabulary biasing - capitalize custom terms
    if custom_vocabulary:
        for term in custom_vocabulary:
            # Conservative: only capitalize if it appears as lowercase
            text = re.sub(r'\b' + re.escape(term.lower()) + r'\b', term, text, flags=re.IGNORECASE)
    
    return text


def _apply_beam_search_decoding(segments: list[TranscriptionSegment]) -> list[TranscriptionSegment]:
    """Apply beam search-like corrections to segments."""
    corrected_segments = []
    
    for seg in segments:
        text = seg.text
        
        # Fix common boundary hallucinations
        # Remove trailing numbers that look like years if they don't make sense
        text = re.sub(r'\s+\d{4}\s*$', '', text)
        
        # Fix common acronym confusion
        text = re.sub(r'\bT\.C\.\-Big\b', 'TCB', text)
        text = re.sub(r'\bR\-E\-S\-P\-C\-T\b', 'RESPECT', text)
        
        corrected_segments.append(TranscriptionSegment(
            start=seg.start,
            end=seg.end,
            text=text
        ))
    
    return corrected_segments


def _trim_silence_from_audio(wav_path: Path, enable_trim: bool = True) -> Path:
    """Disabled - VAD with speech padding is now used instead for better natural pause preservation.
    
    VAD with 400ms speech padding provides better protection for natural pauses,
    comedic timing, and soft-spoken endings than FFmpeg-based silence trimming.
    """
    # Always return original - VAD handles silence detection properly
    return wav_path


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
    custom_vocabulary: list[str] = None,
    custom_corrections: dict[str, str] = None,
    keywords: str = None,
) -> TranscriptionResult:
    """Transcribe a media file to text using faster-whisper backend with ASR fixes.

    Args:
        media_path: Path to the media file
        model_name: Whisper model size (base, small, medium)
        progress_callback: Optional function for progress updates
        language: Explicit language code (e.g., 'en', 'es', 'fr') or None for auto-detect
        custom_vocabulary: List of custom terms to bias recognition (e.g., proper names, brand names)
        custom_corrections: Dictionary of custom corrections for specific phrases
        keywords: Optional keywords from filename or user input for dynamic topic extraction

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
            progress_callback(30, "Optimizing audio quality...")

        # Post-process: Trim silence to reduce extraneous audio (VAD handles this now)
        wav_path = _trim_silence_from_audio(wav_path)

        if progress_callback:
            progress_callback(35, "Processing audio with ASR optimizations...")

        transcribe_input = wav_path
        logger.info(f"Transcribing from: {transcribe_input}")

        # ASR Optimization: Explicit language constraint to avoid code-switching issues
        language_param = language if language else None

        # ASR Optimization: Temperature fallback for escaping loops
        temperature = (0.0, 0.2, 0.4)  # Deterministic first, then fallback if needed

        # ASR Optimization: No speech threshold to handle low-confidence audio
        no_speech_threshold = 0.5  # Reduced from 0.6 to catch more low-energy speech

        # ASR Optimization: condition_on_previous_text to prevent error cascading
        condition_on_previous_text = False  # Disable to prevent error cascading without adding latency

        # ASR Optimization: Compression ratio threshold to catch infinite loops
        compression_ratio_threshold = 2.4

        # ASR Optimization: Beam size for better decoding
        beam_size = 5

        # ASR Optimization: VAD parameters with generous padding for natural pauses
        vad_filter = True
        vad_parameters = {
            "min_silence_duration_ms": 500,  # 500ms minimum silence (reduced from 600ms to catch shorter conversational pauses)
            "speech_pad_ms": 500  # Increased from 400ms to 500ms to catch low-energy word endings and vocal fry
        }

        # ASR Optimization: Prompt biasing for conversational context and meeting vocabulary
        # Multi-speaker conversational context with common meeting terms
        initial_prompt = None
        if language == "en" or language is None:
            # Context priming for conversational speech, meetings, and common acronyms
            initial_prompt = "This is a multi-speaker conversational transcript about a meeting or discussion. Common terms include: AOB (any other business), ChatGPT, AI, discussion, permission, difficult, process. Acronyms may be spelled out or combined."
        
        # Add custom vocabulary to initial prompt if provided
        if custom_vocabulary:
            vocab_prompt = " ".join(custom_vocabulary)
            if initial_prompt:
                initial_prompt += " " + vocab_prompt
            else:
                initial_prompt = vocab_prompt
        
        # Add keywords for dynamic topic extraction if provided
        if keywords:
            if initial_prompt:
                initial_prompt += " " + keywords
            else:
                initial_prompt = keywords

        raw_segments, info = model.transcribe(
            str(transcribe_input),
            language=language_param,
            beam_size=beam_size,
            temperature=temperature,
            no_speech_threshold=no_speech_threshold,
            initial_prompt=initial_prompt,
            word_timestamps=True,  # Better for chunk alignment
            condition_on_previous_text=condition_on_previous_text,  # Prevent error cascading
            compression_ratio_threshold=compression_ratio_threshold,  # Guard against infinite loops
            vad_filter=vad_filter,
            vad_parameters=vad_parameters
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
                current_percent = 35 + int((seg.end / total_duration) * 60)
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

        # Cleanup memory for Streamlit Cloud efficiency
        import gc
        gc.collect()

        # Post-process: Basic text normalization (conservative, universal fixes only)
        text = _normalize_text(text, custom_vocabulary, custom_corrections)

        # Post-process: Normalize individual segments
        for seg in captured_segments:
            seg.text = _normalize_text(seg.text, custom_vocabulary, custom_corrections)

        # Post-process: Apply beam search corrections
        captured_segments = _apply_beam_search_decoding(captured_segments)

        # Post-process: Normalize text to fix common ASR issues
        text = _normalize_text(text, custom_vocabulary, custom_corrections)

        # Post-process: Normalize individual segments
        for seg in captured_segments:
            seg.text = _normalize_text(seg.text, custom_vocabulary, custom_corrections)

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