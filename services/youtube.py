from __future__ import annotations
from pathlib import Path
import logging
from typing import Literal, Optional
import yt_dlp
import os

# Simple settings
class Settings:
    storage_uploads = Path("storage/uploads")

settings = Settings()

from .file_manager import sanitize_filename

logger = logging.getLogger(__name__)

def download_from_youtube(url: str, audio_only: bool = True) -> Path:
    """Download media from YouTube using yt-dlp Python API and return local file path."""
    out_dir = settings.storage_uploads
    out_dir.mkdir(parents=True, exist_ok=True)

    # Template: title + id to avoid collisions
    out_tmpl = str(out_dir / "%(title)s-%(id)s.%(ext)s")

    ydl_opts = {
        'outtmpl': out_tmpl,
        'quiet': True,
        'no_warnings': True,
        # Use simpler format selection to avoid complex extraction
        'format': 'bestaudio/best' if audio_only else 'best',
        # Add options to bypass YouTube restrictions
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'referer': 'https://www.youtube.com/',
        'extract_flat': False,
        'fragment_retries': 10,
        'retryfragments': 10,
        'no_cookies': True,
        'download_archive': None,
        'overwrites': True,
        'http_chunk_size': 10485760,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Iniciando download do YouTube: {url}")
            # extract_info com download=True baixa e retorna metadados
            info = ydl.extract_info(url, download=True)

            if info is None:
                raise RuntimeError("YouTube extraction returned None - video may be unavailable or restricted")

            # prepare_filename retorna o nome esperado do arquivo
            # Nota: se o yt-dlp fizer merge (video+audio) ou converter, o nome final pode mudar.
            # Mas como estamos pedindo bestaudio[ext=m4a], geralmente é direto.
            # Se for playlist, info é uma lista, mas aqui assumimos vídeo único ou pegamos o primeiro.

            if 'entries' in info:
                # É uma playlist ou resultado de busca
                info = info['entries'][0]

            filename = ydl.prepare_filename(info)
            path = Path(filename)

            if not path.exists():
                # Às vezes o yt-dlp muda a extensão se fizer pós-processamento não solicitado
                # Tentar encontrar arquivo com mesmo stem
                candidates = list(out_dir.glob(f"{path.stem}.*"))
                if candidates:
                    path = candidates[0]
                else:
                    raise RuntimeError(f"Downloaded file not found: {filename}")

            logger.info(f"Download concluído: {path}")

            # Sanitize the filename to remove accents and spaces
            safe_filename = sanitize_filename(path.name)
            safe_path = path.with_name(safe_filename)

            if safe_path != path:
                # If target exists, we might overwrite or fail.
                # Since we have ID in filename, collision implies same video.
                # Rename/move safely.
                path.replace(safe_path)
                logger.info(f"Arquivo renomeado para: {safe_path}")
                return safe_path

            return path
            
    except Exception as e:
        logger.exception(f"Erro ao baixar do YouTube: {e}")
        raise RuntimeError(f"Falha ao baixar do YouTube: {e}") from e
