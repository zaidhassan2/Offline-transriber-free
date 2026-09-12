# 🎙️ AI Video Transcriber

**Transform your videos into text with local AI - No cloud, No data leaks**

Developed by [Zaid Hassan](https://zaidhassan.me)

---

## ✨ Features

* 🎥 **Video Upload Support:** Upload `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` files
* 🚀 **Real-Time Progress:** Watch the transcription status live with progress indicators
* 🛡 **Privacy-First:** Entire pipeline runs locally - no data sent to cloud
* 🤖 **AI-Powered:** Uses OpenAI Whisper for accurate speech-to-text
* ⚡ **GPU Acceleration:** Automatic CUDA support for NVIDIA GPUs
* 📁 **Multiple Export Formats:** Download as `.txt`, `.srt`, `.docx`, `.pdf`
* 🕐 **Timestamped Segments:** Each segment tagged with `[HH:MM:SS]` format
* 🎨 **Premium UI:** Modern, responsive design with gradient styling

---

## 🛠 Tech Stack

* **Frontend:** Streamlit (Python web framework)
* **AI Engine:** OpenAI Whisper (local speech-to-text)
* **Audio Processing:** FFmpeg (video to audio extraction)
* **File Processing:** Python's built-in libraries
* **Deployment:** Streamlit Cloud (free tier available)

---

## 📂 Project Structure

```txt
transcriber_streamlit/
 ├─ app.py                  # Main Streamlit application
 ├─ requirements.txt        # Python dependencies
 ├─ .streamlit/
 │   └─ config.toml        # Streamlit configuration
 ├─ services/
 │   ├─ transcriber.py      # Whisper transcription engine
 │   ├─ file_manager.py    # File handling utilities
 │   └─ youtube.py         # YouTube download (optional)
 ├─ storage/
 │   ├─ uploads/           # Temporary media files
 │   └─ transcriptions/    # Generated transcripts
 └─ README.md              # This file
```

---

## ⚙️ Requirements

* **Python 3.10 to 3.12** (Recommended for GPU support)
* **FFmpeg**: Required for audio extraction from video files
* **NVIDIA GPU** (Optional): For faster transcription

> **⚠️ Important Note on Python 3.13+:** The official PyTorch binaries with CUDA support often lag behind the latest Python releases. If you are using Python 3.13 or 3.14, `pip install torch` might fallback to the CPU-only version. For the best experience with NVIDIA GPUs, please use Python 3.10, 3.11, or 3.12.

---

## 🚀 Quickstart

### Local Development

**Windows (PowerShell):**

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# Download from: https://ffmpeg.org/download.html

# Run the application
streamlit run app.py
```

**macOS/Linux:**

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (if not already installed)
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Run the application
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Streamlit Cloud Deployment

### Option 1: Deploy via Streamlit Cloud (Recommended)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/zaidhassan2/your-repo.git
   git push -u origin main
   ```

2. **Deploy to Streamlit:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Select `app.py` as the main file
   - Click "Deploy"

### Option 2: Deploy via CLI

```bash
# Install Streamlit CLI
pip install streamlit

# Login to Streamlit
streamlit login

# Deploy
streamlit run app.py
```

---

## 📖 Usage

1. **Upload:** Choose a video file (MP4, MOV, AVI, MKV, WEBM)
2. **Select Model:** Choose AI model size (tiny, base, small, medium)
3. **Transcribe:** Click "Start Transcription" button
4. **Wait:** Watch the progress as AI processes your video
5. **Download:** Get your transcript in multiple formats

**Key Features:**

* 📹 **Video Upload:** Drag & drop or select video files
* 🎯 **Model Selection:** Choose accuracy vs speed trade-off
* 📊 **Progress Tracking:** Real-time status updates
* 📝 **Transcript View:** Full text with editable text area
* 🕐 **Timestamped Segments:** Expandable view with `[HH:MM:SS]` format
* 💾 **Multiple Downloads:** TXT, SRT, DOCX, PDF formats
* 📋 **Copy Function:** One-click text copying

---

## 🔒 Privacy & Security

This application is designed for **local/offline execution**:

- ✅ **No Cloud Processing:** All AI processing happens on your machine
- ✅ **No Data Collection:** No video or audio data sent to external services
- ✅ **Offline Capability:** Works without internet after initial setup
- ✅ **Your Data Stays Private:** Complete control over your content

---

## 🧩 Troubleshooting

### FFmpeg Error
The app requires FFmpeg for audio extraction. Please install FFmpeg:
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

### Slow Transcription
- Try a smaller model (tiny or base)
- If you have an NVIDIA GPU, ensure CUDA PyTorch is installed
- Use Python 3.10-3.12 for better GPU support

### Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Streamlit Cloud Issues
- Check the [Streamlit Cloud documentation](https://docs.streamlit.io/)
- Ensure all dependencies are in requirements.txt
- Verify the main file is set to `app.py`

---

## 📜 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

---

## 🤝 Support

For questions or support, visit [zaidhassan.me](https://zaidhassan.me) or open an issue on GitHub.

---

**Keywords:** AI transcription, speech-to-text, video transcription, offline transcription, Whisper AI, voice recognition, local AI, privacy-first AI, Streamlit app, video to text, audio transcription, subtitle generation
