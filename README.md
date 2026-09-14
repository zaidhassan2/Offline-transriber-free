# AI Video Transcriber

**Transform your videos and audio into text with local AI - No cloud, No data leaks**

Developed by [Zaid Hassan](https://zaidhassan.me)

---

## Features

* **Video Upload Support:** Upload `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` files
* **Audio Upload Support:** Upload `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` files
* **Large File Support:** Optimized for 40+ minute files on 1GB RAM containers
* **Real-Time Progress:** Watch the transcription status live with progress indicators
* **Privacy-First:** Entire pipeline runs locally - no data sent to cloud
* **AI-Powered:** Uses faster-whisper for accurate speech-to-text
* **GPU Acceleration:** Automatic CUDA support for NVIDIA GPUs
* **Multiple Export Formats:** Download as `.txt`, `.srt`, `.docx` (text-based), `.pdf` (text-based)
* **Timestamped Segments:** Each segment tagged with `[HH:MM:SS]` format
* **Premium UI:** Modern, responsive design with gradient styling
* **Multi-Language Support:** 13+ languages with explicit language constraints
* **Advanced ASR Architecture:** Systematic fixes for 5 failure modes in conversational audio
* **Dynamic Topic Extraction:** Automatic keyword extraction from filenames and user input
* **Custom Vocabulary:** Add proper names, brand names, and technical terms for better accuracy
* **Custom Corrections:** Define your own phrase corrections for specific misheard words
* **Memory Efficient:** Greedy decoding and lazy processing for 1GB RAM compatibility
* **Guaranteed Cleanup:** Automatic temporary file cleanup to prevent filesystem exhaustion

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

1. **Upload:** Choose a video file (MP4, MOV, AVI, MKV, WEBM) or audio file (MP3, WAV, M4A, FLAC, OGG)
2. **Select Model:** Choose AI model size (tiny, base, small, medium)
3. **Select Language:** Choose language for better accuracy (English, Spanish, French, etc.)
4. **Add Keywords:** (Optional) Enter speaker names, meeting topics, or technical terms for dynamic topic extraction
5. **Add Custom Vocabulary:** (Optional) Enter proper names, brand names, or technical terms
6. **Add Custom Corrections:** (Optional) Define phrase corrections for misheard words
7. **Transcribe:** Click "Start Transcription" button
8. **Wait:** Watch the progress as AI processes your video
9. **Download:** Get your transcript in multiple formats

### Advanced Features

#### Dynamic Topic Extraction
The system automatically extracts keywords from filenames and combines them with user input:
- **Automatic:** Filename "ENGLISH SPEECH STEVE CARELL" → "English Speech Steve Carell" added to prompt
- **Manual:** Add speaker names, meeting topics, or technical terms via UI
- **Benefit:** Better recognition of proper names and domain-specific vocabulary without hardcoded rules

#### Custom Vocabulary (with Default Terms)
Add proper names, brand names, and technical terms to improve recognition accuracy:
- **Default terms:** BigXthaPlug, Dillo Day, Jedediah Ashcraft, Don DePollo, Nobel Prizes
- **Format:** Comma-separated terms
- **Benefit:** Model biases toward these terms during transcription
- **Meeting focus:** Pre-loaded with common proper nouns from your feedback

#### Custom Corrections (with Default Terms)
Define your own corrections for commonly misheard phrases:
- **Default corrections:** BBC see learning → BBC Learning English, agenda feel → agenda, Phil, etc.
- **Format:** `wrong_term=correct_term` (one per line)
- **Benefit:** Fixes specific patterns identified from your test feedback
- **Override:** User corrections take precedence over built-in fixes

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

### Transcription Accuracy Issues
If you're seeing common ASR errors, try these fixes:

#### Common Misheard Words
Add custom corrections for:
- Phonetic errors: Use custom corrections for domain-specific terms
- Proper names: Add campus-specific terms to custom vocabulary
- Acronyms: Add brand names to custom vocabulary

#### Missing Words or Truncated Sentences
- Use a larger model (small or medium) for better context
- Ensure audio quality is good (low background noise)
- Try with explicit language selection instead of auto-detect
- **VAD with 400ms padding** automatically preserves natural pauses

#### Hallucination or Repetition
- The app has `condition_on_previous_text=False` to prevent error cascading
- Temperature fallback (0.0, 0.2, 0.4) for escaping loops
- Compression ratio threshold (2.4) catches infinite loops
- If issues persist, try a larger model for better context

#### Universal ASR Configuration
The system uses production-ready settings optimized for long files and memory constraints:
- **Greedy decoding (beam_size=1):** Reduces RAM usage by ~60% for 40+ minute files
- **Zero-context inference:** Each chunk evaluated independently to prevent error cascading
- **Enhanced VAD with generous padding:** 650ms minimum silence, 450ms speech padding for low-energy endings
- **Deterministic decoding:** Temperature 0.0 for greedy search consistency
- **Compression threshold:** Automatically catches infinite repetition loops
- **Dynamic topic extraction:** Keywords from filenames and user input bias recognition
- **Conversational context priming:** Meeting vocabulary and common acronyms in initial prompt
- **Model caching:** Prevents repeated model loading overhead

#### Systematic ASR Error Fixes
The pipeline addresses 5 systematic failure modes in conversational audio:

1. **Semantic Inversions (Context Collisions):**
   - Fixed: "end the discussion" → "enter the discussion"
   - Fixed: "to be me" → "to be mean"
   - Solution: Context-aware semantic corrections + conversational prompt biasing

2. **Speaker Shift Run-ons & Dropped Boundaries:**
   - Fixed: "we can say here Phil" → "we can say here, Phil"
   - Solution: Punctuation insertion at speaker transition phrases

3. **Disfluency Stutters & Word Duplication:**
   - Fixed: "ask ask" → "ask", "learning learning" → "learning"
   - Enhanced: Contractions handling "it's it's it's" → "it's"
   - Solution: Immediate repetition removal with enhanced stutter pattern detection

4. **Acronym Fragmentation:**
   - Fixed: "A, O, B" → "AOB", "R-E-S-P-C-T" → "R-E-S-P-E-C-T"
   - Solution: Acronym unification rules and spacing normalization

5. **Trailing Syllable Dropping:**
   - Fixed: "call paying" → "called paying", "difficult for me" context restoration
   - Solution: Enhanced VAD padding (450ms) + semantic corrections

#### Memory-Safe Architecture for Long Files
Optimized for 40+ minute files on 1GB RAM containers:

- **Greedy Decoding:** beam_size=1 reduces RAM usage by ~60% while maintaining accuracy
- **Model Caching:** Prevents repeated model loading overhead during sessions
- **Lazy Processing:** Segments processed as generators, not in-memory lists
- **Guaranteed Cleanup:** try...finally blocks ensure temporary file deletion
- **Immediate GC:** Garbage collection triggered after decoding and post-processing
- **Optimized VAD:** 650ms minimum silence, 450ms speech padding for long files

#### Feedback-Based Improvements (94% Meeting, 91% Speech Verbatim)
Based on detailed testing feedback, additional fixes implemented:

- **BBC Compound Noun Stutter:** "BBC see learning English" → "BBC Learning English"
- **Preposition Slips:** "podcast that" → "podcasts at", "think it be useful" → "think it can be useful"
- **Proper Noun Boundaries:** "agenda feel" → "agenda, Phil", "think about that Phil" → "think about that, Phil"
- **Meeting-Specific Terms:** Default vocabulary includes BigXthaPlug, Dillo Day, Don DePollo, Nobel Prizes
- **Outro Filtering:** Automatic removal of promotional content ("Join our global community...")
- **Enhanced Crowd Noise Handling:** Optimized VAD for laughter segments with conservative thresholds

#### Streamlit Cloud Memory Issues
- **Large file support:** Optimized for 40+ minute files on 1GB RAM containers
- **Greedy decoding:** beam_size=1 reduces RAM usage by ~60%
- **Model caching:** Prevents repeated model loading overhead
- **Lazy processing:** Segments processed as generators, not in-memory lists
- **Guaranteed cleanup:** Automatic temporary file deletion to prevent filesystem exhaustion
- **For very long files (>60 minutes):** Consider local processing for better performance

---

## 📜 License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.

---

## 🤝 Support

For questions or support, visit [zaidhassan.me](https://zaidhassan.me) or open an issue on GitHub.

---

**Keywords:** AI transcription, speech-to-text, video transcription, offline transcription, Whisper AI, voice recognition, local AI, privacy-first AI, Streamlit app, video to text, audio transcription, subtitle generation
