import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import sys

# Page configuration
st.set_page_config(
    page_title="AI Transcriber",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 1.5rem;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    .upload-section {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2.5rem;
        border-radius: 1.5rem;
        border: 2px solid #e9ecef;
        margin: 1.5rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    .upload-section h2 {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 1.5rem;
    }
    .success-box {
        background: linear-gradient(145deg, #d4edda 0%, #c3e6cb 100%);
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1.5rem 0;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(21, 87, 36, 0.15);
    }
    .transcript-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 1rem;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        max-height: 500px;
        overflow-y: auto;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    .timestamp {
        color: #667eea;
        font-weight: 600;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
    }
    .info-badge {
        background: linear-gradient(145deg, #e7f3ff 0%, #d0e8ff 100%);
        color: #0056b3;
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
        font-weight: 500;
        display: inline-block;
        margin: 0.25rem;
        border: 1px solid #b8d4fe;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 0.75rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 0.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    .section-title {
        color: #667eea;
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 1px solid #dee2e6;
    }
    footer a {
        color: #667eea;
        text-decoration: none;
        font-weight: 500;
    }
    footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'selected_language' not in st.session_state:
    st.session_state.selected_language = None

# Header
st.markdown("""
<div class="main-header">
    <h1>AI Video Transcriber</h1>
    <p>Transform your videos and audio into text with local AI - No cloud, No data leaks</p>
</div>
""", unsafe_allow_html=True)

# Import transcription services
sys.path.append(str(Path(__file__).parent))

try:
    from services.transcriber import transcribe_file, TranscriptionResult
except ImportError as e:
    st.error(f"❌ Failed to import transcription services: {e}")
    st.stop()

# Initialize uploaded_file globally to prevent NameError
uploaded_file = None

# Helper functions for export formats
def create_srt_content(segments):
    """Create SRT subtitle format content."""
    srt_lines = []
    for i, seg in enumerate(segments):
        timestamp_start = f"{int(seg.start // 3600):02d}:{int((seg.start % 3600) // 60):02d}:{int(seg.start % 60):02d},{int((seg.start % 1) * 1000):03d}"
        timestamp_end = f"{int(seg.end // 3600):02d}:{int((seg.end % 3600) // 60):02d}:{int(seg.end % 60):02d},{int((seg.end % 1) * 1000):03d}"
        srt_lines.append(f"{i+1}")
        srt_lines.append(f"{timestamp_start} --> {timestamp_end}")
        srt_lines.append(seg.text)
        srt_lines.append("")
    return "\n".join(srt_lines)

def create_docx_content(text):
    """Create DOCX content (simple text-based approach)."""
    return f"TRANSCRIPT\n{'='*50}\n\n{text}\n\nGenerated by AI Transcriber\nDeveloped by Zaid Hassan\n\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def create_pdf_content(text):
    """Create PDF content (basic text-based approach)."""
    return f"TRANSCRIPT\n{'='*50}\n\n{text}\n\nGenerated by AI Transcriber\nDeveloped by Zaid Hassan\n\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

# Sidebar with information
with st.sidebar:
    st.markdown("### Instructions")
    st.markdown("""
    1. **Upload** a video or audio file
    2. **Select** language (optional)
    3. **Click** "Start Transcription"
    4. **Wait** for AI processing
    5. **Download** your transcript in multiple formats
    """)

    st.markdown("---")
    st.markdown("### Privacy First")
    st.markdown("""
    - Local processing only
    - No data sent to cloud
    - Works offline after setup
    - Your data stays private
    """)

    st.markdown("---")
    st.markdown("### Settings")
    model_size = st.selectbox(
        "AI Model Size",
        ["tiny", "base", "small", "medium"],
        index=1,
        help="Larger models are more accurate but slower"
    )

    language = st.selectbox(
        "Language",
        ["Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian", "Japanese", "Korean", "Chinese", "Hindi", "Arabic"],
        index=0,
        help="Select language for better accuracy. Auto-detect works best for mixed content.",
        key="language_selector"
    )

    st.markdown("---")
    st.markdown("### Advanced Settings")
    
    # Custom vocabulary input
    custom_vocab = st.text_area(
        "Custom Vocabulary (Optional)",
        placeholder="Enter proper names, brand names, or technical terms separated by commas (e.g., Dillo Day, BigXthaPlug, Jedediah Ashcraft, Don DePollo)",
        help="Add custom terms to improve recognition of proper names and specialized vocabulary. Default terms include common meeting entities.",
        height=80,
        value="BigXthaPlug, Dillo Day, Jedediah Ashcraft, Don DePollo, Nobel Prizes"  # Default common proper nouns
    )
    
    # Parse custom vocabulary
    custom_vocabulary_list = None
    if custom_vocab.strip():
        custom_vocabulary_list = [term.strip() for term in custom_vocab.split(',') if term.strip()]
    
    # Custom corrections input
    custom_corrections_input = st.text_area(
        "Custom Corrections (Optional)",
        placeholder="Enter corrections in format: wrong_term=correct_term (one per line)\nExample:\nterm oil=turmoil\nfather seg=Father's Day",
        help="Add custom corrections for common misheard phrases. Default corrections include common meeting and speech errors.",
        height=80,
        value="BBC see learning English=BBC Learning English\npodcast that BBC Learning English=podcasts at BBC Learning English\nthink it be useful=think it can be useful\nagenda feel=agenda, Phil\nthink about that Phil=think about that, Phil\nhappy father's sake=Happy Father's Day"  # Default common corrections
    )
    
    # Parse custom corrections
    custom_corrections_dict = {}
    if custom_corrections_input and custom_corrections_input.strip():
        for line in custom_corrections_input.strip().split('\n'):
            if '=' in line:
                wrong, correct = line.split('=', 1)
                custom_corrections_dict[wrong.strip()] = correct.strip()
    
    # Keywords / Speaker Names (for dynamic topic extraction)
    keywords_input = st.text_input(
        "Keywords / Speaker Names (Optional)",
        placeholder="Enter speaker names, meeting topics, or technical terms (e.g., Steve Carell, Q4 Review, Budget)",
        help="Add keywords to improve recognition of proper names and meeting-specific terminology."
    )

    st.markdown("---")
    st.markdown("### Developer")
    st.markdown("Developed by [Zaid Hassan](https://zaidhassan.me)")

# Main content area
st.markdown("## Upload Media File")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a video or audio file",
    type=['mp4', 'mov', 'avi', 'mkv', 'webm', 'mp3', 'wav', 'm4a', 'flac', 'ogg'],
    help="Supported video formats: MP4, MOV, AVI, MKV, WEBM. Supported audio formats: MP3, WAV, M4A, FLAC, OGG"
)

# Extract keywords from filename for automatic topic extraction (after file upload)
filename_keywords = None
if uploaded_file:
    filename = uploaded_file.name
    # Extract first 100 characters from filename as context
    filename_keywords = filename[:100].replace('_', ' ').replace('-', ' ')

# Combine user keywords with filename keywords
combined_keywords = None
if uploaded_file or keywords_input:
    all_keywords = []
    if keywords_input and keywords_input.strip():
        all_keywords.append(keywords_input.strip())
    if filename_keywords:
        all_keywords.append(filename_keywords)
    combined_keywords = " ".join(all_keywords) if all_keywords else None

if uploaded_file:
    st.markdown(f"""
    <div class="info-badge">
        File: {uploaded_file.name}
    </div>
    <div class="info-badge">
        Size: {uploaded_file.size / (1024*1024):.2f} MB
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        start_button = st.button("Start Transcription", type="primary", disabled=st.session_state.processing)
    with col2:
        clear_button = st.button("Clear", disabled=st.session_state.processing)

    if clear_button:
        st.session_state.current_file = None
        st.session_state.transcription_result = None
        st.rerun()

    if start_button and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.current_file = uploaded_file
        # Get language from sidebar at the time of button click
        language_map = {
            "Auto-detect": None,
            "English": "en",
            "Spanish": "es", 
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Dutch": "nl",
            "Russian": "ru",
            "Japanese": "ja",
            "Korean": "ko",
            "Chinese": "zh",
            "Hindi": "hi",
            "Arabic": "ar"
        }
        selected_language = language_map[language]
        st.session_state.selected_language = selected_language

        # Create temporary file with proper cleanup
        tmp_path = None
        try:
            # Step 1: Stream directly to disk to prevent RAM bloat
            with st.spinner("Uploading file to disk..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = Path(tmp_file.name)
                    file_size_mb = tmp_path.stat().st_size / (1024 * 1024)
                    st.info(f"File uploaded: {file_size_mb:.2f} MB")
                    
                    # Warning for very large files
                    if file_size_mb > 200:
                        st.warning("⚠️ Large file detected (>200MB). Processing may take 10-20 minutes and could time out on free tier.")
                    elif file_size_mb > 100:
                        st.info("ℹ️ Large file detected (>100MB). Processing may take 5-10 minutes.")

            # Progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Initializing AI model...")
            progress_bar.progress(10)

            # Transcribe
            def progress_callback(percent, message):
                progress_bar.progress(percent)
                status_text.text(message)

            status_text.text("Extracting audio from media...")
            progress_bar.progress(20)

            try:
                with st.spinner("Processing media... This may take several minutes for 40+ min files."):
                    result = transcribe_file(
                        tmp_path, 
                        model_size, 
                        progress_callback, 
                        st.session_state.selected_language,
                        custom_vocabulary_list,
                        custom_corrections_dict,
                        combined_keywords
                    )
            except Exception as transcribe_error:
                st.error(f"Transcription failed: {str(transcribe_error)}")
                st.error(f"Error type: {type(transcribe_error).__name__}")
                raise

            status_text.text("Transcription complete!")
            progress_bar.progress(100)

            # Store result
            st.session_state.transcription_result = result
            st.session_state.processing = False

            st.rerun()

        except Exception as e:
            st.session_state.processing = False
            st.error(f"Transcription failed: {str(e)}")
        
        finally:
            # Immediate storage cleanup to release container tmpfs memory
            # Guaranteed cleanup regardless of success or failure
            if tmp_path and tmp_path.exists():
                tmp_path.unlink()
            import gc
            gc.collect()

elif st.session_state.processing:
    st.info("Processing in progress...")

# Display transcription results
if st.session_state.transcription_result:
    result = st.session_state.transcription_result

    st.markdown("---")
    st.markdown("### Transcription Complete")

    # Success message
    st.markdown(f"""
    <div class="success-box">
        <strong>Success!</strong> Your media has been transcribed with {len(result.segments)} segments.
    </div>
    """, unsafe_allow_html=True)

    # Transcript display
    st.markdown("### Transcript")

    # Transcript viewer - ensure text is properly displayed
    if result.text:
        st.text_area(
            "Full Transcript",
            result.text,
            height=300,
            key="transcript_text",
            help="Click on the transcript text to select and copy, or use the copy button below"
        )
    else:
        st.warning("No transcript text available")

    # Timestamped segments
    if result.segments:
        with st.expander("View Timestamped Segments", expanded=True):
            for segment in result.segments:
                timestamp = f"[{int(segment.start // 3600):02d}:{int((segment.start % 3600) // 60):02d}:{int(segment.start % 60):02d}]"
                st.markdown(f"""
                <div class="transcript-container">
                    <span class="timestamp">{timestamp}</span> {segment.text}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.warning("No timestamped segments available")

    # Action buttons
    st.markdown("---")
    st.markdown("### Download Transcript")

    col1, col2, col3 = st.columns(3)

    with col1:
        # Use Streamlit's built-in copy functionality
        st.code(result.text, language=None)

    with col2:
        # TXT download
        txt_content = result.text
        st.download_button(
            label="Download TXT",
            data=txt_content,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    with col3:
        # SRT download
        srt_content = create_srt_content(result.segments)
        st.download_button(
            label="Download SRT",
            data=srt_content,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.srt",
            mime="text/plain"
        )

    # Additional downloads
    st.markdown("### Additional Formats")
    col4, col5 = st.columns(2)

    with col4:
        # Simple text-based DOCX
        docx_content = create_docx_content(result.text)
        st.download_button(
            label="Download DOCX",
            data=docx_content,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
            mime="text/plain"
        )

    with col5:
        # Simple text-based PDF
        pdf_content = create_pdf_content(result.text)
        st.download_button(
            label="Download PDF (Text)",
            data=pdf_content,
            file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="text/plain"
        )

    # Reset button
    st.markdown("---")
    if st.button("Process Another File"):
        st.session_state.transcription_result = None
        st.session_state.current_file = None
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<footer>
    <p>AI Transcriber | Local AI-Powered Speech to Text</p>
    <p>Developed by <a href="https://zaidhassan.me" target="_blank">Zaid Hassan</a></p>
</footer>
""", unsafe_allow_html=True)
