import streamlit as st
import numpy as np
import soundfile as sf
from scipy.signal import firwin, lfilter
import io
import librosa
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="Digital Music Equalizer", layout="centered")

# --- Session state to switch pages ---
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Styles ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');
    .stApp {
        background: linear-gradient(135deg, #0a0a0a, #1a001a);
        color: white;
        font-family: 'Orbitron', sans-serif;
    }
    h1, h2, h3 {
        color: white;
        text-shadow: 0 0 15px #ff69b4;
    }
    .styled-button {
        background: linear-gradient(90deg, #ff5f6d, #845ec2);
        border: none;
        padding: 0.75em 2em;
        font-size: 1.2em;
        color: white;
        font-weight: bold;
        border-radius: 25px;
        cursor: pointer;
        box-shadow: 0 0 20px #ff69b4;
        transition: 0.3s ease;
    }
    .styled-button:hover {
        background: linear-gradient(90deg, #845ec2, #ff5f6d);
        color: black;
    }
    .center {
        text-align: center;
        margin-top: 10em;
    }
    .stDownloadButton button {
        background: linear-gradient(90deg, #ff5f6d, #845ec2);
        color: white;
        font-weight: bold;
        border-radius: 25px;
        border: none;
        box-shadow: 0 0 20px #ff69b4;
        padding: 0.75em 2em;
        font-size: 1.2em;
        transition: 0.3s ease;
    }
    .stDownloadButton button:hover {
        background: linear-gradient(90deg, #845ec2, #ff5f6d);
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# --- Functions ---
def load_audio(file):
    y, sr = librosa.load(file, sr=None, mono=True)
    return y, sr

def bandpass_filter(data, lowcut, highcut, fs, numtaps=101):
    taps = firwin(numtaps, [lowcut, highcut], pass_zero=False, fs=fs)
    return lfilter(taps, 1.0, data)

def apply_equalizer(data, fs, gains):
    bands = [(60, 250), (250, 4000), (4000, 10000)]  # Bass, Mid, Treble
    processed = np.zeros_like(data)
    for (low, high), gain in zip(bands, gains):
        filtered = bandpass_filter(data, low, high, fs)
        processed += filtered * gain
    return processed

# --- Home Page ---
if st.session_state.page == "home":
    st.markdown("""
        <div class="center">
            <h1>🎧 Digital Music Equalizer</h1>
            <p style='font-size: 1.2em;'>Shape your sound with studio-level precision.</p>
            <form action="">
                <button class="styled-button" type="submit" name="start" value="1">Start Now</button>
            </form>
        </div>
    """, unsafe_allow_html=True)

    if st.query_params.get("start") == "1":
        st.session_state.page = "equalizer"
        st.rerun()  # Force rerun to update page

# --- Equalizer Page (Introduction) ---
elif st.session_state.page == "equalizer":
    st.title("🎛️ Digital Music Equalizer")
    st.markdown("""
        <div style="text-align: center; font-size: 1.2em;">
            <p>Welcome to the Digital Music Equalizer! 🎶</p>
            <p>Upload your audio track and fine-tune the frequencies to shape your sound.</p>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.markdown('<button class="styled-button">🎚️ Go to Equalizer Controls</button>', unsafe_allow_html=True):
        st.session_state.page = "controls"
        st.rerun()  # Force rerun to update page

# --- Equalizer Controls Page ---
elif st.session_state.page == "controls":
    st.title("🎚️ Adjust Your Sound")
    uploaded_file = st.file_uploader("🎵 Upload your audio track (WAV or MP3)", type=["wav", "mp3"])

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 100:
            st.error("⚠️ File size exceeds 100 MB limit. Please upload a smaller file.")
        else:
            data, fs = load_audio(uploaded_file)
            st.audio(uploaded_file)

            st.subheader("🎚️ Adjust the Frequencies")
            bass = st.slider("Bass Boost (60–250 Hz)", 0.0, 2.0, 1.0, 0.1)
            mid = st.slider("Midrange Boost (250 Hz – 4 kHz)", 0.0, 2.0, 1.0, 0.1)
            treble = st.slider("Treble Boost (4–10 kHz)", 0.0, 2.0, 1.0, 0.1)

            output = apply_equalizer(data, fs, [bass, mid, treble])

            # Save and play
            buf = io.BytesIO()
            sf.write(buf, output, fs, format='WAV')
            st.audio(buf, format='audio/wav')

            # Styled Download Button
            st.markdown('<button class="styled-button">⬇️ Download Processed Audio</button>', unsafe_allow_html=True)

            # Visualization
            st.subheader("🔊 Processed Track Waveform")
            fig, ax = plt.subplots(figsize=(10, 4))
            time = np.linspace(0, len(output) / fs, num=len(output))
            ax.plot(time, output, color="#ff69b4", linewidth=0.5)
            ax.set_title("Processed Audio", fontsize=12, color='#ff69b4')
            ax.set_xlabel("Time [s]", color='white')
            ax.set_ylabel("Amplitude", color='white')
            ax.set_facecolor("#0a0a0a")
            ax.tick_params(colors='white')
            fig.patch.set_facecolor("#0a0a0a")
            st.pyplot(fig)
