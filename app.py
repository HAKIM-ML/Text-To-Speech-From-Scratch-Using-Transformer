import streamlit as st
import torch
import matplotlib.pyplot as plt
from model import TransformerTTS
from melspecs import inverse_mel_spec_to_wav
from write_mp3 import write_mp3
from hyperparams import hp
from text_to_seq import text_to_seq
import os

# Load the model
@st.cache_resource
def load_model():
    train_saved_path = "param/train_SimpleTransfromerTTS.pt"
    state = torch.load(train_saved_path)
    model = TransformerTTS().cuda()
    model.load_state_dict(state["model"])
    return model

model = load_model()

# Streamlit app
st.title("Text to Speech Converter")

# Text input
text = st.text_input("Enter the text you want to convert to speech:")

if st.button("Convert to Speech"):
    if text:
        # Generate speech
        postnet_mel, gate = model.inference(
            text_to_seq(text).unsqueeze(0).cuda(),
            stop_token_threshold=1e-5,
            with_tqdm=False
        )

        audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T)

        # Plot the gate output
        fig, ax = plt.subplots()
        ax.plot(torch.sigmoid(gate[0, :]).detach().cpu().numpy())
        ax.set_title("Gate Output")
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        st.pyplot(fig)

        # Save as MP3
        output_file = "output.mp3"
        write_mp3(
            audio.detach().cpu().numpy(),
            output_file
        )

        # Play the audio
        st.audio(output_file, format="audio/mp3")

        # Provide download link
        with open(output_file, "rb") as file:
            st.download_button(
                label="Download MP3",
                data=file,
                file_name=output_file,
                mime="audio/mp3"
            )

        # Clean up
        os.remove(output_file)
    else:
        st.warning("Please enter some text to convert.")