import IPython
import torch
import matplotlib.pyplot as plt
from model import TransformerTTS
from melspecs import inverse_mel_spec_to_wav
from write_mp3 import write_mp3
from hyperparams import hp
from dataset import TextMelDataset, text_mel_collate_fn
from tts_loss import TTSLoss
from model import TransformerTTS
from melspecs import inverse_mel_spec_to_wav
from text_to_seq import text_to_seq
from tqdm import tqdm

train_saved_path = "param/train_SimpleTransfromerTTS.pt"

state = torch.load(train_saved_path)
model = TransformerTTS().cuda()
model.load_state_dict(state["model"])

text = "london is a capital of england"
name_file = "hello_world.mp3"


postnet_mel, gate = model.inference(
  text_to_seq(text).unsqueeze(0).cuda(),
  stop_token_threshold=1e-5,
  with_tqdm = False
)

audio = inverse_mel_spec_to_wav(postnet_mel.detach()[0].T)

plt.plot(
    torch.sigmoid(gate[0, :]).detach().cpu().numpy()
)

write_mp3(
    audio.detach().cpu().numpy(),
    name_file
)

IPython.display.Audio(
    audio.detach().cpu().numpy(),
    rate=hp.sr
)