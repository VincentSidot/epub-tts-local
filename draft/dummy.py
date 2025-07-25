import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

model = ChatterboxTTS.from_pretrained(device=device)

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."

data = []
for chunk in model.generate(text):
    print(f"Processing chunk of shape: {chunk.shape}")
    data.append(chunk)

wav = torch.cat(data, dim=-1)
print(f"Wavform shape: {wav.shape}")
ta.save("test-1.wav", wav, model.sr)
