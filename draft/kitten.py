from kittentts import KittenTTS

m = KittenTTS("KittenML/kitten-tts-nano-0.1")

available_voices = [
    "expr-voice-2-m",  # 0
    "expr-voice-2-f",  # 1
    "expr-voice-3-m",  # 2
    "expr-voice-3-f",  # 3
    "expr-voice-4-m",  # 4 <-- best from testing
    "expr-voice-4-f",  # 5
    "expr-voice-5-m",  # 6
    "expr-voice-5-f",  # 7
]

speed = 1.3  # Best from testing

prompt = """Geothermal energy taps into the Earth's internal heat to generate electricity or provide direct heating."""

audio = m.generate(
    prompt,
    voice=available_voices[7],
    speed=1.3,
)
# Save the audio
import soundfile as sf

sf.write("output.wav", audio, 24000)
