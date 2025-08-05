# Chatterbox (default)
from audio.kitten_player import KittenAudioPlayer


# Kitten
player = KittenAudioPlayer(
    kitten_voice="expr-voice-4-f",
    kitten_speed=1.3,
)
player(
    "Geothermal energy taps into the Earth's internal heat to generate electricity or provide direct heating."
)
