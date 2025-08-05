import os
import argparse
import logging
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Generator
import torch


def epub_text_generator(file_path: str) -> Generator[str, None, None]:
    """Yields cleaned text blocks from an EPUB file."""
    book = epub.read_epub(file_path)
    for item in book.get_items():
        if item.media_type == "application/xhtml+xml":
            soup = BeautifulSoup(item.get_content(), "html.parser")
            for tag in soup.find_all(["p", "div", "section"]):
                text = tag.get_text(strip=True)
                if text:
                    yield text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audio Player for TTS and EPUB/text streaming",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the EPUB or TXT file to read text from",
    )
    parser.add_argument(
        "-v",
        "--volume",
        type=float,
        default=0.5,
        help="Volume level for audio playback (0.0 to 1.0)",
    )
    parser.add_argument(
        "-s",
        "--sampling-rate-ratio",
        type=float,
        default=1.0,
        help="Ratio to adjust the sampling rate of the audio playback (default: 1.0)",
    )
    parser.add_argument(
        "-b",
        "--block-skip",
        type=int,
        default=0,
        help="Number of text blocks to skip before starting playback",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (if specified, audio will be saved instead of played)",
    )
    parser.add_argument(
        "--voice-clone",
        type=str,
        default=None,
        help="If provided, will clone the targeted voice (Chatterbox only)",
    )
    parser.add_argument(
        "--tts",
        choices=["kitten", "chatterbox"],
        default="kitten",
        help="Choose TTS backend (default: kitten)",
    )
    parser.add_argument(
        "--kitten-voice",
        type=str,
        default="expr-voice-4-f",
        help="Kitten voice name (only used if --tts kitten)",
    )
    parser.add_argument(
        "--kitten-speed",
        type=float,
        default=1.3,
        help="Kitten speech speed (only used if --tts kitten)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(module)s %(filename)s:%(lineno)d - %(message)s",
    )

    args = parse_args()

    if not os.path.exists(args.file_path):
        raise ValueError(f"File {args.file_path} does not exist")

    # Prepare text generator
    if args.file_path.endswith(".epub"):
        logging.info(f"Reading EPUB file: {args.file_path}")
        generator = epub_text_generator(args.file_path)
        for _ in range(args.block_skip):
            next(generator)  # Skip unwanted blocks
    elif args.file_path.endswith(".txt"):
        if args.block_skip != 0:
            logging.warning("Block skip not supported for text files, ignoring.")

        def my_generator():
            with open(args.file_path, "r", encoding="utf-8") as file:
                for line in file:
                    yield line.strip()

        logging.info(f"Reading text file: {args.file_path}")
        generator = my_generator()
    else:
        raise ValueError("Unsupported file type. Use .epub or .txt")

    # Dynamically import backend
    if args.tts == "chatterbox":
        from audio.chatterbox_player import ChatterboxAudioPlayer as AudioPlayer

        player = AudioPlayer(
            volume=args.volume,
            target_punctuation_stop=["."],
            voice_clone=args.voice_clone,
        )
        if args.voice_clone is None:
            player.change_model_encoding(torch.bfloat16)
        if player.device == "cuda":
            logging.info("Using CUDA device, compiling model...")
            player.torch_compile()
    else:
        from audio.kitten_player import KittenAudioPlayer as AudioPlayer

        player = AudioPlayer(
            volume=args.volume,
            target_punctuation_stop=["."],
            kitten_voice=args.kitten_voice,
            kitten_speed=args.kitten_speed,
        )

    # Output to file or stream
    if args.output is not None:
        player.stream_to_file(generator, args.output)
    else:
        player.stream(generator)
