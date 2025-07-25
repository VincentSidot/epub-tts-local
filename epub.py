import os
import argparse
import logging
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Generator

from reader import AudioPlayer
import torch


def epub_text_generator(file_path: str) -> Generator[str, None, None]:
    """Yields cleaned text blocks from an EPUB file."""
    book = epub.read_epub(file_path)

    for item in book.get_items():
        if item.media_type == "application/xhtml+xml":
            soup = BeautifulSoup(item.get_content(), "html.parser")

            # You can tweak these tags depending on the EPUB's formatting
            for tag in soup.find_all(["p", "div", "section"]):
                text = tag.get_text(strip=True)
                if text:
                    yield text


def args():
    parser = argparse.ArgumentParser(
        description="Audio Player for TTS and EPUB text streaming",
    )
    parser.add_argument(
        "file_path",
        type=str,
        help="Path to the EPUB file to read text from",
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
        help="Ratio to adjust the sampling rate of the audio playback (default: 1.0) lower values will decrease the playback speed",
    )

    parser.add_argument(
        "-b",
        "--block-skip",
        type=int,
        default=0,
        help="Number of epub text block to skip before starting playback (this is usefull to skip the first trash blocks at start)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output file path (if specified, the output will be saved to this file else the output will be playbacked)",
    )

    parser.add_argument(
        "--voice-clone",
        type=str,
        default=None,
        help="If provided it will clone the targeted voice",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        # Format is [level] location@file:line:char - message
        format="[%(levelname)s] %(module)s %(filename)s:%(lineno)d - %(message)s",
    )

    args = args()

    if not os.path.exists(args.file_path):
        raise ValueError(f"File {args.file_path} does not exist")

    if args.file_path.endswith(".epub"):
        logging.info(f"Reading EPUB file: {args.file_path}")

        generator = epub_text_generator(args.file_path)

        for _ in range(args.block_skip):
            next(generator)  # Skip the first block of text

    elif args.file_path.endswith(".txt"):
        if args.block_skip != 0:
            logging.warning(
                "Block skip is not supported for text files, ignoring block skip"
            )

        def my_generator():
            with open(args.file_path, "r") as file:
                for line in file:
                    yield line.strip()

        logging.info(f"Reading text file: {args.file_path}")
        generator = my_generator()

    player = AudioPlayer(
        volume=args.volume,
        target_punctuation_stop=["."],
        device="cpu",
        voice_clone=args.voice_clone,
    )

    if args.voice_clone is None:
        # Currently model encoding work only without default voices
        player.change_model_encoding(torch.bfloat16)

    if player.device == "cuda":
        logging.info("Using CUDA device, compile model...")
        player.torch_compile()

    if args.output is not None:
        player.stream_to_file(generator, args.output)
    else:
        player.stream(generator)
