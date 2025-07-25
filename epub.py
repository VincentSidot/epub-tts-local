import argparse
import logging
from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Generator

from reader import AudioPlayer


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

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,  # Set the logging level to INFO
        # Format is [level] location@file:line:char - message
        format="[%(levelname)s] %(module)s %(filename)s:%(lineno)d - %(message)s",
    )

    args = args()

    logging.info(f"Reading EPUB file: {args.file_path}")

    player = AudioPlayer(
        volume=args.volume,
        target_punctuation_stop=["."],
    )
    player.torch_compile()

    player.stream(epub_text_generator(args.file_path))
