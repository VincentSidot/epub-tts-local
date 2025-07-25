from ebooklib import epub
from bs4 import BeautifulSoup
from typing import Generator


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


path = "leo-tolstoy_resurrection.epub"

for i, text_block in enumerate(epub_text_generator(path)):
    print(
        f"Text block {i}: {text_block[:10]}..."
    )  # Print first 10 characters of each block
    if i >= 10:  # Limit to first 10 blocks for brevity
        break

book = epub.read_epub("leo-tolstoy_resurrection.epub")

for item in book.get_items():
    print(
        f"Name: {item.get_name()}, Type: {item.get_type()}, Media type: {item.media_type}"
    )
