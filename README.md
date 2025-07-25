# EPUB to Audio Streamer

A simple test project to convert text from EPUB files into an audio stream using a text-to-speech (TTS) model.

## Description

This project reads text from an EPUB file and streams it as audio. It uses a TTS model to convert text to speech and plays the audio in real-time. The project is designed to handle text chunking, audio crossfading, and volume adjustment for a smooth listening experience.

## Features

| Feature | Description |
|---------|-------------|
| EPUB Reading | Read text from EPUB files. |
| Text Chunking | Split text into manageable chunks for TTS processing. |
| Audio Streaming | Stream text as audio in real-time. |
| Volume Adjustment | Adjustable volume for audio playback. |
| Sampling Rate Adjustment | Adjustable sampling rate for audio playback. |
| Crossfading | Crossfading for smooth audio transitions. |
| Command-Line Interface | Easy usage through command-line arguments. |

## Requirements

| Requirement | Description |
|-------------|-------------|
| Python | Python 3.x |
| Libraries | `sounddevice`, `ebooklib`, `torch`, `BeautifulSoup`, `argparse`, `logging` |

## Installation

1. Clone the repository:

```bash
git clone <repository-url> <target-directory>
cd <target-directory>
git submodule update --init --recursive
```

2. Set up a virtual environment (optional but recommended):

```bash 
python -m venv .venv # Create a virtual environment (if desired or required)
source .venv/bin/activate # Activate the virtual environment (Linux/Mac)
# or
.\.venv\Scripts\activate # Activate the virtual environment (Windows)

3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

To use the script, run it from the command line, providing the path to an EPUB file and optionally specifying the volume and sampling rate ratio:

```bash
python script.py path/to/your/book.epub -v 0.75 -s 0.95
```

| Argument | Description | Default |
|----------|-------------|---------|
| `file_path` | Path to the EPUB file to read text from. | **REQUIRED** |
| `-v`, `--volume` | Volume level for audio playback (0.0 to 1.0). | 0.5 |
| `-s`, `--sampling-rate-ratio` | Ratio to adjust the sampling rate of the audio playback. | 1.0 (currently used in the script, but not working as intended :sad:) |
| `-h`, `--help` | Show help message and exit. | - |


## Example

1. Fetch an EPUB file

You can found some example EPUB files online or create your own. For example, you can download a sample EPUB file from [Project Gutenberg](https://www.gutenberg.org/) or [Open Library](https://openlibrary.org/).

2. Run the script with the EPUB file:
```bash
python script.py path/to/your/book.epub -v 0.8 -s 1.0
```

## License

This project is licensed under the MIT License.

