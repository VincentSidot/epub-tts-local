import logging
import os
import queue
import sys
import argparse
import threading
from typing import Callable, Dict, List, Optional

from ollama import Client
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
import torch

from reader import AudioPlayer

# === Globals ===
console = Console()
chat_history: List[Dict[str, str]] = []
commands: Dict[str, Dict[str, Callable[[List[str]], bool]]] = {}
model_name: str = "gemma3n:e4b"
client: Client = None
system_prompt = "You are a helpful assistant."  # default system prompt


# === Audio Player ===


def inhibit_output(fucntion):
    """Decorator to suppress output of a function."""

    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        try:
            with open(os.devnull, "w") as devnull:
                sys.stdout = devnull
                sys.stderr = devnull
                return fucntion(*args, **kwargs)
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr

    return wrapper


text_chunk_queue = queue.Queue()
end_of_stream_signal = threading.Event()  # Event to signal end of stream
end_stream_signal = object()  # Special object to signal end of stream
end_program_signal = object()  # Special object to signal end of program


# @inhibit_output
def text_to_speech_worker(tts_backend: str):
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()

    # Import only the selected backend to avoid unnecessary dependencies
    if tts_backend == "chatterbox":
        from audio.chatterbox_player import ChatterboxAudioPlayer as AudioPlayer

        audio_player = AudioPlayer(
            volume=0.7,
            target_punctuation_stop=[".", "!", "?"],
            voice_clone=None,  # Default voice clone
        )
    else:  # kitten
        from audio.kitten_player import KittenAudioPlayer as AudioPlayer

        audio_player = AudioPlayer(
            volume=0.7,
            target_punctuation_stop=[".", "!", "?"],
            kitten_voice="expr-voice-4-f",
            kitten_speed=1.3,
        )

    playing_audio = True

    # Only Chatterbox supports this
    if hasattr(audio_player, "change_model_encoding"):
        audio_player.change_model_encoding(torch.bfloat16)

    if getattr(audio_player, "device", None) == "cuda":
        logging.info("Using CUDA device, compile model...")
        if hasattr(audio_player, "torch_compile"):
            audio_player.torch_compile()

    def text_generator():
        while True:
            text_chunk = text_chunk_queue.get()
            if text_chunk is end_stream_signal:
                break
            if text_chunk is end_program_signal:
                playing_audio = False
                break
            yield text_chunk

    while playing_audio:
        audio_player.stream(text_generator())
        end_of_stream_signal.set()


# === Command System ===
def register_command(name: str, description: str, handler: Callable[[List[str]], bool]):
    commands[name] = {"description": description, "handler": handler}


def handle_command(raw: str) -> bool:
    parts = raw.strip().split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd in commands:
        return commands[cmd]["handler"](args)
    else:
        console.print(f"[red]Unknown command:[/red] {cmd}")
        return False


# === Core Handlers ===
def exit_handler(args: List[str]) -> bool:
    console.print("👋 [dim]Exiting...[/dim]")
    return True


def help_handler(args: List[str]) -> bool:
    help_text = "\n".join(
        f"[bold]{name}[/bold] — {meta['description']}"
        for name, meta in commands.items()
    )
    console.print(Panel.fit(help_text, title="Available Commands", border_style="blue"))
    return False


def clear_handler(args: List[str]) -> bool:
    chat_history.clear()
    console.print("🧼 [green]Chat context cleared.[/green]")
    return False


def model_handler(args: List[str]) -> bool:
    global model_name
    if not args:
        console.print(f"📦 [yellow]Current model:[/yellow] {model_name}")
    else:
        model_name = args[0]
        console.print(f"✅ [green]Model set to:[/green] {model_name}")
    return False


# Command handler for /system
def system_handler(args: List[str]) -> bool:
    global system_prompt
    if not args:
        console.print(f"💡 [cyan]Current system prompt:[/cyan] {system_prompt}")
    else:
        # Join args to support multi-word prompts
        system_prompt = " ".join(args)
        console.print(f"✅ [green]System prompt updated to:[/green] {system_prompt}")
    return False


# === Register Commands ===
register_command("/exit", "Exit the chat", exit_handler)
register_command("/help", "Show this help message", help_handler)
register_command("/clear", "Clear chat context (history)", clear_handler)
register_command(
    "/model", "Set or display current model (/model [name])", model_handler
)
register_command(
    "/system", "Set or show system prompt (/system [prompt])", system_handler
)


# === UI & Chat ===
def print_intro():
    console.print(
        Panel.fit(
            f"🤖 [bold]Ollama Chat CLI[/bold]\nModel: [green]{model_name}[/green]\n\n"
            f"Type a message or use [blue]/commands[/blue]. Use /help for a list.",
            border_style="cyan",
        )
    )


def chat_loop(
    tts_queue: Optional[queue.Queue] = None,
    end_of_stream_signal: Optional[threading.Event] = None,
):
    completer = WordCompleter(list(commands.keys()), ignore_case=True)
    print_intro()

    global chat_history
    chat_history = [{"role": "system", "content": system_prompt}]

    while True:
        try:
            user_input = prompt("🧑 You > ", completer=completer).strip()
            if user_input.startswith("/"):
                if handle_command(user_input):
                    break
                continue

            chat_history.append({"role": "user", "content": user_input})
            console.print("🤖 [bold green]Model[/bold green]: ", end="", soft_wrap=True)

            stream = client.chat(
                model=model_name,
                messages=chat_history,
                stream=True,
            )

            output = ""
            for chunk in stream:
                content = chunk["message"]["content"]
                if tts_queue is not None:
                    tts_queue.put(content)  # Send partial text to TTS worker
                console.print(content, end="", soft_wrap=True)
                output += content

            if tts_queue is not None:
                tts_queue.put(end_stream_signal)  # Signal end of stream
            if end_of_stream_signal is not None:
                end_of_stream_signal.wait()  # Wait for TTS worker to finish

            console.print()

            chat_history.append({"role": "assistant", "content": output})

        except KeyboardInterrupt:
            console.print("\n🛑 [dim]Interrupted. Exiting.[/dim]")
            break


# === CLI Argument Parser ===
# === CLI Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Ollama Chat CLI")
    parser.add_argument("--model", "-m", default=model_name, help="Model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument(
        "--tts",
        choices=["kitten", "chatterbox"],
        default="kitten",
        help="Choose TTS backend (default: kitten)",
    )
    return parser.parse_args()


# === Main ===
def main():
    global model_name, client
    args = parse_args()
    model_name = args.model
    client = Client(host=args.host)

    tts_worker = threading.Thread(target=text_to_speech_worker, args=(args.tts,))
    tts_worker.start()

    chat_loop(tts_queue=text_chunk_queue, end_of_stream_signal=end_of_stream_signal)

    text_chunk_queue.put(end_program_signal)

    tts_worker.join()


if __name__ == "__main__":
    main()
