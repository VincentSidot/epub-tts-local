import sys
import argparse
from typing import Callable, Dict, List

from ollama import Client
from rich.console import Console
from rich.panel import Panel
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter

# === Globals ===
console = Console()
chat_history: List[Dict[str, str]] = []
commands: Dict[str, Dict[str, Callable[[List[str]], bool]]] = {}
model_name: str = "gemma3n:e4b"
client: Client = None
system_prompt = "You are a helpful assistant."  # default system prompt


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


def chat_loop():
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
                console.print(content, end="", soft_wrap=True)
                output += content

            console.print()

            chat_history.append({"role": "assistant", "content": output})

        except KeyboardInterrupt:
            console.print("\n🛑 [dim]Interrupted. Exiting.[/dim]")
            sys.exit(0)


# === CLI Argument Parser ===
def parse_args():
    parser = argparse.ArgumentParser(description="Ollama Chat CLI")
    parser.add_argument("--model", "-m", default="gemma3n:e4b", help="Model to use")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    return parser.parse_args()


# === Main ===
def main():
    global model_name, client
    args = parse_args()
    model_name = args.model
    client = Client(host=args.host)
    chat_loop()


if __name__ == "__main__":
    main()
