
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
import argparse
import torch
import yaml
from safetensors.torch import load_model


from GPT.Nova import NovaLM


def load_config() -> dict:
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device: str | torch.device | None) -> torch.device:
    if device:
        device = torch.device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    return device


def build_model(config: dict, checkpoint_dir: Path, device: str | torch.device | None) -> NovaLM:
    model_file = checkpoint_dir / "model.safetensors"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n  {model_file}\n"
            f"Expected a final checkpoint containing model.safetensors."
        )

    print(f"Loading checkpoint: {checkpoint_dir}")
    print(f"Device: {device}")

    model = NovaLM(config)
    load_model(model, str(model_file), strict=True, device=str(device))
    model.to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}\n\n")


    return model

def checkpoint_path(config: dict, phase: int) -> Path:
    save_dir = ROOT / config["Train"]["save_dir"]
    return save_dir / f"Phase_{phase}" / "final"


def run_generation(model: NovaLM, config: dict, device: torch.device, args: argparse.Namespace) -> None:
    generation_cfg = config.get("Generation", {})

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter prompt: ")

    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty

    if temperature is None:
        temperature = float(generation_cfg.get("temperature", 0.7))
    if top_k is None:
        top_k = int(generation_cfg.get("top_k", 0))
    if top_p is None:
        top_p = float(generation_cfg.get("top_p", 1.0))
    if repetition_penalty is None:
        repetition_penalty = float(generation_cfg.get("repetition_penalty", 1.0))

    print("=" * 72)
    print(f"Nova generation — Phase {args.phase}")
    print("=" * 72)
    print(f"Prompt: {prompt!r}")
    print(
        f"Settings: max_new_tokens={args.max_new_tokens}, "
        f"temperature={temperature}, top_k={top_k}, "
        f"top_p={top_p}, repetition_penalty={repetition_penalty}"
    )
    print("-" * 72)

    output = model.generate(prompt, max_new_tokens=args.max_new_tokens,  temperature=temperature, top_k=top_k,
                            top_p=top_p, repetition_penalty=repetition_penalty, device=device, return_prompt=True)

    print(output)
    print("-" * 72)


def run_chat(model: NovaLM, config: dict, device: torch.device, args: argparse.Namespace) -> None:
    generation_cfg = config.get("Generation", {})

    temperature = (args.temperature if args.temperature is not None else float(generation_cfg.get("temperature", 0.7)))
    top_k = (args.top_k if args.top_k is not None else int(generation_cfg.get("top_k", 0)))
    top_p = (args.top_p if args.top_p is not None else float(generation_cfg.get("top_p", 1.0)))
    repetition_penalty = (args.repetition_penalty if args.repetition_penalty is not None else float(generation_cfg.get("repetition_penalty", 1.0)))

    default_system = generation_cfg.get("default_sys_prompt")

    messages: list[dict[str, str]] = []

    print("=" * 72)
    print("Nova chat — Phase 3")
    print("=" * 72)
    print("Type 'exit', 'quit', or 'bye' to stop.")
    print("Type '/clear' to clear the conversation history.")
    print("-" * 72)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nNova: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Nova: Goodbye!")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            print("Conversation cleared.")
            print()
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            thinking, answer = model.chat(messages, max_new_tokens=args.max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p,  repetition_penalty=repetition_penalty,
                                          system_prompt=default_system, device=device, return_thinking=True)
        except RuntimeError as exc:
            # Keep the history valid if generation fails.
            messages.pop()
            print(f"Generation error: {exc}")
            print()
            continue

        if thinking and args.show_thinking:
            print(f"Nova [thinking]: {thinking}")

        print(f"Nova: {answer}")
        print()

        messages.append({"role": "assistant", "content": answer})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Nova 2 generation and Phase 3 chat."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--chat", action="store_true", help="Chat with the Phase 3 final checkpoint.")
    mode.add_argument("--generate", action="store_true", help="Generate text from a Phase 1 or Phase 2 final checkpoint.")

    parser.add_argument("--phase", type=int, default=None, choices=(1, 2), help="Phase to load for --generate.",)
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for --generate.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. 0 = greedy.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling. 0 = disabled.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Nucleus sampling threshold. 1.0 = disabled.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        help="Penalty applied to already-seen tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--show-thinking",
        action="store_true",
        help="In chat mode, print the model's <thinking> section when present.",
    )

    args = parser.parse_args()

    if args.generate and args.phase is None:
        parser.error("--generate requires --phase 1 or --phase 2.")

    if args.chat and args.phase is not None:
        parser.error("--phase is only used with --generate.")

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be greater than 0.")

    if args.top_k is not None and args.top_k < 0:
        parser.error("--top-k must be >= 0.")

    if args.top_p is not None and not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1].")

    if args.repetition_penalty is not None and args.repetition_penalty <= 0.0:
        parser.error("--repetition-penalty must be > 0.")

    return args


def main() -> None:
    args = parse_args()
    config = load_config()
    device = get_device(args.device)

    if args.chat:
        phase = 3
    else:
        phase = args.phase

    checkpoint_dir = checkpoint_path(config, phase)
    model = build_model(config, checkpoint_dir, device)

    if args.chat:
        run_chat(model, config, device, args)
    else:
        run_generation(model, config, device, args)


if __name__ == "__main__":
  """
  python test.py --chat
  python test.py --generate --phase 1
  python test.py --generate --phase 2
  
  python test.py --generate --phase 1 --prompt "The future of AI is"
  python test.py --generate --phase 2 --temperature 0.8 --top-k 20
  python test.py --chat --show-thinking
  python test.py --chat --max-new-tokens 256
    """
  main()
