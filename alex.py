import argparse
import os
import re
import sys


def normalize_model_path(raw_path: str) -> str:
    path = os.path.expanduser(raw_path.strip())

    # Convert Windows path like C:\models\foo.gguf to WSL path /mnt/c/models/foo.gguf
    win_drive = re.match(r"^([A-Za-z]):[\\/](.*)$", path)
    if win_drive:
        drive = win_drive.group(1).lower()
        rest = win_drive.group(2).replace("\\", "/")
        path = f"/mnt/{drive}/{rest}"

    return os.path.abspath(path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal GGUF chat CLI via llama.cpp")
    parser.add_argument("model_path", help="Path to .gguf model file")
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context window")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max generated tokens per turn")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all possible)",
    )
    parser.add_argument(
        "--system",
        default="You are a helpful assistant.",
        help="System prompt",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    model_path = normalize_model_path(args.model_path)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Tip: In WSL, Windows C: drive is usually mounted at /mnt/c")
        sys.exit(1)
    if not model_path.lower().endswith(".gguf"):
        print(f"Expected a .gguf file, got: {model_path}")
        sys.exit(1)

    try:
        from llama_cpp import Llama, _internals

        # Work around a llama-cpp-python cleanup bug when model init fails
        # before `sampler` is created on the internal model wrapper.
        original_close = _internals.LlamaModel.close

        def safe_close(self):
            if not hasattr(self, "sampler"):
                self.sampler = None
            if not hasattr(self, "custom_samplers"):
                self.custom_samplers = []
            if not hasattr(self, "_exit_stack"):
                return
            try:
                original_close(self)
            except Exception:
                return

        _internals.LlamaModel.close = safe_close
    except Exception as exc:
        print("Missing dependency: llama-cpp-python")
        print("Install with: .venv/bin/pip install llama-cpp-python")
        print(f"Import error: {exc}")
        sys.exit(1)

    print(f"Loading GGUF model: {model_path}")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
        )
    except Exception as exc:
        print(f"Failed to load GGUF model: {exc}")
        sys.exit(1)

    messages = [{"role": "system", "content": args.system}]
    print("Chat ready. Commands: /exit, /quit, /clear")
    while True:
        try:
            user_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_text.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if user_text.lower() == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("Conversation cleared.")
            continue
        if not user_text:
            continue

        messages.append({"role": "user", "content": user_text})
        try:
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            assistant_text = response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            # Fallback for GGUFs that do not include a chat template.
            try:
                prompt_parts = [f"System: {args.system}"]
                for msg in messages[1:]:
                    role = msg["role"].capitalize()
                    prompt_parts.append(f"{role}: {msg['content']}")
                prompt_parts.append("Assistant:")
                prompt = "\n".join(prompt_parts)

                response = llm.create_completion(
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    stop=["\nUser:", "\nSystem:"],
                )
                assistant_text = response["choices"][0]["text"].strip()
            except Exception as fallback_exc:
                print(f"Generation failed: {exc}")
                print(f"Fallback completion also failed: {fallback_exc}")
                messages.pop()
                continue

        print(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})


if __name__ == "__main__":
    main()
