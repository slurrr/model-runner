# entrypoints for running the model in different modes

python runner.py Nanbeige4.1-3B
python runner.py --config Nanbeige4.1-3B
python runner.py Nanbeige4.1-3B -8bit
python runner.py Nanbeige4.1-3B -4bit

python chat.py Nanbeige4.1-3B
python chat.py --config Nanbeige4.1-3B
python chat.py Nanbeige4.1-3B --dtype bfloat16
python chat.py Nanbeige4.1-3B -8bit
python chat.py Nanbeige4.1-3B -4bit --dtype float16 --system "You are concise."

python tui_chat.py --config Nanbeige4.1-3B
python tui_chat.py --config Nanbeige4.1-3B --show-thinking
