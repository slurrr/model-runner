# Nanbeige4.1-3B templates

This folder is a place to keep **repo-owned** prompt/template overrides or experiments for `Nanbeige4.1-3B` without editing files inside the model checkpoint directory.

Current sources of truth for this checkpoint live inside the local model folder:
- `/home/poop/ml/models/Nanbeige4.1-3B/tokenizer_config.json` (default chat template)
- `/home/poop/ml/models/Nanbeige4.1-3B/tokenizer_config_search.json` (alternate “search” template)

If you create overrides here, prefer storing them as:
- a JSON file containing `{"chat_template": "...jinja..."}` (compatible with `chat.py --chat-template <path>`)
- or a `.jinja` file if it’s purely for experimentation/reference

