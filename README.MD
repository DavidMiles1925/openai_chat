# GPT GUIde

A lightweight desktop chat application that integrates with OpenAI models and provides a simple GUI for chatting, importing documents for context, exporting chat history, and generating images. The UI is implemented in Python using Tkinter and the OpenAI Python SDK.

- Default assistant name: `Jeeves` (configurable in `config.py`)
- Default model: `gpt-5-mini` (configurable in `config.py`)

---

## Features

- Chat with an OpenAI model using a desktop GUI (Tkinter).
- Streaming assistant responses in the chat window for a responsive feel.
- Import files for context: `.txt`, `.md`, `.docx`, `.pdf`, `xlsx`, and any text-based file. (preview shown and file appended to conversation).
- Export chat history to a timestamped `.txt` file.
- Generate images from prompts (256x256, 512x512, 1024x1024); images saved to disk and displayed inline.
- Simple animated "Thinking..." indicator while requests are in flight.
- Threaded networking so the UI remains responsive.
- Model selection menu: switch which chat model you are talking to at runtime without losing conversation context.
  - Switching models preserves the conversation context: imported files and previous messages remain in the shared conversation, so the new model receives the full conversation history for subsequent requests.
- Simple configuration via `config.py`.
- In-app management of API Key through menu - Set, remove, or test API Key

---

## Requirements

- Python 3.8+ (3.10+ recommended)
- Network access to the OpenAI API
- Tkinter (usually included with Python; see OS-specific section below if missing)
- See `requirements.txt` for Python dependencies and testing tools

---

## Installation

1. Clone the repo:

```bash
git clone <your-repo-url>
cd <your-repo>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

[**Linux Instructions for installing tkinter**](#linux-debianubuntu--fedora--arch)

3. Set your API Key using the instructions in the secion below titled [API Key Management](#set-the-api-key)

---

## Configuration

### Configuring the program

`config.py` is used to control basic settings in the app.

### Set the API Key

You can provide an OpenAI API key in two ways:

1. In-app (**RECOMMENDED: You can set your API Key by clicking the `Key` Menu and Selecting `Set API Key`**)

2. Environment variable (`OPENAI_KEY`)

### Available Models

- The model can be switched via the `Model` menu.
- The GUI's Model menu uses the list of available model names (see Features). The menu selection overrides the `DEFAUL_MODEL_VERSION` runtime default for chat requests while the app is running. You can update the list of models in the `config.py` file.
- On startup, the app will prefer the `DEFAULT_MODEL_VERSION` from `config.py` as the initial selection if it matches one of the available menu options. Otherwise it will default to the first model in the menu.
- Changing the selected model from the menu does not clear or alter the conversation history — it only changes which model will receive the next request.
- Image generation still uses `IMAGE_MODEL_VERSION` from `config.py` (unchanged).

---

## Running the app

With virtualenv active and config set:

```bash
python main.py
```

The GUI will open. Type messages in the bottom input area and press "Send".

---

## Usage (UI guide)

### Interface

- Chat area: top pane shows the conversation (read-only). Assistant messages stream into this area as they arrive.
- Input box: type your message in the lower text box and click "Send".
- Send button: disabled while a request is in progress.
- Status label: shows "Idle" or "Thinking..." with animated dots.
- Menu → File:
  - Export Chat: saves a timestamped `.txt` file containing the chat contents.
  - Import File: choose a `.txt`, `.md`, `.docx`, or `.pdf`. The content is appended to the conversation as a user message, and a truncated preview is shown in the chat.
  - Generate Image...: open an image prompt dialog to generate images (saved to folder and inserted into chat).
- Menu → Model:
  - Select which chat model to use for subsequent messages.
  - Available models (menu items): gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o-mini.
  - The selected model is used for requests when you press "Send". The app keeps the conversation intact when switching models; the next model receives the full conversation history.
- Menu → Key:
  - Set API Key: Sets the API key with option to use only for session of to persist.
  - Remove API Key: Clears the API key.
  - Test API Key: Sends small API call to validate the key.

### File import / supported formats

- Text files: `.txt`, `.md` — read as UTF-8 (with replacement on encoding errors).
- Word documents: `.docx` — read using `python-docx` (paragraphs joined with newline).
- PDF: `.pdf` — text extracted per page with `pypdf` and joined with double newlines.
- Unknown extensions: attempt to read as text and display an error if not readable.

- The app warns if a file exceeds an arbitrary threshold (default ~200,000 characters in code). Models have token limits; consider summarizing or chunking large files before sending.
- Imported documents are appended as a user message so the model can use them as context for subsequent messages.
- The preview for imported files is truncated beyond a limit (configurable in code).

### Image generation

- The app uses the Images API (config references model `gpt-image-1`).
- Sizes: 256x256, 512x512, 1024x1024.
- The app attempts to handle responses containing either base64 (`b64_json`) or a URL and saves images to the selected folder as `image-YYYYmmdd-HHMMSS.png`.
- Images are displayed inline; the app keeps references to Tk PhotoImage objects to prevent garbage collection.
- Image generation is not affected by the Model menu and continues to use `IMAGE_MODEL_VERSION` from `config.py`.

---

## API Key Management

### App Behavior

- Sources the app will use for an API key (in priority order):

  1. Key stored in the OS keyring (via the `keyring` Python package).
  2. Session key if the user set a key for the current session via the Set API Key dialog (and did not choose to save it).
  3. Environment variable OPENAI_KEY (if present). See

- When you `Set API Key` in the app and choose "Remember", the key is saved into the OS keychain via `keyring`. If you elect not to remember it, the key is used for the current session only (kept in memory) and will be cleared on app restart.
- The app validates keys before saving (makes a tiny, cheap API call).
- The Send and Generate Image controls are only enabled when a usable key is present (session/keyring or env var).
- If you use the `Forget API Key` menu action, the app will:
  - Delete the key stored in the OS keychain (if any).
  - Clear any session-in-memory key and re-initialize the API client.
- However, if you have a persistent environment variable `OPENAI_KEY` set in your OS (for example in your shell profile or system environment), the SDK will continue to see and use that environment variable even after you forget the key in the keychain. In other words:
  - `Forget API Key` will remove the key from the keychain and clear the session, **but it cannot and will not remove a persistent `OPENAI_KEY` environment variable**.
  - To completely stop the app from using an API key that came from an environment variable, you must remove that environment variable from your system (see [Troubleshooting](#troubleshooting) below).

### Setting Enviroment Variables by OS

The enviroment variable that the program will look for (after checking keyring) is called `OPENAI_KEY`

#### Windows

_Set API Key_

```powershell
setx OPENAI_KEY "sk-..."
```

_Remove API Key_

```powershell
[Environment]::SetEnvironmentVariable("OPENAI_KEY", $null, "User")
```

#### macOS

- Set your API key in shell:

```bash
export OPENAI_KEY="sk-..."
```

#### Linux (Debian/Ubuntu / Fedora / Arch)

_Set your API key:_

```bash
export OPENAI_KEY="sk-..."
```

_Install tkinter:_

- Debian / Ubuntu:

```bash
sudo apt update
sudo apt install python3-tk
```

- Fedora:

```bash
sudo dnf install python3-tkinter
```

- Arch:

```bash
sudo pacman -S tk
```

_Remove API Key_

- To remove a persistent export in your shell startup files (~/.bashrc, ~/.profile, ~/.bash_profile, ~/.zshrc):
  - Manually open the file and remove the line like: export OPENAI_KEY="…"
  - Or remove with sed (make a backup first):
    ```bash
    cp ~/.bashrc ~/.bashrc.bak
    sed -i '/^export OPENAI_KEY=/d' ~/.bashrc
    ```
    Then log out/in or source the file:
    ```bash
    source ~/.bashrc
    ```
- If it was set in /etc/environment or ~/.pam_environment:
  ```bash
  sudo cp /etc/environment /etc/environment.bak
  sudo sed -i '/^OPENAI_KEY=/d' /etc/environment
  ```
  Then log out/in for changes to take effect

## Model Guide: Which to Choose

### gpt-5 (full model)

_When it shines_

- Multi-file reasoning (e.g., “Here’s my repo, fix X” and it remembers relationships between files).
- Deep debugging of subtle logic or performance issues.
- Refactoring big chunks of code while keeping style consistent.
- Explaining advanced algorithms in detail before coding them.

_Downsides_

- More expensive per call.
- Slower than mini/nano, so quick “one-liner” requests feel overkill.

### gpt-5-mini

_When it shines_

- Day-to-day Python work: writing functions, fixing bugs, writing tests.
- Understanding short/medium snippets and producing clean solutions.
- Good balance of cost vs correctness — often 90–95% as good as full gpt-5 for typical coding tasks.

_Limits_

- Can stumble on very tricky multi-step logic or nuanced cross-file dependencies.
- More likely than full gpt-5 to “hallucinate” library methods that don’t exist (but still way better than nano).

### gpt-5-nano

_When it shines_

- Super quick “generate boilerplate” tasks:
- Create a FastAPI endpoint skeleton.
- Make a pandas dataframe example.
- Write a regex for X.
- Very cheap for iterative trial-and-error stuff.

_Limits_

- Reasoning depth noticeably lower — more manual correction required.
- Not great at debugging tricky problems or explaining advanced concepts.
- Shorter context means you can’t dump in a whole module.

### gpt-4o-mini

- Better than gpt-5-nano at multi-step reasoning and edge cases.
- Comparable to gpt-5-mini for many everyday Python tasks — especially short-to-medium ones.
- Behind gpt-5-mini and gpt-5 in:
  - Handling very large contexts.
  - Understanding complex, intertwined logic across files.
  - Producing fully correct answers on first try for tricky bugs.

### Latency / Speed

- GPT-4o-mini is very fast — close to gpt-5-nano speeds.
- GPT-5-mini is slightly slower but still quick enough for most dev work.
- GPT-5 is noticeably slower for small requests.

### Cost

Current OpenAI pricing at the time of writing (rounded per 1M tokens):

- GPT-5 (full): ~$1.25 input / ~$10.00 output

- GPT-5-mini: ~$0.25 input / ~$2.00 output

- GPT-4o-mini: ~$0.15 input / ~$0.60 output

- GPT-5-nano: ~$0.05 input / ~$0.40 output

### Practical Usage (for a single developer)

**Default to gpt-5-mini for most coding help — best cost/accuracy trade-off.**

**Switch to gpt-5 only when:**

- You’re stuck debugging something subtle and already tried mini.
- You’re doing big multi-file refactors.
- You need deeper conceptual explanation.

**Use gpt-5-nano for:**

- Tiny, low-risk generation tasks.
- Rapid back-and-forth where you’ll manually review anyway.

---

## Troubleshooting

### Issues/Solutions

- Authentication / invalid API key:
  - Ensure `OPENAI_KEY` is set and valid (no stray quotes, no trailing spaces).
- Forget API Key appears ineffective after restart:
  - If you previously set a persistent environment variable OPENAI_KEY, the app will still pick that up after you forget the key in the keychain. Deleting the keychain entry does not delete environment variables. Remove the OPENAI_KEY environment variable from your system to fully stop the app from seeing that key.
  - To clear the environment variable:
    - Windows (PowerShell session only): $env:OPENAI_KEY = $null
    - Windows (remove persistent user variable, PowerShell):
      [Environment]::SetEnvironmentVariable("OPENAI_KEY", $null, "User")
      (You may need to sign out / sign in or restart to fully remove persistent variables from new sessions.)
  - macOS / Linux (bash/zsh): Remove or edit any lines in ~/.bashrc, ~/.bash_profile or ~/.zshrc that export OPENAI_KEY; then restart the terminal or source the file.
- Keyring backend issues on Linux:
  - On some minimal Linux desktop setups no secret service may be installed; keyring can fall back to an insecure local file backend. If you rely on secure storage, install Secret Service / GNOME Keyring / KWallet support. Optionally install the secretstorage package to improve Secret Service integration (may require dbus).
- `ModuleNotFoundError: No module named 'tkinter'`:
  - Follow the OS-specific instructions above to install Tk support.
- PDF imports produce no text:
  - The PDF may be scanned images. Use OCR or obtain a text-based PDF.
- Streaming doesn't show output:
  - Check network connectivity and the installed OpenAI SDK version. Run from a terminal to view stack traces (easier to debug).
- Image generation returns no image data:
  - Inspect console logs for the raw API response. Response fields vary across SDK/API versions.
- Model switching oddities:
  - Different models have different capabilities and token limits. Switching to a model with a smaller context window may cause long conversation histories to exceed that model's limit; consider clearing or saving/loading conversations if you need to use a model with a much smaller context window.
  - If a model is unavailable or unsupported by your account, API requests will fail — check the error printed to the terminal.

### Implementation notes & internals

- UI built with Tkinter and `ScrolledText` for chat display.
- Conversation is a list of message dicts with keys `role` and `content`.
- Background threads handle network requests to keep the UI responsive.
- A `threading.Lock` (`conversation_lock`) protects access to the shared conversation list.
- Streaming is implemented via `stream=True` from the OpenAI client and appends partial "delta" content to the chat widget as it arrives.
- For better testability, see `file_utils.py` (in repository root), which provides file-reading utilities used by the GUI. You may import that module from `main.py`.
- The Model menu uses Tk radio menu entries bound to a Tkinter StringVar. The selected model's value is captured when sending so the intended model is used for that request.

### Debugging tips:

- Run `python main.py` from a terminal to see printed stack traces and debug messages.
- Add logging or temporary print statements to background threads to inspect API responses.

---

## Resources

### Security & privacy

- Treat the OpenAI API key as sensitive; do not commit it to source control.
- Imported file contents are sent to OpenAI as user messages. Do not import confidential documents if you do not want them transmitted.
- If distributing the app, add user-facing notices and consider adding a local-only mode if/when you support local models.

### Contributing

Follow the guidance in `CONTRIBUTING.md` (included). Summary:

1. Fork the repository.
2. Create a feature branch.
3. Implement and test changes.
4. Open a Pull Request with a description.

### Suggested improvements / roadmap

- Save/load conversation sessions.
- Prompt templates & reply-style presets.
- Configurable model parameters (temperature, max_tokens).
- Automatic chunking & summarization for large files.
- Drag-and-drop file import.
- Theming / Dark mode.
- Local/offline model support.
- Unit tests, CI with OpenAI client mocked, and packaged releases.

### Acknowledgements & License

- Built using the OpenAI API.
- GUI: Tkinter
- DOCX: python-docx
- PDF extraction: pypdf
- Image handling: Pillow

This project is distributed under the MIT License — see the `LICENSE` file in the repository root.
