from openai import OpenAI
import threading
import tkinter as tk
from tkinter import scrolledtext, filedialog, Menu
from datetime import datetime
import os
from docx import Document
from pypdf import PdfReader
import base64
import urllib.request
import traceback

# Pillow for image loading & conversion to Tkinter PhotoImage
from PIL import Image, ImageTk

from config import DEFAULT_MODEL_VERSION, ASSISTANT_NAME, IMAGE_MODEL_VERSION, AVAILABLE_MODELS

try:
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLE_FILTER = getattr(Image, "LANCZOS", Image.BICUBIC)

# Initialize the OpenAI API with your API key
client = OpenAI(
    api_key=os.getenv("OPENAI_KEY", "No_Key"),  # this is also the default, it can be omitted
)

# Default model: prefer the config value if present and in the list, otherwise use the first available
_default_model = DEFAULT_MODEL_VERSION if DEFAULT_MODEL_VERSION in AVAILABLE_MODELS else AVAILABLE_MODELS[0]

# Tkinter variable to hold the currently selected model for chat completions.
# This allows the user to switch models from the GUI while preserving conversation context.
# Note: We don't change IMAGE_MODEL_VERSION here; image generation still uses the config value.
current_model_var = None  # will be initialized after app is created

conversation = [
    {"role": "system", "content": f"You are chatting with a user. Respond helpfully. Your name is {ASSISTANT_NAME}"}
]

# Lock to guard access to conversation when used from multiple threads
conversation_lock = threading.Lock()

# Variables for waiting/animation state (used on main thread)
waiting = False
anim_after_id = None
anim_dots = 0

# Keep references to PhotoImage objects so Tkinter doesn't garbage-collect them
image_refs = []

# Default folder for saving images
default_image_folder = os.getcwd()

# Functions to control the visual waiting indicator
def animate_wait():
    global anim_dots, anim_after_id
    if not waiting:
        return
    anim_dots = (anim_dots + 1) % 4  # 0..3 dots
    status_label.config(text="Thinking" + "." * anim_dots)
    anim_after_id = app.after(500, animate_wait)  # schedule next frame

def start_waiting():
    global waiting, anim_dots, anim_after_id
    if waiting:
        return
    waiting = True
    anim_dots = 0
    status_label.config(text="Thinking")
    # start the animation loop
    anim_after_id = app.after(500, animate_wait)

def stop_waiting():
    global waiting, anim_after_id
    if not waiting:
        return
    waiting = False
    # cancel scheduled after callback if any
    if anim_after_id is not None:
        try:
            app.after_cancel(anim_after_id)
        except Exception:
            pass
        anim_after_id = None
    status_label.config(text="Idle")

# Function to handle the OpenAI chat API call (non-blocking, streaming)
def send_message():
    user_message = user_input.get("1.0", tk.END).strip()
    if not user_message:
        return

    # Disable the send button on the main thread immediately
    send_button.config(state='disabled')

    # Append the user's message to the shared conversation under the lock
    with conversation_lock:
        conversation.append({"role": "user", "content": user_message})

    # Update the UI to show the user's message and clear input (on main thread)
    def show_user_message():
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, "You: " + user_message + "\n")
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)
        user_input.delete("1.0", tk.END)

    app.after(0, show_user_message)

    # Start the waiting indicator (on main thread)
    start_waiting()

    # Capture the selected model at the moment of sending so the request uses the intended model
    selected_model = current_model_var.get()

    # Worker function runs in a background thread to avoid freezing the UI
    def worker(model_to_use):
        try:
            # Make a snapshot/copy of the conversation to send to the API
            with conversation_lock:
                messages_for_api = conversation.copy()

            # Use streaming to receive partial responses
            stream = client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_api,
                stream=True
            )

            # Insert assistant header once (on main thread)
            def start_assistant_message():
                chat_history.configure(state='normal')
                chat_history.insert(tk.END, f"{ASSISTANT_NAME}: ")
                chat_history.configure(state='disabled')
                chat_history.see(tk.END)

            app.after(0, start_assistant_message)

            assistant_full = ""

            # Iterate over streaming events
            for event in stream:
                # The structure may vary slightly depending on SDK versions.
                # We attempt to get delta content in a robust way.
                delta_text = ""
                try:
                    choice0 = event.choices[0]
                    delta = getattr(choice0, "delta", None)
                    if delta is None:
                        delta = choice0.get("delta", {})
                    if hasattr(delta, "content"):
                        delta_text = delta.content or ""
                    else:
                        delta_text = delta.get("content", "") if isinstance(delta, dict) else ""
                except Exception:
                    delta_text = ""

                if delta_text:
                    assistant_full += delta_text
                    # Append this chunk to the chat history on the main thread
                    def append_chunk(chunk=delta_text):
                        chat_history.configure(state='normal')
                        chat_history.insert(tk.END, chunk)
                        chat_history.configure(state='disabled')
                        chat_history.see(tk.END)
                    app.after(0, append_chunk)

            # After stream completes, add a couple of newlines and append to conversation
            def finalize_assistant_message():
                chat_history.configure(state='normal')
                chat_history.insert(tk.END, "\n\n")
                chat_history.configure(state='disabled')
                chat_history.see(tk.END)

            app.after(0, finalize_assistant_message)

            # Append assistant response to shared conversation under the lock
            with conversation_lock:
                conversation.append({"role": "assistant", "content": assistant_full})

        except Exception as e:
            # Schedule UI update on the main thread to show error
            def show_error():
                chat_history.configure(state='normal')
                chat_history.insert(tk.END, "Error: " + str(e) + "\n\n")
                chat_history.configure(state='disabled')
                chat_history.see(tk.END)

            app.after(0, show_error)

        finally:
            # Re-enable the send button and stop waiting indicator on the main thread
            def on_request_complete():
                send_button.config(state='normal')
                stop_waiting()

            app.after(0, on_request_complete)

    # Start background thread
    t = threading.Thread(target=worker, args=(selected_model,), daemon=True)
    t.start()

def export_chat():
    filename = f'chat_history-{datetime.now().strftime("%m%b%y-%H%M")}.txt'
    chat_content = chat_history.get('1.0', tk.END)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(chat_content)
    print(f"Exported {filename}")

def read_text_file(path):
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def read_docx_file(path):
    try:
        doc = Document(path)
    except Exception as e:
        raise RuntimeError(f"Error reading DOCX: {e}")
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def read_pdf_file(path):
    try:
        reader = PdfReader(path)
    except Exception as e:
        raise RuntimeError(f"Error opening PDF: {e}")
    text_chunks = []
    for page in reader.pages:
        try:
            text = page.extract_text()
        except Exception:
            text = None
        if text:
            text_chunks.append(text)
    return "\n\n".join(text_chunks)

def import_file():
    filepath = filedialog.askopenfilename(
        title="Import file for context",
        filetypes=[
            ("All files", "*.*"),
            ("Text files", "*.txt;*.md"),
            ("Word documents", "*.docx"),
            ("PDF files", "*.pdf")
        ]
    )
    if not filepath:
        return

    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext in ('.txt', '.md'):
            content = read_text_file(filepath)
        elif ext == '.docx':
            content = read_docx_file(filepath)
        elif ext == '.pdf':
            content = read_pdf_file(filepath)
        else:
            # Try text fallback for unknown extensions
            try:
                content = read_text_file(filepath)
            except Exception:
                raise RuntimeError("Unsupported file type and not readable as text.")
    except Exception as e:
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"Error importing file {os.path.basename(filepath)}: {e}\n\n")
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)
        return

    # Optionally check file length and warn about token limits
    max_chars_warn = 200_000  # arbitrary big number; adjust as desired
    if len(content) > max_chars_warn:
        chat_history.configure(state='normal')
        chat_history.insert(tk.END,
                            f"Warning: imported file is very large ({len(content)} chars). "
                            "You may want to summarize or split it before sending to the model.\n\n")
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)

    # Append to shared conversation as a user turn (thread-safe)
    with conversation_lock:
        conversation.append({"role": "user", "content": content})

    # Show a preview in the chat window (truncate preview if very large)
    preview_limit = 1000
    preview = content if len(content) <= preview_limit else content[:preview_limit] + "\n...[truncated preview]"
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, f"You (imported {os.path.basename(filepath)}):\n{preview}\n\n")
    chat_history.configure(state='disabled')
    chat_history.see(tk.END)

# ------------------------------
# Image generation functions (top-level)
# ------------------------------

def generate_image_dialog():
    dlg = tk.Toplevel(app)
    dlg.title("Generate Image")
    dlg.grab_set()  # modal

    # Prompt label & text
    tk.Label(dlg, text="Enter image prompt:").grid(row=0, column=0, padx=8, pady=(8, 0), sticky='w')
    prompt_text = tk.Text(dlg, wrap=tk.WORD, width=60, height=6)
    prompt_text.grid(row=1, column=0, padx=8, pady=8, sticky='nsew')

    # Size option
    size_frame = tk.Frame(dlg)
    size_frame.grid(row=2, column=0, sticky='w', padx=8)
    tk.Label(size_frame, text="Size:").pack(side='left', padx=(0,6))
    size_var = tk.StringVar(value="1024x1024")
    size_choices = ["256x256", "512x512", "1024x1024"]
    size_menu = tk.OptionMenu(size_frame, size_var, *size_choices)
    size_menu.pack(side='left')

    # Folder selection
    folder_frame = tk.Frame(dlg)
    folder_frame.grid(row=3, column=0, sticky='we', padx=8, pady=(6, 0))
    tk.Label(folder_frame, text="Save folder:").pack(side='left')
    folder_label_var = tk.StringVar(value=default_image_folder)
    folder_label = tk.Label(folder_frame, textvariable=folder_label_var, anchor='w')
    folder_label.pack(side='left', padx=(6,8), fill='x', expand=True)

    def choose_folder():
        chosen = filedialog.askdirectory(title="Choose folder to save images", initialdir=folder_label_var.get() or os.getcwd())
        if chosen:
            folder_label_var.set(chosen)

    choose_btn = tk.Button(folder_frame, text="Choose...", command=choose_folder)
    choose_btn.pack(side='right')

    # Buttons
    button_frame = tk.Frame(dlg)
    button_frame.grid(row=4, column=0, pady=(8, 8), sticky='e')

    def on_submit():
        prompt = prompt_text.get("1.0", tk.END).strip()
        if not prompt:
            return
        size = size_var.get()
        folder = folder_label_var.get() or os.getcwd()
        dlg.destroy()
        start_image_generation(prompt, size, folder)

    submit_btn = tk.Button(button_frame, text="Generate", command=on_submit)
    submit_btn.pack(side='right', padx=5)
    cancel_btn = tk.Button(button_frame, text="Cancel", command=dlg.destroy)
    cancel_btn.pack(side='right')

    dlg.update_idletasks()
    dlg.minsize(dlg.winfo_reqwidth(), dlg.winfo_reqheight())

def start_image_generation(prompt, size, folder):
    # Append user message about image prompt to conversation
    with conversation_lock:
        conversation.append({"role": "user", "content": f"[Image prompt] {prompt}"})

    app.after(0, lambda: show_user_prompt_ui(prompt, size))

    # Start waiting and disable send button while generating
    start_waiting()
    send_button.config(state='disabled')

    # Start the worker thread
    t = threading.Thread(target=image_worker, args=(prompt, size, folder), daemon=True)
    t.start()

def image_worker(prompt, size, folder):
    try:
        # Ensure folder exists
        os.makedirs(folder, exist_ok=True)

        # Call Images API
        response = client.images.generate(
            model=IMAGE_MODEL_VERSION,
            prompt=prompt,
            size=size
        )

        # extract image data (b64 or url)
        img_b64 = None
        img_url = None
        try:
            first = response.data[0]
            img_b64 = getattr(first, "b64_json", None) or (first.get("b64_json") if isinstance(first, dict) else None)
            img_url = getattr(first, "url", None) or (first.get("url") if isinstance(first, dict) else None)
        except Exception:
            pass

        saved_filename = None
        if img_b64:
            try:
                img_bytes = base64.b64decode(img_b64)
                saved_filename = os.path.join(folder, f"image-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
                with open(saved_filename, "wb") as f:
                    f.write(img_bytes)
            except Exception as e:
                raise RuntimeError(f"Failed to decode/save image: {e}")
        elif img_url:
            try:
                saved_filename = os.path.join(folder, f"image-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png")
                urllib.request.urlretrieve(img_url, saved_filename)
            except Exception as e:
                raise RuntimeError(f"Failed to download image from URL: {e}")
        else:
            # If no image data, try to stringify response
            raise RuntimeError(f"No image data returned. Response: {response}")

        # Append assistant message to conversation
        assistant_text = f"[Image generated: {saved_filename}] (prompt: {prompt})"
        with conversation_lock:
            conversation.append({"role": "assistant", "content": assistant_text})

        # Schedule UI update to display assistant message and the inline image
        app.after(0, lambda: show_assistant_image_ui(saved_filename, assistant_text))

    except Exception as e:
        print("Image generation error:", traceback.format_exc())
        app.after(0, lambda: show_image_error_ui(e))

    finally:
        app.after(0, cleanup_after_image)

def show_user_prompt_ui(prompt, size):
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, f"You (image prompt, {size}): {prompt}\n")
    chat_history.configure(state='disabled')
    chat_history.see(tk.END)

def show_assistant_image_ui(saved_filename, assistant_text):
    try:
        # Insert assistant header and text
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"{ASSISTANT_NAME}: {assistant_text}\n")

        # Load image via PIL
        pil_img = Image.open(saved_filename)

        # Compute maximum display width based on chat widget width
        chat_width = chat_history.winfo_width()
        if not chat_width or chat_width < 50:
            max_display_width = 600
        else:
            max_display_width = max(200, chat_width - 80)

        w, h = pil_img.size
        if w > max_display_width:
            ratio = max_display_width / w
            new_size = (int(w * ratio), int(h * ratio))

            pil_img = pil_img.resize(new_size, resample=RESAMPLE_FILTER)

        # Convert to PhotoImage for Tkinter and keep a reference
        tk_img = ImageTk.PhotoImage(pil_img)
        image_refs.append(tk_img)  # keep reference so Tk doesn't GC it

        # Insert the image into the text widget
        chat_history.image_create(tk.END, image=tk_img)
        chat_history.insert(tk.END, "\n\n")
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)

    except Exception as e_ui:
        chat_history.configure(state='normal')
        chat_history.insert(tk.END, f"Error displaying image: {e_ui}\n\n")
        chat_history.configure(state='disabled')
        chat_history.see(tk.END)

def show_image_error_ui(exc):
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, "Error generating image: " + str(exc) + "\n\n")
    chat_history.configure(state='disabled')
    chat_history.see(tk.END)

def cleanup_after_image():
    send_button.config(state='normal')
    stop_waiting()

# Model menu: provides radio entries so the user can switch models at runtime without losing conversation context
def on_model_change():
    # Called on the main thread when the user selects a different model.
    selected = current_model_var.get()
    # Optionally indicate in the chat that the model was switched (keeps conversation context intact).
    chat_history.configure(state='normal')
    chat_history.insert(tk.END, f"[Model switched to: {selected}]\n\n")
    chat_history.configure(state='disabled')
    chat_history.see(tk.END)

# ------------------------------
# Build the GUI
# ------------------------------

# Create the main window
app = tk.Tk()
app.title("ChatGPT GUI")

# Initialize the current_model_var as a Tkinter StringVar bound to the main app
current_model_var = tk.StringVar(app, value=_default_model)

# Add a menu bar with 'File' menu and a new 'Model' menu for selecting chat model
menu = Menu(app)
app.config(menu=menu)

# File menu
filemenu = Menu(menu, tearoff=0)
menu.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Export Chat", command=export_chat)
filemenu.add_command(label="Import File", command=import_file)
filemenu.add_command(label="Generate Image...", command=generate_image_dialog)

modelmenu = Menu(menu, tearoff=0)
menu.add_cascade(label="Model", menu=modelmenu)
for m in AVAILABLE_MODELS:
    modelmenu.add_radiobutton(label=m, variable=current_model_var, value=m, command=on_model_change)

# Configure grid weights so the UI resizes with the window.
# Give the chat history the expanding weight; keep input row fixed/small.
app.grid_rowconfigure(0, weight=1)   # chat history expands
app.grid_rowconfigure(1, weight=0)   # input stays compact
app.grid_rowconfigure(2, weight=0)   # status minimal

# Column 0 (main text area) expands horizontally; column 1 (send button) keeps a fixed width.
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=0, minsize=80)

# Create a text area for the chat history
chat_history = scrolledtext.ScrolledText(app, state='disabled', wrap=tk.WORD)
chat_history.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

# Create a text area for user input (smaller height)
user_input = tk.Text(app, wrap=tk.WORD, height=4)  # <-- smaller message box
user_input.grid(row=1, column=0, padx=10, pady=10, sticky='nsew')

# Button to send message
send_button = tk.Button(app, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10, sticky='ns')

# Status label to indicate waiting/idle
status_label = tk.Label(app, text="Idle", anchor='w')
status_label.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=(0,10))

# Let Tk compute requested sizes, then set the window's minimum and initial geometry
app.update_idletasks()
min_w = app.winfo_reqwidth()
min_h = app.winfo_reqheight()
app.minsize(min_w, min_h)
app.geometry(f"{min_w}x{min_h}")

app.mainloop()