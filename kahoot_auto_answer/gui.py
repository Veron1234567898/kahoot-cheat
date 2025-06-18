import tkinter as tk
from tkinter import messagebox, scrolledtext
import os
import subprocess
import threading

class KahootAutoAnswerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Kahoot Auto Answer")

        # API Key Input
        self.api_key_label = tk.Label(master, text="Gemini API Key:")
        self.api_key_label.pack()
        self.api_key_entry = tk.Entry(master, width=50)
        self.api_key_entry.pack()
        # Pre-fill if environment variable exists
        if "GEMINI_API_KEY" in os.environ:
            self.api_key_entry.insert(0, os.environ["GEMINI_API_KEY"])

        # Start Button
        self.start_button = tk.Button(master, text="Start Auto Answer", command=self.start_auto_answer)
        self.start_button.pack()

        # Stop Button
        self.stop_button = tk.Button(master, text="Stop Auto Answer", command=self.stop_auto_answer, state=tk.DISABLED)
        self.stop_button.pack()

        # Output Console
        self.output_console = scrolledtext.ScrolledText(master, height=20, width=80, state=tk.DISABLED)
        self.output_console.pack()

        self.process = None
        self.thread = None

    def start_auto_answer(self):
        api_key = self.api_key_entry.get()
        if not api_key:
            messagebox.showerror("Error", "Please enter your Gemini API Key.")
            return

        os.environ["GEMINI_API_KEY"] = api_key

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.output_console.config(state=tk.NORMAL)
        self.output_console.delete(1.0, tk.END)
        self.output_console.insert(tk.END, "Starting Kahoot Auto Answer...\n")
        self.output_console.config(state=tk.DISABLED)

        # Run the main script as a subprocess and capture output
        def run_process():
            try:
                # Adjust path if needed
                self.process = subprocess.Popen(
                    ["python", "kahoot_auto_answer/main.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                for line in self.process.stdout:
                    self.append_output(line)
                self.process.stdout.close()
                self.process.wait()
            except Exception as e:
                self.append_output(f"Failed to start main script: {e}\n")
            finally:
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)

        self.thread = threading.Thread(target=run_process, daemon=True)
        self.thread.start()

    def stop_auto_answer(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.append_output("Process terminated by user.\n")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def append_output(self, text):
        self.output_console.config(state=tk.NORMAL)
        self.output_console.insert(tk.END, text)
        self.output_console.see(tk.END)
        self.output_console.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = KahootAutoAnswerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
