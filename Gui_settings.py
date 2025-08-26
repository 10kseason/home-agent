import os
import json
import yaml
import tkinter as tk
from tkinter import filedialog, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AGENT_CFG_PATH = os.path.join(BASE_DIR, "agent", "config.yaml")
OVERLAY_CFG_PATH = os.path.join(BASE_DIR, "Overlay", "config.yaml")

DEFAULTS = {
    "python": os.path.join(".venv", "Scripts", "python.exe"),
    "overlay_script": os.path.join("Overlay", "overlay_app.py"),
    "ocr_script": os.path.join("OCR", "main.py"),
    "stt_script": os.path.join("STT", "VSRG-Ts-to-kr.py"),
}

class SettingsGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Home-Agent Settings")
        self.resizable(False, False)
        self.entries = {}
        row = 0
        for key, default in [
            ("python", DEFAULTS["python"]),
            ("overlay_script", DEFAULTS["overlay_script"]),
            ("ocr_script", DEFAULTS["ocr_script"]),
            ("stt_script", DEFAULTS["stt_script"]),
        ]:
            tk.Label(self, text=key.replace('_', ' ').title()+":").grid(row=row, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(self, width=50)
            entry.insert(0, default)
            entry.grid(row=row, column=1, padx=5, pady=5)
            btn = tk.Button(self, text="Browse", command=lambda k=key, e=entry: self._browse_file(k, e))
            btn.grid(row=row, column=2, padx=5, pady=5)
            self.entries[key] = entry
            row += 1
        tk.Button(self, text="Save", command=self.save).grid(row=row, column=0, columnspan=3, pady=10)

    def _browse_file(self, key, entry):
        path = filedialog.askopenfilename(initialdir=BASE_DIR)
        if path:
            rel = os.path.relpath(path, BASE_DIR)
            entry.delete(0, tk.END)
            entry.insert(0, rel)

    def save(self):
        values = {k: self.entries[k].get().strip() for k in self.entries}
        try:
            self._update_agent_config(values)
            self._update_overlay_config(values)
            messagebox.showinfo("Settings", "Configuration updated successfully.")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update settings: {e}")

    def _update_agent_config(self, values):
        if not os.path.exists(AGENT_CFG_PATH):
            return
        with open(AGENT_CFG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        overlay = cfg.get('overlay', {})
        overlay['python'] = values['python']
        overlay['script'] = values['overlay_script']
        overlay['cwd'] = os.path.dirname(values['overlay_script'])
        cfg['overlay'] = overlay
        with open(AGENT_CFG_PATH, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

    def _update_overlay_config(self, values):
        if not os.path.exists(OVERLAY_CFG_PATH):
            return
        with open(OVERLAY_CFG_PATH, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        tools = cfg.get('tools', {})
        ocr = tools.get('ocr.start', {})
        ocr['command'] = values['python']
        args = ocr.get('args', [values['ocr_script']])
        if args:
            args[0] = values['ocr_script']
        else:
            args = [values['ocr_script']]
        ocr['args'] = args
        ocr['cwd'] = os.path.dirname(values['ocr_script'])
        tools['ocr.start'] = ocr

        stt = tools.get('stt.start', {})
        stt['command'] = values['python']
        sargs = stt.get('args', [values['stt_script']])
        if sargs:
            sargs[0] = values['stt_script']
        else:
            sargs = [values['stt_script']]
        stt['args'] = sargs
        stt['cwd'] = os.path.dirname(values['stt_script'])
        tools['stt.start'] = stt
        cfg['tools'] = tools
        with open(OVERLAY_CFG_PATH, 'w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

if __name__ == '__main__':
    SettingsGUI().mainloop()
