#!/usr/bin/env python3
"""
Improved Tkinter GUI frontend for the barpath pipeline.

Features:
- Modern, clean interface with better styling
- Select model directory and model file
- Select input video file
- Configure output settings
- Real-time log output with progress
- Cancel support
- Better error handling
"""
from __future__ import annotations

import os
import sys
import subprocess
import threading
import queue
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog, messagebox
except Exception as e:
    print("tkinter is required to run the GUI. On Debian/Ubuntu install: sudo apt install python3-tk")
    raise


class BarpathTkGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Barpath - Weightlifting Analysis")
        
        # Initialize process tracking
        self.proc = None
        self.proc_thread = None
        self.stop_event = threading.Event()
        self.log_queue: queue.Queue[str] = queue.Queue()
        
        # Variables
        self.model_dir = tk.StringVar()
        self.model_file = tk.StringVar()
        self.video_file = tk.StringVar()
        self.output_video = tk.StringVar(value="output.mp4")
        self.lift_type = tk.StringVar(value="none")
        self.class_name = tk.StringVar(value="endcap")
        self.graphs_dir = tk.StringVar(value="graphs")
        
        # Model paths storage
        self._model_full_paths = []
        
        # Configure styling
        self._configure_styles()
        
        # Create UI
        self._create_widgets()
        self._layout()
        self._populate_default_model_dir()
        self._start_log_pump()
    
    def _configure_styles(self):
        """Configure ttk styles for better appearance"""
        style = ttk.Style()
        
        # Use a modern theme
        available_themes = style.theme_names()
        if 'clam' in available_themes:
            style.theme_use('clam')
        elif 'alt' in available_themes:
            style.theme_use('alt')
        
        # Color scheme
        alabaster = '#F8F8FA'      # Window Background (Off-White)
        white = '#FFFFFF'          # Section Backgrounds
        dark_charcoal = '#333333'  # Main Text
        light_gray = '#D1D1D1'     # Menu Border Line Color
        gold = '#FFD700'           # Button Background
        black = '#000000'          # Button Text
        gold_hover = '#E5C100'     # Darker gold for hover
        
        self.root.configure(bg=alabaster)
        
        # Button styles
        style.configure('TButton', 
                       padding=8, 
                       relief="flat", 
                       background=gold,
                       foreground=black,
                       borderwidth=1,
                       font=('Arial', 9, 'bold'))
        style.map('TButton',
                  background=[('active', gold_hover), ('pressed', '#CCB000')],
                  foreground=[('active', black)])
        
        # Label styles
        style.configure('TLabel', 
                       background=alabaster, 
                       foreground=dark_charcoal, 
                       padding=2,
                       font=('Arial', 9))
        style.configure('Header.TLabel', font=('Arial', 10, 'bold'))
        
        # Entry styles
        style.configure('TEntry', 
                       padding=5,
                       fieldbackground=white,
                       foreground=dark_charcoal,
                       borderwidth=1,
                       bordercolor=light_gray)
        
        # Combobox styles
        style.configure('TCombobox', 
                       padding=5,
                       fieldbackground=white,
                       foreground=dark_charcoal,
                       borderwidth=1,
                       bordercolor=light_gray)
        style.map('TCombobox',
                  fieldbackground=[('readonly', white)],
                  foreground=[('readonly', dark_charcoal)])
        
        # LabelFrame styles (sections with borders)
        style.configure('TLabelframe', 
                       background=white,
                       foreground=dark_charcoal,
                       borderwidth=1,
                       bordercolor=light_gray,
                       relief='solid')
        style.configure('TLabelframe.Label', 
                       background=white,
                       foreground=dark_charcoal,
                       font=('Arial', 10, 'bold'))
        
        # Frame styles
        style.configure('TFrame', background=alabaster)
        
        # Progressbar style
        style.configure('TProgressbar',
                       background=gold,
                       troughcolor=light_gray,
                       borderwidth=1,
                       thickness=20)
    
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        
        # === MODEL SECTION ===
        self.model_frame = ttk.LabelFrame(self.main_frame, text="Model Configuration", padding="10")
        # Set inner background to white
        self.model_frame.configure(style='TLabelframe')
        
        self.model_dir_label = ttk.Label(self.model_frame, text="Model Directory:", style='TLabel')
        self.model_dir_label.configure(background='#FFFFFF')
        self.model_dir_display = ttk.Entry(self.model_frame, textvariable=self.model_dir, 
                                           width=50, state='readonly')
        self.browse_model_btn = ttk.Button(self.model_frame, text="Browse...", 
                                           command=self.browse_model_dir, width=12)
        
        self.model_file_label = ttk.Label(self.model_frame, text="Model File:")
        self.model_file_label.configure(background='#FFFFFF')
        self.model_combo = ttk.Combobox(self.model_frame, textvariable=self.model_file, 
                                        state="readonly", width=47)
        
        self.class_name_label = ttk.Label(self.model_frame, text="Class Name:")
        self.class_name_label.configure(background='#FFFFFF')
        self.class_name_entry = ttk.Entry(self.model_frame, textvariable=self.class_name, width=20)
        
        # === VIDEO SECTION ===
        self.video_frame = ttk.LabelFrame(self.main_frame, text="Video Configuration", padding="10")
        
        self.video_label = ttk.Label(self.video_frame, text="Input Video:")
        self.video_label.configure(background='#FFFFFF')
        self.video_display = ttk.Entry(self.video_frame, textvariable=self.video_file, 
                                       width=50, state='readonly')
        self.browse_video_btn = ttk.Button(self.video_frame, text="Select Video...", 
                                           command=self.browse_video, width=12)
        
        self.output_label = ttk.Label(self.video_frame, text="Output Video:")
        self.output_label.configure(background='#FFFFFF')
        self.output_entry = ttk.Entry(self.video_frame, textvariable=self.output_video, width=50)
        
        self.graphs_label = ttk.Label(self.video_frame, text="Graphs Directory:")
        self.graphs_label.configure(background='#FFFFFF')
        self.graphs_entry = ttk.Entry(self.video_frame, textvariable=self.graphs_dir, width=50)
        
        # === ANALYSIS SECTION ===
        self.analysis_frame = ttk.LabelFrame(self.main_frame, text="Analysis Options", padding="10")
        
        self.lift_label = ttk.Label(self.analysis_frame, text="Lift Type:")
        self.lift_label.configure(background='#FFFFFF')
        self.lift_combo = ttk.Combobox(self.analysis_frame, textvariable=self.lift_type, 
                                       state="readonly", width=20)
        self.lift_combo['values'] = ("none", "clean", "snatch", "jerk")
        self.lift_combo.set("none")
        
        # === CONTROL BUTTONS ===
        self.control_frame = ttk.Frame(self.main_frame)
        
        self.analyze_btn = ttk.Button(self.control_frame, text="▶ Start Analysis", 
                                      command=self.start_analysis, width=20)
        self.cancel_btn = ttk.Button(self.control_frame, text="⏹ Cancel", 
                                     command=self.cancel_analysis, width=20)
        self.cancel_btn.state(['disabled'])
        
        # === LOG SECTION ===
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Analysis Log", padding="5")
        
        self.log_text = tk.Text(self.log_frame, height=15, wrap="word", 
                               bg='#FFFFFF', fg='#333333', 
                               font=('Consolas', 9), relief='solid',
                               borderwidth=1,
                               insertbackground='#FFD700')  # Gold cursor
        self.log_text.configure(state="disabled")
        
        self.log_scroll_y = ttk.Scrollbar(self.log_frame, orient="vertical", 
                                         command=self.log_text.yview)
        self.log_scroll_x = ttk.Scrollbar(self.log_frame, orient="horizontal", 
                                         command=self.log_text.xview)
        self.log_text['yscrollcommand'] = self.log_scroll_y.set
        self.log_text['xscrollcommand'] = self.log_scroll_x.set
        
        # Progress bar
        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(self.log_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
    
    def _layout(self):
        """Layout all widgets"""
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure root grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(3, weight=1)  # Log section expands
        self.main_frame.grid_columnconfigure(0, weight=1)
        
        # === MODEL SECTION LAYOUT ===
        self.model_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.model_frame.grid_columnconfigure(1, weight=1)
        
        self.model_dir_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.model_dir_display.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.browse_model_btn.grid(row=0, column=2, sticky="e")
        
        self.model_file_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.model_combo.grid(row=1, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))
        
        self.class_name_label.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.class_name_entry.grid(row=2, column=1, sticky="w", pady=(10, 0))
        
        # === VIDEO SECTION LAYOUT ===
        self.video_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.video_frame.grid_columnconfigure(1, weight=1)
        
        self.video_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.video_display.grid(row=0, column=1, sticky="ew", padx=(0, 10))
        self.browse_video_btn.grid(row=0, column=2, sticky="e")
        
        self.output_label.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.output_entry.grid(row=1, column=1, sticky="ew", columnspan=2, pady=(10, 0))
        
        self.graphs_label.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        self.graphs_entry.grid(row=2, column=1, sticky="ew", columnspan=2, pady=(10, 0))
        
        # === ANALYSIS SECTION LAYOUT ===
        self.analysis_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        
        self.lift_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.lift_combo.grid(row=0, column=1, sticky="w")
        
        # === CONTROL BUTTONS LAYOUT ===
        self.control_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)
        
        self.analyze_btn.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.cancel_btn.grid(row=0, column=1, padx=(5, 0), sticky="ew")
        
        # === LOG SECTION LAYOUT ===
        self.log_frame.grid(row=4, column=0, sticky="nsew")
        self.log_frame.grid_rowconfigure(1, weight=1)
        self.log_frame.grid_columnconfigure(0, weight=1)
        
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5))
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_scroll_y.grid(row=1, column=1, sticky="ns")
        self.log_scroll_x.grid(row=2, column=0, sticky="ew")
    
    def _populate_default_model_dir(self):
        """Try to find and set a default model directory"""
        candidates = [
            Path(__file__).resolve().parent.parent.parent / 'models',  # repo_root/models
            Path(__file__).resolve().parent.parent / 'models',  # sibling models
            Path(__file__).resolve().parent / 'models',  # local models
        ]
        
        chosen = None
        for cand in candidates:
            if cand.exists() and cand.is_dir():
                chosen = cand
                break
        
        if chosen is not None:
            self.model_dir.set(str(chosen))
            self._populate_model_files(str(chosen))
            self.append_log(f"✓ Found models directory: {chosen}")
        else:
            self.model_dir.set('')
            self.model_combo['values'] = ()
            self.append_log("⚠ No models directory found. Please select one.")
    
    def _populate_model_files(self, dirpath: str):
        """Populate model file dropdown with .pt files from directory"""
        p = Path(dirpath)
        pt_files = sorted([x for x in p.glob('*.pt')])
        
        if pt_files:
            display_names = [x.name for x in pt_files]
            self.model_combo['values'] = display_names
            self._model_full_paths = [str(x) for x in pt_files]
            self.model_combo.current(0)
            self.model_file.set(display_names[0])
            self.append_log(f"✓ Found {len(pt_files)} model file(s)")
        else:
            self.model_combo['values'] = ()
            self._model_full_paths = []
            self.model_file.set('')
            self.append_log("⚠ No .pt model files found in directory")
    
    def browse_model_dir(self):
        """Open directory browser for model selection"""
        d = filedialog.askdirectory(title='Select Model Directory')
        if d:
            self.model_dir.set(d)
            self._populate_model_files(d)
    
    def browse_video(self):
        """Open file browser for video selection"""
        path = filedialog.askopenfilename(
            title='Select Input Video',
            filetypes=[
                ('Video Files', '*.mp4 *.avi *.mov *.MOV *.mkv *.webm'),
                ('All files', '*.*')
            ]
        )
        if path:
            self.video_file.set(path)
            self.append_log(f"✓ Selected video: {Path(path).name}")
    
    def append_log(self, text: str):
        """Append text to log (called from main thread only)"""
        self.log_text.configure(state="normal")
        
        # Strip ANSI escape codes (used by Rich)
        import re
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        text_clean = ansi_escape.sub('', text)
        
        # Handle Rich progress bars (carriage return updates)
        if '\r' in text_clean:
            # Split by carriage return and only keep the last part
            parts = text_clean.split('\r')
            text_clean = parts[-1]
            
            # Delete the last line if it exists (to update in place)
            last_line_start = self.log_text.index("end-1c linestart")
            last_line_end = self.log_text.index("end-1c lineend")
            last_line_content = self.log_text.get(last_line_start, last_line_end)
            
            # If the last line looks like a progress bar (contains Rich characters), delete it
            if last_line_content.strip() and any(char in last_line_content for char in ['█', '━', '╸', '%']):
                self.log_text.delete(last_line_start, last_line_end)
        
        # Skip empty lines from progress bar updates
        if not text_clean.strip():
            self.log_text.configure(state="disabled")
            return
        
        # Add color coding for special messages
        start_pos = self.log_text.index("end-1c")
        self.log_text.insert("end", text_clean + "\n")
        end_pos = self.log_text.index("end-1c")
        
        # Color code based on content (colors adjusted for light background)
        text_lower = text_clean.lower()
        if any(x in text_clean for x in ['✓', 'success', 'complete']):
            self.log_text.tag_add("success", start_pos, end_pos)
            self.log_text.tag_config("success", foreground='#2E7D32')  # Dark green
        elif any(x in text_clean for x in ['❌', 'error', 'failed']):
            self.log_text.tag_add("error", start_pos, end_pos)
            self.log_text.tag_config("error", foreground='#C62828')  # Dark red
        elif any(x in text_clean for x in ['⚠', 'warning', 'cancelled']):
            self.log_text.tag_add("warning", start_pos, end_pos)
            self.log_text.tag_config("warning", foreground='#F57C00')  # Dark orange
        elif '===' in text_clean or '---' in text_clean:
            self.log_text.tag_add("separator", start_pos, end_pos)
            self.log_text.tag_config("separator", foreground='#B8860B', font=('Consolas', 9, 'bold'))  # Dark goldenrod
        elif any(char in text_clean for char in ['█', '━', '╸', '%']):  # Rich progress bars
            self.log_text.tag_add("progress", start_pos, end_pos)
            self.log_text.tag_config("progress", foreground='#B8860B')  # Dark goldenrod
        
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
    
    def _start_log_pump(self):
        """Periodically check queue and update log"""
        def pump():
            try:
                while True:
                    line = self.log_queue.get_nowait()
                    self.append_log(line)
            except queue.Empty:
                pass
            self.root.after(200, pump)
        
        pump()
    
    def start_analysis(self):
        """Start the analysis pipeline"""
        # Validation
        if not self.video_file.get():
            messagebox.showerror('Error', 'Please select an input video file')
            return
        
        if not self._model_full_paths:
            messagebox.showerror('Error', 'Please select a model directory with .pt files')
            return
        
        sel_idx = self.model_combo.current()
        if sel_idx < 0:
            messagebox.showerror('Error', 'Please select a model file')
            return
        
        model_path = self._model_full_paths[sel_idx]
        
        # Find CLI script
        repo_root = Path(__file__).resolve().parent.parent.parent
        cli_script = repo_root / 'barpath' / 'cli' / 'barpath_cli.py'
        
        if not cli_script.exists():
            messagebox.showerror('Error', f'CLI script not found at: {cli_script}')
            return
        
        # Build command
        cmd = [
            sys.executable,
            str(cli_script),
            '--input_video', str(self.video_file.get()),
            '--model', model_path,
            '--output_video', str(self.output_video.get()),
            '--lift_type', str(self.lift_type.get()),
            '--class_name', str(self.class_name.get())
        ]
        
        # Update UI
        self.analyze_btn.state(['disabled'])
        self.cancel_btn.state(['!disabled'])
        self.progress_var.set(0)
        
        # Clear log
        self.log_text.configure(state="normal")
        self.log_text.delete('1.0', 'end')
        self.log_text.configure(state="disabled")
        
        self.append_log("=" * 60)
        self.append_log("Starting Barpath Analysis Pipeline")
        self.append_log("=" * 60)
        self.append_log(f"Video: {Path(self.video_file.get()).name}")
        self.append_log(f"Model: {Path(model_path).name}")
        self.append_log(f"Lift Type: {self.lift_type.get()}")
        self.append_log("=" * 60)
        
        # Start subprocess in background thread
        self.stop_event.clear()
        
        def target():
            env = dict(os.environ)
            env['PYTHONUNBUFFERED'] = '1'
            # Force Rich to use simple output mode without live rendering
            env['TERM'] = 'dumb'
            
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env
                )
            except Exception as e:
                self.log_queue.put(f"❌ Failed to start pipeline: {e}")
                self._on_process_finished(1)
                return
            
            # Stream output
            try:
                if self.proc.stdout is not None:
                    for line in self.proc.stdout:
                        if line:
                            # Send the line as-is, including any \r characters
                            # The append_log method will handle ANSI codes and carriage returns
                            self.log_queue.put(line.rstrip('\n'))
                        if self.stop_event.is_set():
                            break
            except Exception as e:
                self.log_queue.put(f"❌ Error reading output: {e}")
            
            # Wait for process
            if self.proc.poll() is None and not self.stop_event.is_set():
                try:
                    self.proc.wait()
                except Exception:
                    pass
            
            rc = self.proc.returncode if self.proc is not None else 1
            self._on_process_finished(rc)
        
        self.proc_thread = threading.Thread(target=target, daemon=True)
        self.proc_thread.start()
    
    def cancel_analysis(self):
        """Cancel the running analysis"""
        self.log_queue.put('\n⚠ Cancelling analysis...')
        self.stop_event.set()
        
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
    
    def _on_process_finished(self, returncode: int):
        """Called when subprocess finishes"""
        def ui_update():
            if returncode == 0:
                self.log_queue.put('\n' + "=" * 60)
                self.log_queue.put('✓ Pipeline completed successfully!')
                self.log_queue.put("=" * 60)
                self.progress_var.set(100)
                messagebox.showinfo('Success', 'Analysis completed successfully!')
            else:
                self.log_queue.put('\n' + "=" * 60)
                self.log_queue.put(f'❌ Pipeline failed with exit code {returncode}')
                self.log_queue.put("=" * 60)
                self.progress_var.set(0)
            
            # Re-enable buttons
            self.analyze_btn.state(['!disabled'])
            self.cancel_btn.state(['disabled'])
        
        self.root.after(50, ui_update)


def main():
    """Main entry point"""
    root = tk.Tk()
    app = BarpathTkGUI(root)
    
    # Set window size and center it
    window_width = 900
    window_height = 800  # Increased to 800
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')
    
    # Set minimum size
    root.minsize(800, 700)  # Increased minimum height
    
    root.mainloop()


if __name__ == '__main__':
    main()