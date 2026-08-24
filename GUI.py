import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import threading
import os
import queue
import signal
import re
import sys



# =========================================================
# Default training parameters
# =========================================================
DEFAULT_PARAMS = {
    "device": 0,
    "base_latent": 128,
    "embed_dim": 256,
    "fusion_block_num": 1,
    "layer_num_m": 2,
    "layer_num_p": 1,
    "recon_w": 0.6,
    "recon_choice": "model",
    "loss": "adds",
    "start_epoch": 0,
    "lr": 1e-5,
    "min_lr": 1e-6,
    "lr_rate": 0.3,
    "decay_margin": 0.033,
    "decay_rate": 0.82,
    "warm_epoch": 1,
    "filter_enhance": True,
}


class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DTTD model train")
        self.root.geometry("1000x800")

        self.training_process = None
        self.training_thread = None
        self.output_queue = queue.Queue()
        self.latest_output_dir = None

        self.create_widgets()
        self.root.after(50, self.process_output_queue)

        self.python_path = sys.executable

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.preprocess_page = ttk.Frame(self.notebook)
        self.training_page = ttk.Frame(self.notebook)

        self.notebook.add(self.preprocess_page, text="Preprocess")
        self.notebook.add(self.training_page, text="Training")

        self.create_preprocess_page()
        self.create_training_page()
        self.create_console()

    def create_preprocess_page(self):
        frame = ttk.LabelFrame(self.preprocess_page, text="preprocess params", padding=15)
        frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(frame, text="BOP_DATA_PATH:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.bop_data_var = tk.StringVar(value="/home")
        bop_entry = ttk.Entry(frame, textvariable=self.bop_data_var, width=40)
        bop_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_bop_data).grid(row=0, column=2, padx=5)

        ttk.Label(frame, text="OBJECTS_SOURCE_PATH:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.objects_source_var = tk.StringVar(value="/home")
        obj_entry = ttk.Entry(frame, textvariable=self.objects_source_var, width=40)
        obj_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame, text="Browse", command=self.browse_objects_source).grid(row=1, column=2, padx=5)

        ttk.Button(frame, text="Run Preprocess", command=self.start_preprocess).grid(
            row=2, column=0, columnspan=3, pady=10
        )

        ttk.Label(
            frame,
            text="Script uses its own defaults for all other parameters.",
            foreground="gray",
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=5)

    def create_training_page(self):
        params_frame = ttk.LabelFrame(self.training_page, text="training params", padding=15)
        params_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(params_frame, text="Dataset path:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar(value="/home")
        dataset_entry = ttk.Entry(params_frame, textvariable=self.dataset_var, width=40)
        dataset_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(params_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5)

        ttk.Label(params_frame, text="Batch:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.batch_var = tk.StringVar(value="4")
        batch_spinbox = ttk.Spinbox(params_frame, from_=1, to=64, textvariable=self.batch_var, width=10)
        batch_spinbox.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(params_frame, text="Epoch:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.epoch_var = tk.StringVar(value="100")
        epoch_spinbox = ttk.Spinbox(params_frame, from_=2, to=1000, textvariable=self.epoch_var, width=10)
        epoch_spinbox.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        info_text = "other params:\n"
        info_text += f"• Device: {DEFAULT_PARAMS['device']}\n"
        info_text += f"• Loss: {DEFAULT_PARAMS['loss']}\n"
        info_text += f"• LR: {DEFAULT_PARAMS['lr']}"

        ttk.Label(
            params_frame,
            text=info_text,
            justify=tk.LEFT,
            foreground="gray",
        ).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=10)

        button_frame = ttk.Frame(self.training_page)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_btn = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(button_frame, text="Clear Console", command=self.clear_output).pack(side=tk.LEFT, padx=5)

    def create_console(self):
        console_frame = ttk.LabelFrame(self.root, text="Console", padding=5)
        console_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.console_text = scrolledtext.ScrolledText(
            console_frame,
            height=18,
            width=120,
            font=("Courier", 9),
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)
        self.console_text.config(state=tk.DISABLED)

    def browse_bop_data(self):
        folder = filedialog.askdirectory(title="Select BOP Data Folder")
        if folder:
            self.bop_data_var.set(folder)

    def browse_objects_source(self):
        folder = filedialog.askdirectory(title="Select Objects Source Folder")
        if folder:
            self.objects_source_var.set(folder)

    def browse_dataset(self):
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.dataset_var.set(folder)

    def append_output(self, text):
        self.output_queue.put(text)

    def append_preprocess_output(self, text):
        self.output_queue.put(text)

    def process_output_queue(self):
        lines = []
        try:
            while True:
                lines.append(self.output_queue.get_nowait())
        except queue.Empty:
            pass

        if lines:
            self.console_text.config(state=tk.NORMAL)
            for line in lines:
                self.console_text.insert(tk.END, line + "\n")
            self.console_text.see(tk.END)
            self.console_text.config(state=tk.DISABLED)

        self.root.after(50, self.process_output_queue)

    def clear_output(self):
        self.console_text.config(state=tk.NORMAL)
        self.console_text.delete(1.0, tk.END)
        self.console_text.config(state=tk.DISABLED)

    def build_command(self):
        dataset_root = self.dataset_var.get().strip()
        optim_batch = self.batch_var.get().strip()
        nepoch = self.epoch_var.get().strip()
        dataset_config = os.path.join(dataset_root, "dataset_config")

        if not dataset_root:
            raise ValueError("Please enter Dataset path")

        if not os.path.exists(dataset_root):
            raise ValueError(f"Dataset path not exist: {dataset_root}")

        cmd = [
            self.python_path, "train.py",
            "--device", str(DEFAULT_PARAMS["device"]),
            "--dataset_root", dataset_root,
            "--dataset_config", dataset_config,
            "--output_dir", "./result/exp",
            "--base_latent", str(DEFAULT_PARAMS["base_latent"]),
            "--embed_dim", str(DEFAULT_PARAMS["embed_dim"]),
            "--fusion_block_num", str(DEFAULT_PARAMS["fusion_block_num"]),
            "--layer_num_m", str(DEFAULT_PARAMS["layer_num_m"]),
            "--layer_num_p", str(DEFAULT_PARAMS["layer_num_p"]),
            "--recon_w", str(DEFAULT_PARAMS["recon_w"]),
            "--recon_choice", DEFAULT_PARAMS["recon_choice"],
            "--loss", DEFAULT_PARAMS["loss"],
            "--optim_batch", optim_batch,
            "--start_epoch", str(DEFAULT_PARAMS["start_epoch"]),
            "--lr", str(DEFAULT_PARAMS["lr"]),
            "--min_lr", str(DEFAULT_PARAMS["min_lr"]),
            "--lr_rate", str(DEFAULT_PARAMS["lr_rate"]),
            "--decay_margin", str(DEFAULT_PARAMS["decay_margin"]),
            "--decay_rate", str(DEFAULT_PARAMS["decay_rate"]),
            "--nepoch", nepoch,
            "--warm_epoch", str(DEFAULT_PARAMS["warm_epoch"]),
        ]

        if DEFAULT_PARAMS["filter_enhance"]:
            cmd.append("--filter_enhance")

        return cmd

    def build_eval_command(self, best_model_path):
        dataset_root = self.dataset_var.get().strip()

        cmd = [
            self.python_path, "eval.py",
            "--dataset_root", dataset_root,
            "--model", best_model_path,
            "--base_latent", "128",
            "--embed_dim", "256",
            "--fusion_block_num", "1",
            "--layer_num_m", "2",
            "--layer_num_p", "1",
            "--output", "eval_results",
            "--filter",
        ]
        return cmd

    def build_preprocess_command(self):
        script_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preprocess")
        script_path = os.path.join(script_dir, "preprocess.sh")

        if not os.path.exists(script_path):
            raise ValueError(f"Script not found: {script_path}")

        bop_data_path = self.bop_data_var.get().strip()
        objects_source_path = self.objects_source_var.get().strip()

        if not bop_data_path:
            raise ValueError("Please enter BOP_DATA_PATH")

        if not objects_source_path:
            raise ValueError("Please enter OBJECTS_SOURCE_PATH")

        env = os.environ.copy()
        env["BOP_DATA_PATH"] = bop_data_path
        env["OBJECTS_SOURCE_PATH"] = objects_source_path

        return script_path, env

    def run_training(self):
        try:
            cmd = self.build_command()
            self.latest_output_dir = None

            self.append_output(f"[Start] Executing command: {' '.join(cmd)}")
            self.append_output(f"[Working Directory] {os.getcwd()}")

            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            for line in self.training_process.stdout:
                line = line.rstrip()
                if line:
                    self.append_output(line)

                    m = re.search(r"output directory:\s*(.+)$", line)
                    if m:
                        self.latest_output_dir = m.group(1).strip()

            self.training_process.wait()

            if self.training_process.returncode == 0:
                self.append_output("[Complete] Training process has finished")

                if self.latest_output_dir:
                    best_model_path = os.path.join(self.latest_output_dir, "checkpoints", "best.pth")

                    if os.path.exists(best_model_path):
                        self.append_output(f"[Eval] Found best checkpoint: {best_model_path}")
                        eval_cmd = self.build_eval_command(best_model_path)
                        self.append_output(f"[Eval] Executing command: {' '.join(eval_cmd)}")

                        eval_process = subprocess.Popen(
                            eval_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            start_new_session=True,
                        )

                        for line in eval_process.stdout:
                            line = line.rstrip()
                            if line:
                                self.append_output(line)

                        eval_process.wait()

                        if eval_process.returncode == 0:
                            self.append_output("[Complete] Evaluation process has finished")
                        else:
                            self.append_output("[Error] Evaluation failed")
                    else:
                        self.append_output(f"[Warn] best.pth not found: {best_model_path}")
                else:
                    self.append_output("[Warn] Could not detect training output directory, skip evaluation")
            else:
                self.append_output("[Error] Training failed, skip evaluation")

        except Exception as e:
            self.append_output(f"[Error] {str(e)}")

        finally:
            self.root.after(0, self.training_finished)

    def run_preprocess(self):
        try:
            script_path, env = self.build_preprocess_command()
            self.append_preprocess_output(f"[Start] Running: {script_path}")
            self.append_preprocess_output(f"[BOP_DATA_PATH] {self.bop_data_var.get().strip()}")
            self.append_preprocess_output(f"[OBJECTS_SOURCE_PATH] {self.objects_source_var.get().strip()}")

            process = subprocess.Popen(
                ["bash", script_path],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )

            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.append_preprocess_output(line)

            process.wait()

            if process.returncode == 0:
                self.append_preprocess_output("[Complete] Preprocess finished")
            else:
                self.append_preprocess_output("[Error] Preprocess failed")

        except Exception as e:
            self.append_preprocess_output(f"[Error] {str(e)}")
        finally:
            self.root.after(0, self.training_finished)

    def start_training(self):
        try:
            dataset_root = self.dataset_var.get().strip()
            if not dataset_root:
                messagebox.showerror("Error", "Please enter Dataset path")
                return

            if self.training_thread and self.training_thread.is_alive():
                return

            self.append_output("=" * 60)
            self.append_output("Training started...")
            self.append_output("=" * 60)

            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            self.training_thread = threading.Thread(target=self.run_training, daemon=True)
            self.training_thread.start()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.training_finished()

    def start_preprocess(self):
        try:
            if self.training_thread and self.training_thread.is_alive():
                return

            self.append_preprocess_output("=" * 60)
            self.append_preprocess_output("Preprocess started...")
            self.append_preprocess_output("=" * 60)

            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            self.training_thread = threading.Thread(target=self.run_preprocess, daemon=True)
            self.training_thread.start()

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.training_finished()

    def stop_training(self):
        if not self.training_process:
            return

        try:
            os.killpg(os.getpgid(self.training_process.pid), signal.SIGTERM)
            self.training_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.training_process.pid), signal.SIGKILL)
        except Exception as e:
            self.append_output(f"[Stop Error] {str(e)}")
        finally:
            self.training_finished()
            self.append_output("[Stop] Training has been interrupted by user")

    def training_finished(self):
        self.training_process = None
        self.training_thread = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    root = tk.Tk()
    gui = TrainingGUI(root)
    root.mainloop()