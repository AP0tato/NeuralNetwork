#!/usr/bin/env python3
"""Neural Network Training with GPU acceleration via Metal."""

import json
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import threading
import numpy as np
from pathlib import Path
import ctypes
import time
import gc

class NeuralNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Trainer")
        self.root.geometry("600x400")
        
        self.selected_dataset_path = None
        self.lib = None
        self.network_ptr = None
        
        # Load the GPU library
        try:
            lib_path = Path("build/libneural_backend.dylib")
            if not lib_path.exists():
                messagebox.showerror("Error", "GPU library not found. Build the project first.")
                return
            self.lib = ctypes.CDLL(str(lib_path))
            self._setup_ctypes()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load GPU library: {e}")
            return
        
        # Create UI
        self._create_widgets()
    
    def _setup_ctypes(self):
        """Setup ctypes function signatures."""
        # On macOS, symbol names may have underscore prefix
        # Network creation
        create_func = getattr(self.lib, 'nn_create_network', None) or getattr(self.lib, '_nn_create_network')
        create_func.argtypes = [
            ctypes.c_uint,                      # input_size
            ctypes.POINTER(ctypes.c_uint),     # hidden_sizes
            ctypes.c_uint,                      # num_hidden
            ctypes.c_uint                       # output_size
        ]
        create_func.restype = ctypes.c_int
        self.lib.nn_create_network = create_func
        
        # Training
        train_func = getattr(self.lib, 'nn_train_batch', None) or getattr(self.lib, '_nn_train_batch')
        train_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),    # batch_inputs
            ctypes.POINTER(ctypes.c_float),    # batch_labels
            ctypes.c_uint,                      # batch_size
            ctypes.c_uint,                      # input_size
            ctypes.c_uint                       # output_size
        ]
        train_func.restype = ctypes.c_int
        self.lib.nn_train_batch = train_func
        
        # Learning rate
        lr_func = getattr(self.lib, 'nn_set_learning_rate', None) or getattr(self.lib, '_nn_set_learning_rate')
        lr_func.argtypes = [ctypes.c_float]
        lr_func.restype = ctypes.c_int
        self.lib.nn_set_learning_rate = lr_func
        
        self.lib.nn_load_from_bin.argtypes = [ctypes.c_char_p]
        self.lib.nn_load_from_bin.restype = ctypes.c_int
        
        get_lr_func = getattr(self.lib, 'nn_get_learning_rate', None) or getattr(self.lib, '_nn_get_learning_rate')
        get_lr_func.argtypes = []
        get_lr_func.restype = ctypes.c_float
        self.lib.nn_get_learning_rate = get_lr_func
        
        # Cleanup
        destroy_func = getattr(self.lib, 'nn_destroy_network', None) or getattr(self.lib, '_nn_destroy_network')
        destroy_func.argtypes = []
        destroy_func.restype = ctypes.c_int
        self.lib.nn_destroy_network = destroy_func
        self.lib.nn_save_to_bin.argtypes = [ctypes.c_char_p]
        self.lib.nn_save_to_bin.restype = ctypes.c_int
        
    def _refresh_dropdown(self):
        """Updates the dropdown list from models.json."""
        path = self._get_json_path()
        if path.exists():
            with open(path, 'r') as f:
                self.models_config = json.load(f)
            
            menu = self.model_dropdown["menu"]
            menu.delete(0, "end")
            for name in self.models_config.keys():
                menu.add_command(label=name, command=lambda n=name: self._on_model_select(n))
            
            # Auto-select first one if current is None
            if not hasattr(self, 'current_model_name') and self.models_config:
                first_model = list(self.models_config.keys())[0]
                self._on_model_select(first_model)

    def _on_model_select(self, name):
        """Triggered when a user picks a model from the dropdown."""
        self.current_model_name = name
        self.selected_model_var.set(name)
        meta = self.models_config[name]
        self.model_info_label.config(text=f"Model: {name} ({meta['input']}->{meta['hidden']}->{meta['output']})")
        self._log(f"Switched active model to: {name}")

    def _create_new_model(self):
        """Prompts for new model then refreshes dropdown."""
        # ... (Your existing simpledialog logic here) ...
        # After successfully writing to JSON:
        self._refresh_dropdown()
    
    def _create_widgets(self):
        """GUI with a centered dropdown for model selection."""
        # --- TOP CONTROL PANEL ---
        button_container = tk.Frame(self.root)
        button_container.pack(pady=10, fill='x')

        inner_button_frame = tk.Frame(button_container)
        inner_button_frame.pack(anchor='center')

        tk.Button(inner_button_frame, text="New Model", command=self._create_new_model, width=12).pack(side='left', padx=5)
        tk.Button(inner_button_frame, text="Select Dataset", command=self._select_dataset, width=12).pack(side='left', padx=5)
        
        # --- MODEL DROPDOWN ---
        self.model_options = ["None"]
        self.selected_model_var = tk.StringVar(value="Select Model")
        
        # This creates the dropdown menu
        self.model_dropdown = tk.OptionMenu(
            inner_button_frame, 
            self.selected_model_var, 
            *self.model_options, 
            command=self._on_model_select # Calls this when changed
        )
        self.model_dropdown.config(width=15)
        self.model_dropdown.pack(side='left', padx=5)

        # --- INFO LABELS ---
        self.model_info_label = tk.Label(self.root, text="No Architecture Loaded", font=("Arial", 10, "bold"), fg="cyan")
        self.model_info_label.pack(pady=5)
        
        self.dataset_label = tk.Label(self.root, text="No dataset selected", fg="white")
        self.dataset_label.pack()
        
        tk.Button(self.root, text="START TRAINING", command=self._start_training, 
                  fg="black", font=("Arial", 11, "bold"), width=20).pack(pady=10)
        
        self.status = tk.Label(self.root, text="Ready", fg="white", font=("Arial", 10))
        self.status.pack(pady=5)
        
        # --- LOGGING AREA ---
        log_frame = tk.Frame(self.root)
        log_frame.pack(pady=10, fill='both', expand=True, padx=10)

        self.results = tk.Text(log_frame, height=12, width=70, state='disabled', wrap='none', bg="#1e1e1e", fg="#d4d4d4")
        scrollbar = tk.Scrollbar(log_frame, command=self.results.yview)
        self.results.configure(yscrollcommand=scrollbar.set)

        self.results.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        self._refresh_dropdown() # Populate the dropdown on startup
    
    def _select_dataset(self):
        """Select MNIST dataset directory."""
        path = filedialog.askdirectory(title="Select MNIST dataset directory")
        if path:
            self.selected_dataset_path = Path(path)
            self.dataset_label.config(text=f"Dataset: {self.selected_dataset_path.name}")
    
    def _start_training(self):
        """Start training in background thread."""
        if not self.selected_dataset_path:
            messagebox.showwarning("Warning", "Please select a dataset first")
            return
        
        # Verify dataset files
        train_images = self.selected_dataset_path / "train_images.npy"
        train_labels = self.selected_dataset_path / "train_labels.npy"
        
        if not (train_images.exists() and train_labels.exists()):
            messagebox.showerror("Error", "Dataset must contain train_images.npy and train_labels.npy")
            return
        
        # Start training in background thread
        thread = threading.Thread(
            target=self._train_worker,
            args=(train_images, train_labels),
            daemon=True
        )
        thread.start()
        
    def _get_data_directory(self):
        """Returns the absolute path to the 'data' folder beside 'src'."""
        # Path(__file__).parent is 'src', .parent.parent is the project root
        data_dir = Path(__file__).parent.parent / "data"
        data_dir.mkdir(exist_ok=True)  # Create it if it doesn't exist
        return data_dir
        
    def _get_json_path(self):
        """Returns the path to models.json inside the data folder."""
        return self._get_data_directory() / "models.json"
        
    def _load_config(self):
        """Reads metadata and sets the active model configuration."""
        path = self._get_json_path()
        try:
            if not path.exists():
                # Create a default if it doesn't exist to avoid Errno 2
                initial_data = {"model1": {"input": 784, "hidden": [128, 64], "output": 10}}
                path.write_text(json.dumps(initial_data, indent=4))
            
            with open(path, 'r') as f:
                self.models_config = json.load(f)
            
            # Ask user which model to load
            names = list(self.models_config.keys())
            choice = simpledialog.askstring("Load Model", f"Enter model name:\n({', '.join(names)})")
            
            if choice in self.models_config:
                self.current_model_name = choice
                self.model_var.set(f"Current: {choice}")
                self._log(f"Active model set to: {choice}")
            else:
                messagebox.showwarning("Not Found", f"Model '{choice}' not in models.json")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load models.json: {e}")
            
    def _create_new_model(self):
        """Prompts for metadata and saves to data/models.json."""
        name = simpledialog.askstring("New Model", "Enter unique model name:")
        if not name: return
        
        try:
            # We define these clearly so they are available for the dictionary below
            val_in = simpledialog.askinteger("Input", "Input Size (e.g. 784):", initialvalue=784)
            hid_str = simpledialog.askstring("Hidden", "Hidden Layers (comma separated):", initialvalue="128,64")
            val_out = simpledialog.askinteger("Output", "Output Size (e.g. 10):", initialvalue=10)
            
            # Validation: if user cancels any dialog, simpledialog returns None
            if val_in is None or hid_str is None or val_out is None:
                return

            hidden_list = [int(x.strip()) for x in hid_str.split(",")]
            
            # 1. Load existing data from the data folder
            path = self._get_json_path()
            data = {}
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
            
            # 2. Add the new entry
            data[name] = {
                "input": val_in, 
                "hidden": hidden_list, 
                "output": val_out
            }
            
            # 3. Write back to data/models.json
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            
            self._log(f"Model '{name}' metadata saved to {path}")
            self._refresh_dropdown() # Update the UI dropdown menu
            messagebox.showinfo("Success", f"Model '{name}' created in data folder.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON: {e}")

    def _train_worker(self, train_images_path, train_labels_path):
        try:
            if not hasattr(self, 'current_model_name'):
                self._log("Worker Error: No model configuration loaded!")
                return
        
            meta = self.models_config[self.current_model_name]
            
            # Use dynamic values from JSON
            input_size = meta["input"]
            hidden_list = meta["hidden"]
            output_size = meta["output"]
            
            # Create hidden layers array for C++
            hidden_layers = (ctypes.c_uint * len(hidden_list))(*hidden_list)
            init_res = self.lib.nn_create_network(input_size, hidden_layers, len(hidden_list), output_size)
            
            if init_res != 0:
                self._log("Worker Error: Failed to initialize C++ Network.")
                return
            
            # Define path for .bin file inside the same data folder
            bin_path = self._get_data_directory() / f"{self.current_model_name}.bin"
            
            # Try to load existing weights
            if bin_path.exists():
                self._log(f"Loading weights from {bin_path.name}...")
                self.lib.nn_load_from_bin(str(bin_path.resolve()).encode('utf-8'))
            
            self._log("Loading dataset...")
        
            # 1. Load and immediately convert to C-contiguous to prevent silent copies
            images = np.ascontiguousarray(np.load(train_images_path).astype(np.float32) / 255.0)
            labels = np.load(train_labels_path).astype(np.uint8)

            if len(images.shape) == 3:
                images = images.reshape(images.shape[0], -1)

            # 2. One-hot encoding
            num_samples = len(labels)
            labels_onehot = np.zeros((num_samples, 10), dtype=np.float32)
            labels_onehot[np.arange(num_samples), labels] = 1.0
            labels_onehot = np.ascontiguousarray(labels_onehot)

            # 3. Delete the original labels array to save space
            del labels
            gc.collect()

            batch_size = 32
            num_epochs = 5
            num_batches = num_samples // batch_size

            self._log(f"Starting training on {num_samples} samples...")
            
            for epoch in range(num_epochs):
                epoch_start = time.time()
                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = start_idx + batch_size
                    
                    # Get memory pointers for the batch
                    ptr_in = images[start_idx:end_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    ptr_lab = labels_onehot[start_idx:end_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    
                    self.lib.nn_train_batch.argtypes = [
                        ctypes.POINTER(ctypes.c_float), # batch_inputs
                        ctypes.POINTER(ctypes.c_float), # batch_labels
                        ctypes.c_int,                  # batch_size
                        ctypes.c_int,                  # input_size
                        ctypes.c_int                   # output_size
                    ]
                    self.lib.nn_train_batch(ptr_in, ptr_lab, batch_size, 784, 10)

                    if batch_idx % 100 == 0:
                        self.status.config(text=f"Epoch {epoch+1} - Batch {batch_idx}/{num_batches}")
                
                self._log(f"Epoch {epoch+1} finished in {time.time() - epoch_start:.2f}s")
                
                self.lib.nn_clear_internal_buffers()
                
                gc.collect()

            self._log(f"Saving weights to {bin_path}...")
            res = self.lib.nn_save_to_bin(str(bin_path.resolve()).encode('utf-8'))
            
            if res == 0:
                self._log("Save successful.")
                self.status.config(text="Training Complete & Saved", fg="green")

        except Exception as e:
            self._log(f"Worker Error: {e}")
    
    def _log(self, message):
        """Write message to log display."""
        self.results.config(state='normal')
        self.results.insert('end', message + '\n')
        self.results.see('end')
        self.results.config(state='disabled')
        self.root.update()

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuralNetworkGUI(root)
    root.mainloop()
