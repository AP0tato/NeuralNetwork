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
        self.root.geometry("950x400")
        
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
        
        # Prediction function
        predict_func = getattr(self.lib, 'nn_predict', None) or getattr(self.lib, '_nn_predict')
        predict_func.argtypes = [
            ctypes.POINTER(ctypes.c_float),    # input
            ctypes.POINTER(ctypes.c_float),    # output
            ctypes.c_uint,                      # input_size
            ctypes.c_uint                       # output_size
        ]
        predict_func.restype = ctypes.c_int
        self.lib.nn_predict = predict_func
        
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

    def _get_learning_rate(self):
        """Returns the current learning rate from the slider, clamped to [0, 1]."""
        try:
            value = float(self.learning_rate_var.get())
        except Exception:
            value = 0.1
        return max(0.0, min(1.0, value))

    def _apply_learning_rate(self, value=None):
        """Push the selected learning rate into the backend and update the readout."""
        if value is None:
            value = self._get_learning_rate()

        if hasattr(self, 'learning_rate_value_label'):
            self.learning_rate_value_label.config(text=f"{value:.2f}")

        if self.lib:
            try:
                self.lib.nn_set_learning_rate(ctypes.c_float(value))
            except Exception:
                pass

        return value

    def _on_learning_rate_change(self, _value=None):
        self._apply_learning_rate()

    def _create_new_model(self):
        """Prompts for new model then refreshes dropdown."""
        # ... (Your existing simpledialog logic here) ...
        # After successfully writing to JSON:
        self._refresh_dropdown()
    
    def _create_widgets(self):
        """GUI with a centered dropdown for model selection."""
        # --- TOP CONTROL PANEL ---
        self.learning_rate_var = tk.DoubleVar(value=0.10)

        button_container = tk.Frame(self.root)
        button_container.pack(pady=10, fill='x')

        inner_button_frame = tk.Frame(button_container)
        inner_button_frame.pack(anchor='center')

        tk.Button(inner_button_frame, text="New Model", command=self._create_new_model, width=12).pack(side='left', padx=5)
        tk.Button(inner_button_frame, text="Edit Model", command=self._edit_model, width=12, bg="lightblue").pack(side='left', padx=5)
        tk.Button(inner_button_frame, text="Select Dataset", command=self._select_dataset, width=12).pack(side='left', padx=5)
        tk.Button(inner_button_frame, text="Use/Test", command=self._open_test_window, width=12).pack(side='left', padx=5)
        tk.Button(inner_button_frame, text="Reset Weights", command=self._reset_weights, width=12, bg="orange").pack(side='left', padx=5)
        
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

        lr_frame = tk.Frame(self.root)
        lr_frame.pack(pady=(8, 2), fill='x')

        tk.Label(lr_frame, text="Learning Rate", fg="white", font=("Arial", 10, "bold")).pack()
        slider_row = tk.Frame(lr_frame)
        slider_row.pack(pady=2)

        self.learning_rate_slider = tk.Scale(
            slider_row,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient='horizontal',
            length=300,
            variable=self.learning_rate_var,
            command=self._on_learning_rate_change,
            showvalue=False,
        )
        self.learning_rate_slider.pack(side='left')

        self.learning_rate_value_label = tk.Label(slider_row, text=f"{self._get_learning_rate():.2f}", width=6, fg="cyan")
        self.learning_rate_value_label.pack(side='left', padx=(8, 0))
        
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
    
    def _reset_weights(self):
        """Delete the weights file for the current model to start fresh training."""
        if not hasattr(self, 'current_model_name'):
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        bin_path = self._get_data_directory() / f"{self.current_model_name}.bin"
        
        if not bin_path.exists():
            messagebox.showinfo("Info", f"No weights file found for '{self.current_model_name}'")
            return
        
        # Confirm deletion
        if messagebox.askyesno("Confirm", f"Delete weights for '{self.current_model_name}'?\n\nYou'll need to retrain after this."):
            try:
                bin_path.unlink()
                self._log(f"Deleted weights file: {bin_path.name}")
                messagebox.showinfo("Success", f"Weights deleted. Train a new model to get started!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete weights: {e}")
    
    def _edit_model(self):
        """Edit metadata for the current model (input size, hidden layers, output size)."""
        if not hasattr(self, 'current_model_name'):
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        meta = self.models_config[self.current_model_name]
        
        # Create edit window
        edit_window = tk.Toplevel(self.root)
        edit_window.title(f"Edit Model: {self.current_model_name}")
        edit_window.geometry("500x350")
        
        # Input size
        tk.Label(edit_window, text="Input Size:", font=("Arial", 10, "bold")).pack(pady=(10, 0))
        input_var = tk.StringVar(value=str(meta["input"]))
        tk.Entry(edit_window, textvariable=input_var, width=30).pack(pady=5)
        tk.Label(edit_window, text="(e.g., 784 for 28x28, 768 for 27.7x27.7)", fg="gray", font=("Arial", 8)).pack()
        
        # Hidden layers
        tk.Label(edit_window, text="Hidden Layers (comma-separated):", font=("Arial", 10, "bold")).pack(pady=(15, 0))
        hidden_str = ",".join(str(x) for x in meta["hidden"])
        hidden_var = tk.StringVar(value=hidden_str)
        tk.Entry(edit_window, textvariable=hidden_var, width=30).pack(pady=5)
        tk.Label(edit_window, text="(e.g., 128,64 for two layers with 128 and 64 nodes)", fg="gray", font=("Arial", 8)).pack()
        
        # Output size
        tk.Label(edit_window, text="Output Size:", font=("Arial", 10, "bold")).pack(pady=(15, 0))
        output_var = tk.StringVar(value=str(meta["output"]))
        tk.Entry(edit_window, textvariable=output_var, width=30).pack(pady=5)
        tk.Label(edit_window, text="(e.g., 10 for digit classification 0-9)", fg="gray", font=("Arial", 8)).pack()
        
        def save_changes():
            """Validate and save the edited metadata."""
            try:
                # Validate input
                input_size = int(input_var.get())
                if input_size <= 0:
                    messagebox.showerror("Error", "Input size must be positive")
                    return
                
                hidden_list = [int(x.strip()) for x in hidden_var.get().split(",")]
                if any(h <= 0 for h in hidden_list):
                    messagebox.showerror("Error", "All hidden layer sizes must be positive")
                    return
                
                output_size = int(output_var.get())
                if output_size <= 0:
                    messagebox.showerror("Error", "Output size must be positive")
                    return
                
                # Update configuration
                path = self._get_json_path()
                with open(path, 'r') as f:
                    data = json.load(f)
                
                data[self.current_model_name] = {
                    "input": input_size,
                    "hidden": hidden_list,
                    "output": output_size
                }
                
                with open(path, 'w') as f:
                    json.dump(data, f, indent=4)
                # Update in-memory config and refresh dropdown without prompting
                self.models_config = data
                self._refresh_dropdown()
                # Keep the edited model selected
                try:
                    self._on_model_select(self.current_model_name)
                except Exception:
                    pass
                
                self._log(f"Updated '{self.current_model_name}' metadata: input={input_size}, hidden={hidden_list}, output={output_size}")
                messagebox.showinfo("Success", f"Model metadata updated!\n\nNote: You'll need to retrain the model with the new architecture.")
                edit_window.destroy()
                
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save changes: {e}")
        
        # Buttons
        button_frame = tk.Frame(edit_window)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="Save Changes", command=save_changes, bg="lightgreen", width=12).pack(side='left', padx=5)
        tk.Button(button_frame, text="Cancel", command=edit_window.destroy, width=12).pack(side='left', padx=5)
    
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

            def _choose_input_shape(size):
                """Pick a 2D shape close to square for a flat input size."""
                root = int(size ** 0.5)
                for height in range(root, 0, -1):
                    if size % height == 0:
                        width = size // height
                        return height, width
                return size, 1

            def _resize_to_model_input(image, out_height, out_width):
                """Nearest-neighbor resize from the dataset image to the model input shape."""
                in_height, in_width = image.shape
                row_idx = (np.linspace(0, in_height - 1, out_height)).astype(np.int32)
                col_idx = (np.linspace(0, in_width - 1, out_width)).astype(np.int32)
                return image[np.ix_(row_idx, col_idx)]
            
            # Create hidden layers array for C++
            hidden_layers = (ctypes.c_uint * len(hidden_list))(*hidden_list)
            init_res = self.lib.nn_create_network(input_size, hidden_layers, len(hidden_list), output_size)
            
            if init_res != 0:
                self._log("Worker Error: Failed to initialize C++ Network.")
                return

            self._apply_learning_rate()
            
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
            
            def fast_augment(images_batch, input_size):
                """Simple translation (shift) using only NumPy."""
                # CALCULATE BATCH SIZE DYNAMICALLY
                num_images = images_batch.shape[0]
                
                # RESHAPE TO: (Batch Size, Height, Width)
                # This turns [24448] into [32, 28, 28]
                batch_2d = images_batch.reshape(num_images, 28, 28)
                
                augmented = np.zeros_like(batch_2d)
                
                for i in range(num_images):
                    # Random shift between -2 and 2 pixels
                    dy, dx = np.random.randint(-2, 3, 2)
                    
                    # Shift the 2D image
                    shifted = np.roll(batch_2d[i], shift=(dy, dx), axis=(0, 1))
                    
                    # Clean up the edges so they don't 'wrap'
                    if dy > 0: shifted[:dy, :] = 0
                    elif dy < 0: shifted[dy:, :] = 0
                    if dx > 0: shifted[:, :dx] = 0
                    elif dx < 0: shifted[:, dx:] = 0
                    
                    augmented[i] = shifted
                    
                # Flatten back to (Batch Size, 784) for the C++ backend
                return augmented.reshape(num_images, input_size)

            if len(images.shape) == 3:
                flat_input_size = images.shape[1] * images.shape[2]
                if flat_input_size != input_size:
                    target_height, target_width = _choose_input_shape(input_size)
                    resized_images = np.empty((images.shape[0], input_size), dtype=np.float32)
                    for index, image in enumerate(images):
                        resized_images[index] = _resize_to_model_input(image, target_height, target_width).flatten()
                    images = np.ascontiguousarray(resized_images)
                else:
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
                    
                    # Grab the original images
                    current_images = images[start_idx:end_idx]
                    
                    # APPLY AUGMENTATION HERE
                    # Only augment 50% of the time so the model still sees "perfect" digits too
                    if np.random.random() > 0.5:
                        current_images = fast_augment(current_images, input_size)
                    
                    # Get memory pointers for the batch
                    ptr_in = current_images.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    ptr_lab = labels_onehot[start_idx:end_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    
                    self.lib.nn_train_batch.argtypes = [
                        ctypes.POINTER(ctypes.c_float), # batch_inputs
                        ctypes.POINTER(ctypes.c_float), # batch_labels
                        ctypes.c_int,                  # batch_size
                        ctypes.c_int,                  # input_size
                        ctypes.c_int                   # output_size
                    ]
                    self.lib.nn_train_batch(ptr_in, ptr_lab, batch_size, input_size, output_size)

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
    
    def _open_test_window(self):
        """Open a new window for testing the neural network with custom input."""
        if not hasattr(self, 'current_model_name'):
            messagebox.showwarning("Warning", "Please select a model first")
            return
        
        # 1. INITIALIZE BACKEND
        try:
            meta = self.models_config[self.current_model_name]
            input_size, hidden_list, output_size = meta["input"], meta["hidden"], meta["output"]
            hidden_layers = (ctypes.c_uint * len(hidden_list))(*hidden_list)
            self.lib.nn_create_network(input_size, hidden_layers, len(hidden_list), output_size)
            self._apply_learning_rate()
            
            bin_path = self._get_data_directory() / f"{self.current_model_name}.bin"
            if bin_path.exists():
                self.lib.nn_load_from_bin(str(bin_path.resolve()).encode('utf-8'))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize network: {e}")
            return
        
        # 2. WINDOW SETUP
        test_window = tk.Toplevel(self.root)
        test_window.title(f"Test & Train - {self.current_model_name}")
        test_window.geometry("1100x800")
        
        def _choose_input_shape(size):
            root = int(size ** 0.5)
            for h in range(root, 0, -1):
                if size % h == 0: return h, size // h
            return size, 1

        target_height, target_width = _choose_input_shape(input_size)
        
        # 3. LAYOUT CONTAINERS
        main_container = tk.Frame(test_window)
        main_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        left_panel = tk.Frame(main_container)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        right_panel = tk.Frame(main_container)
        right_panel.pack(side='right', fill='both', expand=True)

        # 4. DRAWING CANVAS (Left)
        canvas_size, pixel_size = 28, 15
        pixel_grid = np.zeros((canvas_size, canvas_size), dtype=np.float32)
        pixel_rects = {}

        canvas = tk.Canvas(left_panel, width=canvas_size*pixel_size, height=canvas_size*pixel_size, bg="black")
        canvas.pack()

        for i in range(canvas_size):
            for j in range(canvas_size):
                pixel_rects[(i, j)] = canvas.create_rectangle(j*pixel_size, i*pixel_size, (j+1)*pixel_size, (i+1)*pixel_size, fill="black", outline="#333333")

        def on_canvas_motion(event):
            gx, gy = event.x // pixel_size, event.y // pixel_size
            if 0 <= gx < canvas_size and 0 <= gy < canvas_size:
                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < canvas_size and 0 <= ny < canvas_size:
                            pixel_grid[ny, nx] = min(1.0, pixel_grid[ny, nx] + 0.4)
                            val = int(pixel_grid[ny, nx] * 255)
                            canvas.itemconfig(pixel_rects[(ny, nx)], fill=f"#{val:02x}{val:02x}{val:02x}")

        canvas.bind("<B1-Motion>", on_canvas_motion)

        # 5. PREDICTION LOGIC
        prediction_labels = []
        def test_drawing():
            if np.sum(pixel_grid) == 0: return
            resized = _resize_to_model_input(pixel_grid, target_height, target_width)
            input_data = np.ascontiguousarray(resized.flatten().astype(np.float32))
            output_data = np.ascontiguousarray(np.zeros(10, dtype=np.float32))
            
            self.lib.nn_predict(input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               input_size, 10)
            
            best = np.argmax(output_data)
            for i, lbl in enumerate(prediction_labels):
                conf = max(0, output_data[i] * 100)
                lbl.config(text=f"{i}: {conf:.1f}%", fg="green" if i == best else "black")

        # 6. MANUAL TRAINING LOGIC
        def manual_train():
            if np.sum(pixel_grid) == 0: return
            resized = _resize_to_model_input(pixel_grid, target_height, target_width)
            input_data = np.ascontiguousarray(resized.flatten().astype(np.float32))
            label_data = np.ascontiguousarray(np.zeros(10, dtype=np.float32))
            label_data[correct_digit_var.get()] = 1.0
            
            self.lib.nn_train_batch(input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   label_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                   1, input_size, 10)
            
            self.lib.nn_save_to_bin(str(bin_path.resolve()).encode('utf-8'))
            self._log(f"Manual Train: {correct_digit_var.get()} saved.")
            test_drawing()

        # 7. UI ELEMENTS (Buttons & Labels)
        btn_frame = tk.Frame(left_panel)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Clear", command=lambda: [pixel_grid.fill(0), [canvas.itemconfig(r, fill="black") for r in pixel_rects.values()]]).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Predict", command=test_drawing, bg="lightgreen").pack(side='left', padx=5)

        # Right Panel UI
        pred_frame = tk.LabelFrame(right_panel, text="Predictions")
        pred_frame.pack(fill='both', expand=True, pady=5)
        for i in range(10):
            lbl = tk.Label(pred_frame, text=f"{i}: --", font=("Arial", 12))
            lbl.pack()
            prediction_labels.append(lbl)

        train_frame = tk.LabelFrame(right_panel, text="Correct & Train")
        train_frame.pack(fill='x', pady=5)
        correct_digit_var = tk.IntVar(value=0)
        tk.OptionMenu(train_frame, correct_digit_var, *range(10)).pack(side='left', padx=5)
        tk.Button(train_frame, text="TRAIN ON THIS", command=manual_train, bg="orange").pack(side='left', padx=5)

        def _resize_to_model_input(img, oh, ow):
            ih, iw = img.shape
            ridx = (np.linspace(0, ih-1, oh)).astype(np.int32)
            cidx = (np.linspace(0, iw-1, ow)).astype(np.int32)
            return img[np.ix_(ridx, cidx)]

        test_window.protocol("WM_DELETE_WINDOW", lambda: [self.lib.nn_destroy_network(), test_window.destroy()])
    
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
