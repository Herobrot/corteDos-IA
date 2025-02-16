import numpy as np # type: ignore
import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt # type: ignore
from matplotlib.figure import Figure # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
import threading

def min_max_scale(data: np.ndarray):
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scale = np.where(data_max - data_min == 0, 1, data_max - data_min)
    scaled = (data - data_min) / scale
    return scaled, {"min": data_min, "max": data_max}

def load_data(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    x_scaled, scaler_x = min_max_scale(X)
    y_scaled, scaler_y = min_max_scale(y.reshape(-1, 1))
    y_scaled = y_scaled.flatten()

    return df, x_scaled, y_scaled, scaler_x, scaler_y

class TrainingHistory:
    def __init__(self):
        self.records = []

    def add_record(self, epoch: int, weights: np.ndarray, bias: np.ndarray, loss: float):
        record = {'Época': epoch + 1}
        for i, w in enumerate(weights.flatten()):
            record[f'w{i}'] = w
        record['bias'] = bias[0] if bias.size > 0 else None
        record['Error'] = loss
        self.records.append(record)

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

class TrainingHistoryCallback(keras.callbacks.Callback):
    def __init__(self, history: TrainingHistory, tolerance: float, progress_callback=None):
        super().__init__()
        self.history = history
        self.tolerance = tolerance
        self.progress_callback = progress_callback

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        weights, bias = self.model.dense.get_weights()
        self.history.add_record(epoch, weights, bias, loss)
        if self.progress_callback:
            self.progress_callback(epoch + 1)
        print(f"Epoch {epoch + 1}: Loss = {loss}")
        if loss is not None and loss < self.tolerance:
            print(f"Tolerancia alcanzada: {loss} < {self.tolerance}. Deteniendo entrenamiento.")
            self.model.stop_training = True

class LinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    def call(self, inputs):
        return self.dense(inputs)

def train_model(x, y, lr: float, max_epochs: int, batch_size: int, tolerance: float, progress_callback=None):
    model = LinearRegression()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss=keras.losses.MeanSquaredError())

    history_tracker = TrainingHistory()
    callback = TrainingHistoryCallback(history_tracker, tolerance, progress_callback)

    initial_predictions = model.predict(x).flatten()
    model.fit(x, y, epochs=max_epochs, batch_size=batch_size, verbose=0, callbacks=[callback])
    final_predictions = model.predict(x).flatten()

    return model, history_tracker, initial_predictions, final_predictions, y

class NeuralNetworkGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Red Neuronal - Regresión Lineal (Validación Cruzada)")
        self.root.geometry("1100x650")
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.setup_ui()
        self.df = None
        self.X = None
        self.y = None
        self.scaler_x = None
        self.scaler_y = None
        self.model = None
        self.cv_report = None

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        sidebar = ttk.Frame(main_frame, padding=(10, 10))
        sidebar.grid(row=0, column=0, sticky="ns")

        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.btn_select_file = ttk.Button(sidebar, text="Seleccionar CSV", command=self.select_file)
        self.btn_select_file.grid(row=0, column=0, sticky="ew", pady=5)

        self.lbl_file = ttk.Label(sidebar, text="No se ha seleccionado archivo", wraplength=150)
        self.lbl_file.grid(row=1, column=0, sticky="ew", pady=5)
        
        param_frame = ttk.LabelFrame(sidebar, text="Parámetros de Entrenamiento", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", pady=5)

        ttk.Label(param_frame, text="Máx. Épocas:").grid(row=0, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="10000")
        self.entry_epochs = ttk.Entry(param_frame, textvariable=self.epochs_var, width=10)
        self.entry_epochs.grid(row=0, column=1, sticky="e", padx=5, pady=2)

        ttk.Label(param_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        self.lr_var = tk.StringVar(value="0.01")
        self.entry_lr = ttk.Entry(param_frame, textvariable=self.lr_var, width=10)
        self.entry_lr.grid(row=1, column=1, sticky="e", padx=5, pady=2)

        ttk.Label(param_frame, text="Batch Size:").grid(row=2, column=0, sticky="w")
        self.batch_var = tk.StringVar(value="32")
        self.entry_batch = ttk.Entry(param_frame, textvariable=self.batch_var, width=10)
        self.entry_batch.grid(row=2, column=1, sticky="e", padx=5, pady=2)

        ttk.Label(param_frame, text="Tolerancia:").grid(row=3, column=0, sticky="w")
        self.tol_var = tk.StringVar(value="0.00001")
        self.entry_tol = ttk.Entry(param_frame, textvariable=self.tol_var, width=10)
        self.entry_tol.grid(row=3, column=1, sticky="e", padx=5, pady=2)

        ttk.Label(param_frame, text="K Folds:").grid(row=4, column=0, sticky="w")
        self.kfolds_var = tk.StringVar(value="5")
        self.entry_kfolds = ttk.Entry(param_frame, textvariable=self.kfolds_var, width=10)
        self.entry_kfolds.grid(row=4, column=1, sticky="e", padx=5, pady=2)

        self.btn_train_model = ttk.Button(sidebar, text="Validación Cruzada", command=self.train_model_handler, state="disabled")
        self.btn_train_model.grid(row=5, column=0, sticky="ew", pady=10)
        
        self.progress = ttk.Progressbar(sidebar, orient="horizontal", mode="determinate")
        self.progress.grid(row=6, column=0, sticky="ew", pady=5)

        self.tab_data = ttk.Frame(notebook)
        notebook.add(self.tab_data, text="Vista de Datos")
        self.setup_tab_data(self.tab_data)

        self.tab_history = ttk.Frame(notebook)
        notebook.add(self.tab_history, text="Historial de Pesos")
        self.setup_tab_history(self.tab_history)

        self.tab_plot = ttk.Frame(notebook)
        notebook.add(self.tab_plot, text="Gráfica")
        self.setup_tab_plot(self.tab_plot)

        self.tab_error = ttk.Frame(notebook)
        notebook.add(self.tab_error, text="Evolución del Error")
        self.setup_tab_error(self.tab_error)

        self.tab_cv_report = ttk.Frame(notebook)
        notebook.add(self.tab_cv_report, text="Reporte CV")
        self.setup_tab_cv_report(self.tab_cv_report)

    def setup_tab_data(self, parent):
        self.tree_data = ttk.Treeview(parent, show="headings")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=vsb.set)
        self.tree_data.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def setup_tab_history(self, parent):
        self.tree_history = ttk.Treeview(parent, show="headings")
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self.tree_history.xview)
        self.tree_history.configure(xscrollcommand=hsb.set)
        self.tree_history.grid(row=0, column=0, sticky="nsew")
        hsb.grid(row=1, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def setup_tab_plot(self, parent):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def setup_tab_error(self, parent):
        self.fig_error = Figure(figsize=(5, 4), dpi=100)
        self.ax_error = self.fig_error.add_subplot(111)
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=parent)
        self.canvas_error.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def setup_tab_cv_report(self, parent):
        self.tree_cv_report = ttk.Treeview(parent, show="headings")
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self.tree_cv_report.xview)
        self.tree_cv_report.configure(xscrollcommand=hsb.set)
        self.tree_cv_report.grid(row=0, column=0, sticky="nsew")
        hsb.grid(row=1, column=0, sticky="ew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.lbl_file.config(text=file_path)
            self.df, self.X, self.y, self.scaler_x, self.scaler_y = load_data(file_path)
            self.populate_data_tree(self.df)
            self.btn_train_model.config(state="normal")

    def populate_data_tree(self, df: pd.DataFrame):
        for item in self.tree_data.get_children():
            self.tree_data.delete(item)
        self.tree_data["columns"] = list(df.columns)
        for col in df.columns:
            self.tree_data.heading(col, text=col)
            self.tree_data.column(col, width=100, anchor="center")
        for _, row in df.head(10).iterrows():
            self.tree_data.insert("", "end", values=list(row))

    def train_model_handler(self):
        try:
            max_epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            tolerance = float(self.tol_var.get())
            k_folds = int(self.kfolds_var.get())
        except Exception:
            messagebox.showerror("Error", "Verifica los parámetros de entrenamiento.")
            return

        self.progress.config(maximum=k_folds, value=0)
        self.btn_train_model.config(state="disabled")
        thread = threading.Thread(target=self.run_cross_validation, args=(self.X, self.y, lr, max_epochs, batch_size, tolerance, k_folds))
        thread.start()

    def run_cross_validation(self, X, y, lr, max_epochs, batch_size, tolerance, k_folds):
        n = len(X)
        indices = np.arange(n)
        rng = np.random.default_rng()
        rng.shuffle(indices)
        fold_sizes = np.full(k_folds, n // k_folds, dtype=int)
        fold_sizes[: n % k_folds] += 1

        current = 0
        models_info = []
        cv_report = []
        for i in range(k_folds):
            start = current
            end = current + fold_sizes[i]
            current = end
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            x_train, y_train = X[train_indices], y[train_indices]
            x_val, y_val = X[val_indices], y[val_indices]
            model, history_tracker, _, _, _ = train_model(x_train, y_train, lr, max_epochs, batch_size, tolerance)
            val_loss = model.evaluate(x_val, y_val, verbose=0)
            df_history = history_tracker.get_dataframe()

            weight_cols = [col for col in df_history.columns if col.startswith('w')]
            w0_list = [df_history.iloc[0][col] for col in weight_cols]
            wf_list = [df_history.iloc[-1][col] for col in weight_cols]
            w0_str = ", ".join(f"{float(val):.4f}" for val in w0_list)
            wf_str = ", ".join(f"{float(val):.4f}" for val in wf_list)
            error_t = df_history.iloc[-1]["Error"]
            error_v = val_loss
            total_error = (error_t + error_v) / 2.0
            epochs_run = df_history.shape[0]
            cv_report.append({
                "k": i + 1,
                "W0": w0_str,
                "Wf": wf_str,
                "errorT": f"{error_t:.8f}",
                "errorV": f"{error_v:.8f}",
                "error": f"{total_error:.8f}",
                "# epocas": epochs_run
            })
            models_info.append({
                'model': model,
                'history': history_tracker,
                'val_loss': val_loss
            })
            self.root.after(0, lambda i=i: self.progress.config(value=i+1))

        best_info = min(models_info, key=lambda m: m['val_loss'])
        best_model = best_info['model']
        best_history = best_info['history']
        final_pred = best_model.predict(X).flatten()
    
        self.cv_report = pd.DataFrame(cv_report, columns=["k", "W0", "Wf", "errorT", "errorV", "error", "# epocas"])
        self.root.after(0, lambda: self.display_results(best_history, final_pred, y, self.cv_report))

    def display_results(self, history_tracker: TrainingHistory, final_pred: np.ndarray, actual: np.ndarray, cv_report_df: pd.DataFrame):
        df_history = history_tracker.get_dataframe()
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)
        self.tree_history["columns"] = list(df_history.columns)
        for col in df_history.columns:
            self.tree_history.heading(col, text=col)
            self.tree_history.column(col, anchor="center", width=80)
        for _, row in df_history.iterrows():
            self.tree_history.insert("", "end", values=list(row))

        self.ax.clear()
        self.ax.plot(actual, 'b-', label='y deseada')
        self.ax.plot(final_pred, 'r--', label='y calculada')
        self.ax.set_title("Comparación: y deseada vs y calculada (Mejor Modelo)")
        self.ax.set_xlabel("Muestra")
        self.ax.set_ylabel("Valor")
        self.ax.legend()
        self.canvas.draw()

        self.ax_error.clear()
        self.ax_error.plot(df_history["Época"], df_history["Error"], marker='o', color='green', linestyle='-', label="Error")
        self.ax_error.set_title("Evolución del Error (Mejor Modelo)")
        self.ax_error.set_xlabel("Época")
        self.ax_error.set_ylabel("Error")
        self.ax_error.legend()
        self.canvas_error.draw()

        for item in self.tree_cv_report.get_children():
            self.tree_cv_report.delete(item)
        self.tree_cv_report["columns"] = list(cv_report_df.columns)
        for col in cv_report_df.columns:
            self.tree_cv_report.heading(col, text=col)
            if col in ["W0", "Wf"]:
                width = 260
            else:
                width = 100
            self.tree_cv_report.column(col, anchor="center", width=width)
        for _, row in cv_report_df.iterrows():
            self.tree_cv_report.insert("", "end", values=list(row))
        self.btn_train_model.config(state="normal")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = NeuralNetworkGUI()
    app.run()