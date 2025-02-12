import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# =============================================================================
# Funciones y clases para el procesamiento y entrenamiento
# =============================================================================

# Función para realizar el escalado Min-Max sin usar sklearn
def min_max_scale(data: np.ndarray):
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    scale = np.where(data_max - data_min == 0, 1, data_max - data_min)
    scaled = (data - data_min) / scale
    return scaled, {"min": data_min, "max": data_max}

# Función para leer y preprocesar el dataset CSV
def load_data(csv_path: str):
    df = pd.read_csv(csv_path, sep=";")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()

    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].values.astype(np.float32)

    # Escalado usando función propia
    X_scaled, scaler_X = min_max_scale(X)
    y_scaled, scaler_y = min_max_scale(y.reshape(-1, 1))
    y_scaled = y_scaled.flatten()

    return df, X_scaled, y_scaled, scaler_X, scaler_y

# Clase para almacenar el historial de entrenamiento
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

# Callback para almacenar el historial, actualizar el progreso y detener el entrenamiento
# cuando se alcance la tolerancia de error.
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

# Arquitectura de la Red Neuronal (regresión lineal simple)
class LinearRegression(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))

    def call(self, inputs):
        return self.dense(inputs)

# Función para entrenar el modelo utilizando model.fit y el callback personalizado.
def train_model(X, y, lr: float, max_epochs: int, batch_size: int, tolerance: float, progress_callback=None):
    model = LinearRegression()
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss=keras.losses.MeanSquaredError())

    history_tracker = TrainingHistory()
    callback = TrainingHistoryCallback(history_tracker, tolerance, progress_callback)

    # Predicción inicial (opcional)
    initial_predictions = model.predict(X).flatten()
    model.fit(X, y, epochs=max_epochs, batch_size=batch_size, verbose=0, callbacks=[callback])
    final_predictions = model.predict(X).flatten()

    return model, history_tracker, initial_predictions, final_predictions, y

# =============================================================================
# Interfaz Gráfica Moderna con tkinter
# =============================================================================

class NeuralNetworkGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Red Neuronal - Regresión Lineal")
        self.root.geometry("1000x600")
        # Usar un tema moderno de ttk (por ejemplo, 'clam', 'vista' o 'alt')
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.setup_ui()
        # Variables para almacenar datos y modelo
        self.df = None
        self.X = None
        self.y = None
        self.scaler_X = None
        self.scaler_y = None
        self.model = None

    def setup_ui(self):
        # Marco principal que utiliza grid
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Sidebar de controles (columna izquierda)
        sidebar = ttk.Frame(main_frame, padding=(10, 10))
        sidebar.grid(row=0, column=0, sticky="ns")

        # Área de visualización: Notebook (columna derecha)
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # ================================
        # Sidebar: Controles
        # ================================
        self.btn_select_file = ttk.Button(sidebar, text="Seleccionar CSV", command=self.select_file)
        self.btn_select_file.grid(row=0, column=0, sticky="ew", pady=5)

        self.lbl_file = ttk.Label(sidebar, text="No se ha seleccionado archivo", wraplength=150)
        self.lbl_file.grid(row=1, column=0, sticky="ew", pady=5)

        # Parámetros de entrenamiento
        param_frame = ttk.LabelFrame(sidebar, text="Parámetros de Entrenamiento", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", pady=5)

        # Se usa un máximo de épocas como límite de seguridad, pero el entrenamiento se
        # detendrá en cuanto se alcance la tolerancia.
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

        self.btn_train_model = ttk.Button(sidebar, text="Entrenar Modelo", command=self.train_model_handler, state="disabled")
        self.btn_train_model.grid(row=4, column=0, sticky="ew", pady=10)

        # Barra de progreso
        self.progress = ttk.Progressbar(sidebar, orient="horizontal", mode="determinate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=5)

        # ================================
        # Notebook: Pestañas de visualización
        # ================================

        # Pestaña 1: Vista de Datos
        self.tab_data = ttk.Frame(notebook)
        notebook.add(self.tab_data, text="Vista de Datos")
        self.setup_tab_data(self.tab_data)

        # Pestaña 2: Historial de Pesos
        self.tab_history = ttk.Frame(notebook)
        notebook.add(self.tab_history, text="Historial de Pesos")
        self.setup_tab_history(self.tab_history)

        # Pestaña 3: Gráfica de Resultados
        self.tab_plot = ttk.Frame(notebook)
        notebook.add(self.tab_plot, text="Gráfica")
        self.setup_tab_plot(self.tab_plot)

        # Pestaña 4: Evolución del Error
        self.tab_error = ttk.Frame(notebook)
        notebook.add(self.tab_error, text="Evolución del Error")
        self.setup_tab_error(self.tab_error)

    # Configuración de la pestaña de vista de datos
    def setup_tab_data(self, parent):
        self.tree_data = ttk.Treeview(parent, show="headings")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=vsb.set)
        self.tree_data.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    # Configuración de la pestaña de historial de pesos
    def setup_tab_history(self, parent):
        self.tree_history = ttk.Treeview(parent, show="headings")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree_history.yview)
        self.tree_history.configure(yscrollcommand=vsb.set)
        self.tree_history.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    # Configuración de la pestaña de la gráfica de comparación
    def setup_tab_plot(self, parent):
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    # Configuración de la pestaña de evolución del error
    def setup_tab_error(self, parent):
        self.fig_error = Figure(figsize=(5, 4), dpi=100)
        self.ax_error = self.fig_error.add_subplot(111)
        self.canvas_error = FigureCanvasTkAgg(self.fig_error, master=parent)
        self.canvas_error.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    # =============================================================================
    # Funciones de manejo de eventos
    # =============================================================================

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.lbl_file.config(text=file_path)
            self.df, self.X, self.y, self.scaler_X, self.scaler_y = load_data(file_path)
            self.populate_data_tree(self.df)
            self.btn_train_model.config(state="normal")

    def populate_data_tree(self, df: pd.DataFrame):
        # Limpiar la vista previa anterior
        for item in self.tree_data.get_children():
            self.tree_data.delete(item)
        self.tree_data["columns"] = list(df.columns)
        for col in df.columns:
            self.tree_data.heading(col, text=col)
            self.tree_data.column(col, width=100, anchor="center")
        # Insertar las primeras 10 filas
        for _, row in df.head(10).iterrows():
            self.tree_data.insert("", "end", values=list(row))

    def train_model_handler(self):
        try:
            max_epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            tolerance = float(self.tol_var.get())
        except Exception as e:
            messagebox.showerror("Error", "Verifica los parámetros de entrenamiento.")
            return

        self.progress.config(maximum=max_epochs, value=0)
        self.btn_train_model.config(state="disabled")
        # Ejecutar el entrenamiento en un hilo para no bloquear la UI
        thread = threading.Thread(target=self.run_training, args=(self.X, self.y, lr, max_epochs, batch_size, tolerance))
        thread.start()

    def run_training(self, X, y, lr, max_epochs, batch_size, tolerance):
        # Función para actualizar la barra de progreso (se programa en el hilo principal)
        def progress_update(epoch):
            self.root.after(0, lambda: self.progress.config(value=epoch))

        model, history_tracker, initial_pred, final_pred, actual = train_model(
            X, y, lr=lr, max_epochs=max_epochs, batch_size=batch_size, tolerance=tolerance, progress_callback=progress_update
        )
        # Actualizar la interfaz con los resultados al finalizar el entrenamiento
        self.root.after(0, lambda: self.display_results(history_tracker, final_pred, actual))

    def display_results(self, history_tracker: TrainingHistory, final_pred: np.ndarray, actual: np.ndarray):
        # Actualizar el historial de pesos en la pestaña correspondiente
        df_history = history_tracker.get_dataframe()
        for item in self.tree_history.get_children():
            self.tree_history.delete(item)
        self.tree_history["columns"] = list(df_history.columns)
        for col in df_history.columns:
            self.tree_history.heading(col, text=col)
            self.tree_history.column(col, anchor="center", width=80)
        for _, row in df_history.iterrows():
            self.tree_history.insert("", "end", values=list(row))

        # Actualizar la gráfica de comparación
        self.ax.clear()
        self.ax.plot(actual, 'b-', label='y deseada')
        self.ax.plot(final_pred, 'r--', label='y calculada')
        self.ax.set_title("Comparación: y deseada vs y calculada")
        self.ax.set_xlabel("Muestra")
        self.ax.set_ylabel("Valor")
        self.ax.legend()
        self.canvas.draw()

        # Actualizar la gráfica de evolución del error
        self.ax_error.clear()
        self.ax_error.plot(df_history["Época"], df_history["Error"], marker='o', color='green', linestyle='-', label="Error")
        self.ax_error.set_title("Evolución del Error")
        self.ax_error.set_xlabel("Época")
        self.ax_error.set_ylabel("Error")
        self.ax_error.legend()
        self.canvas_error.draw()

        self.btn_train_model.config(state="normal")

    def run(self):
        self.root.mainloop()

# =============================================================================
# Ejecución de la aplicación
# =============================================================================

if __name__ == "__main__":
    app = NeuralNetworkGUI()
    app.run()
