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
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold

# Función para calcular la especificidad a partir de la matriz de confusión
def compute_specificity(cm):
    specificities = []
    for i in range(len(cm)):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        spec = TN / (TN + FP) if (TN + FP) > 0 else 0
        specificities.append(spec)
    return np.mean(specificities)

# =============================================================================
# Funciones para procesamiento y carga de datos
# =============================================================================

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
    y = df.iloc[:, -1].values.astype(str)  # Etiquetas: "Perro", "Gato", "Perico"
    X_scaled, scaler_X = min_max_scale(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return df, X_scaled, y_encoded, scaler_X, None, le

# =============================================================================
# Modelo de Clasificación (Perceptrón de una sola capa)
# =============================================================================

class ClassificationModel(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.dense = layers.Dense(num_classes, activation="softmax",
                                  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01))
    def call(self, inputs):
        return self.dense(inputs)

def train_model_classification(X, y, lr: float, max_epochs: int, batch_size: int, num_classes: int):
    model = ClassificationModel(num_classes)
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    history = model.fit(X, y, epochs=max_epochs, batch_size=batch_size, verbose=0)
    return model, history

# =============================================================================
# Interfaz Gráfica con tkinter (Validación Cruzada y Métricas)
# =============================================================================

class NeuralNetworkGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clasificación - Validación Cruzada")
        self.root.geometry("1150x700")
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.setup_ui()
        self.df = None
        self.X = None
        self.y = None
        self.scaler_X = None
        self.le = None  # LabelEncoder
        self.best_model = None
        self.cv_report = None  # Reporte de validación cruzada (Reporte Matriz)
        self.metrics_report = None  # Reporte de la matriz de confusión y classification_report

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        sidebar = ttk.Frame(main_frame, padding=(10,10))
        sidebar.grid(row=0, column=0, sticky="ns")
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=0, column=1, sticky="nsew", padx=(10,0))
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        self.btn_select_file = ttk.Button(sidebar, text="Seleccionar CSV", command=self.select_file)
        self.btn_select_file.grid(row=0, column=0, sticky="ew", pady=5)
        self.lbl_file = ttk.Label(sidebar, text="No se ha seleccionado archivo", wraplength=150)
        self.lbl_file.grid(row=1, column=0, sticky="ew", pady=5)

        # Parámetros de Entrenamiento (solo máximo de épocas)
        param_frame = ttk.LabelFrame(sidebar, text="Parámetros de Entrenamiento", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Label(param_frame, text="Máx. Épocas:").grid(row=0, column=0, sticky="w")
        # Valor por defecto 20 épocas
        self.epochs_var = tk.StringVar(value="20")
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
        ttk.Label(param_frame, text="K Folds:").grid(row=3, column=0, sticky="w")
        self.kfolds_var = tk.StringVar(value="5")
        self.entry_kfolds = ttk.Entry(param_frame, textvariable=self.kfolds_var, width=10)
        self.entry_kfolds.grid(row=3, column=1, sticky="e", padx=5, pady=2)

        self.btn_train_model = ttk.Button(sidebar, text="Validación Cruzada", command=self.train_model_handler, state="disabled")
        self.btn_train_model.grid(row=4, column=0, sticky="ew", pady=10)
        self.progress = ttk.Progressbar(sidebar, orient="horizontal", mode="determinate")
        self.progress.grid(row=5, column=0, sticky="ew", pady=5)

        # Pestañas del Notebook:
        self.tab_data = ttk.Frame(notebook)
        notebook.add(self.tab_data, text="Vista de Datos")
        self.setup_tab_data(self.tab_data)
        # La pestaña de Reporte CV se renombra a Reporte Matriz para mostrar las métricas por fold
        self.tab_cv_report = ttk.Frame(notebook)
        notebook.add(self.tab_cv_report, text="Reporte Matriz")
        self.setup_tab_cv_report(self.tab_cv_report)
        # Pestaña "Métricas" se mantiene para la matriz de confusión y el classification_report
        self.tab_metrics = ttk.Frame(notebook)
        notebook.add(self.tab_metrics, text="Métricas")
        self.setup_tab_metrics(self.tab_metrics)

    def setup_tab_data(self, parent):
        self.tree_data = ttk.Treeview(parent, show="headings")
        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.tree_data.yview)
        self.tree_data.configure(yscrollcommand=vsb.set)
        self.tree_data.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
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

    def setup_tab_metrics(self, parent):
        # Pestaña "Métricas": se muestra la matriz de confusión y el classification_report
        self.fig_cm = Figure(figsize=(5, 4), dpi=100)
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=parent)
        self.canvas_cm.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.text_report = tk.Text(parent, height=8, wrap="none", font=("Courier New", 10))
        self.text_report.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.lbl_file.config(text=file_path)
            self.df, self.X, self.y, self.scaler_X, _, self.le = load_data(file_path)
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
            k_folds = int(self.kfolds_var.get())
        except Exception as e:
            messagebox.showerror("Error", "Verifica los parámetros de entrenamiento.")
            return
        self.progress.config(maximum=k_folds, value=0)
        self.btn_train_model.config(state="disabled")
        thread = threading.Thread(target=self.run_cross_validation, args=(self.X, self.y, lr, max_epochs, batch_size, k_folds))
        thread.start()

    def run_cross_validation(self, X, y, lr, max_epochs, batch_size, k_folds):
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        models_info = []
        fold_report = []
        all_y_true = []
        all_y_pred = []
        fold_number = 1
        
        for train_index, val_index in kf.split(X, y):
            X_train, X_val = X[train_index], X[val_index]
            y_train, y_val = y[train_index], y[val_index]
            model, _ = train_model_classification(X_train, y_train, lr, max_epochs, batch_size, num_classes=3)
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            y_val_pred_prob = model.predict(X_val, verbose=0)
            y_val_pred = np.argmax(y_val_pred_prob, axis=1)
            all_y_true.extend(y_val)
            all_y_pred.extend(y_val_pred)
            fold_acc = accuracy_score(y_val, y_val_pred)
            fold_prec = precision_score(y_val, y_val_pred, average='macro', zero_division=0)
            fold_rec = recall_score(y_val, y_val_pred, average='macro', zero_division=0)
            fold_f1 = f1_score(y_val, y_val_pred, average='macro', zero_division=0)
            fold_report.append({
                "k": fold_number,
                "Accuracy": f"{fold_acc:.4f}",
                "Precision": f"{fold_prec:.4f}",
                "Recall": f"{fold_rec:.4f}",
                "F1-Score": f"{fold_f1:.4f}",
                "# epocas": max_epochs
            })
            models_info.append({
                'model': model,
                'val_loss': val_loss
            })
            self.root.after(0, lambda: self.progress.step(1))
            fold_number += 1
        
        best_info = min(models_info, key=lambda m: m['val_loss'])
        self.best_model = best_info['model']
        self.cv_report = pd.DataFrame(fold_report, columns=["k", "Accuracy", "Precision", "Recall", "F1-Score", "# epocas"])
        final_pred_prob = self.best_model.predict(X, verbose=0)
        final_pred = np.argmax(final_pred_prob, axis=1)
        
        # Calcular métricas globales y classification_report
        report_str = classification_report(y, final_pred, target_names=self.le.classes_, zero_division=0)
        cm = confusion_matrix(y, final_pred)
        acc = accuracy_score(y, final_pred)
        prec = precision_score(y, final_pred, average='macro', zero_division=0)
        rec = recall_score(y, final_pred, average='macro', zero_division=0)
        f1 = f1_score(y, final_pred, average='macro', zero_division=0)
        spec = compute_specificity(cm)
        self.metrics_report = {
            "Exactitud": acc,
            "Precisión": prec,
            "Sensibilidad": rec,
            "Especificidad": spec,
            "Puntuación F": f1,
            "Matriz": cm,
            "Report": report_str
        }
        self.root.after(0, lambda: self.display_results(final_pred, y, self.cv_report, self.metrics_report))

    def display_results(self, final_pred: np.ndarray, actual: np.ndarray, cv_report_df: pd.DataFrame, metrics_report: dict):
        # Actualizar pestaña "Reporte Matriz"
        for item in self.tree_cv_report.get_children():
            self.tree_cv_report.delete(item)
        self.tree_cv_report["columns"] = list(cv_report_df.columns)
        for col in cv_report_df.columns:
            self.tree_cv_report.heading(col, text=col)
            self.tree_cv_report.column(col, anchor="center", width=120)
        for _, row in cv_report_df.iterrows():
            self.tree_cv_report.insert("", "end", values=list(row))
        
        # Actualizar pestaña "Métricas": matriz de confusión y reporte
        cm = metrics_report["Matriz"]
        classes = self.le.classes_
        cm_df = pd.DataFrame(cm, index=classes, columns=classes)
        print(cm_df)

        
        # Limpiar el eje y eliminar la barra de color anterior
        self.ax_cm.clear()
        # Eliminar la barra de color si existe
        if len(self.fig_cm.axes) > 1:
            self.fig_cm.delaxes(self.fig_cm.axes[-1])
        
        # Generar nuevo heatmap
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=self.ax_cm)
        self.ax_cm.set_xlabel("Predichos")
        self.ax_cm.set_ylabel("Reales")
        self.ax_cm.set_title("Matriz de Confusión")
        for i in range(cm_df.shape[0]):
            self.ax_cm.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='green', lw=2))
        self.canvas_cm.draw()
    
    # Actualizar reporte de clasificación
        self.text_report.delete("1.0", tk.END)
        self.text_report.insert(tk.END, metrics_report["Report"])

        self.btn_train_model.config(state="normal")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = NeuralNetworkGUI()
    app.run()
