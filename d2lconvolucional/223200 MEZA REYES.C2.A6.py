import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
import tensorflow as tf # type: ignore
from d2l import tensorflow as d2l_tf
from tensorflow import keras # type: ignore
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.preprocessing import image_dataset_from_directory # type: ignore
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt # type: ignore
from matplotlib.figure import Figure # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
import threading
import seaborn as sns # type: ignore
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score # type: ignore
from sklearn.model_selection import KFold # type: ignore
from PIL import Image, ImageTk # type: ignore
from tkinter import simpledialog

def normalize_dataset(ds):    
    # La normalización en d2l con tensorflow mantiene la estructura original
    return ds.map(lambda x, y: (tf.cast(x, tf.float32)/255.0, y))

def load_images(root_folder, image_size=(24,24), batch_size=32, validation_split=0.2, seed=123):    
    # Usamos las funciones de d2l para cargar y procesar imágenes con TensorFlow
    # pero mantenemos la estructura del código original
    train_ds_orig = tf.keras.preprocessing.image_dataset_from_directory(
        root_folder,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="training",
        seed=seed
    )
    val_ds_orig = tf.keras.preprocessing.image_dataset_from_directory(
        root_folder,
        labels="inferred",
        label_mode="int",
        color_mode="grayscale",
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed
    )    
    class_names = train_ds_orig.class_names
    
    # Aplicamos la normalización usando d2l_tf para mantener compatibilidad
    train_ds = normalize_dataset(train_ds_orig)
    val_ds = normalize_dataset(val_ds_orig)
    
    # Preparamos los datasets con el formato de d2l
    train_ds = d2l_tf.preprocess_data(train_ds, batch_size)
    val_ds = d2l_tf.preprocess_data(val_ds, batch_size)
    
    return train_ds, val_ds, class_names

def build_cnn_model(input_shape, num_classes, lr):
    # Usar d2l con TensorFlow para crear el modelo CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', 
                               input_shape=input_shape),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])
    
    # Inicializar usando d2l
    d2l_tf.initialize_parameters(model)
    
    # Compilar el modelo usando la función de pérdida y el optimizador
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    return model

class NeuralNetworkGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Clasificación de Imágenes - Red Neuronal Convolucional")
        self.root.geometry("1150x700")
        self.style = ttk.Style(self.root)
        self.style.theme_use('clam')
        self.setup_ui()
        self.train_ds = None
        self.val_ds = None
        self.class_names = None
        self.best_model = None
        self.y_true = None
        self.y_pred = None
        self.X = None
        self.y = None

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
        
        self.btn_select_folder = ttk.Button(sidebar, text="Seleccionar Carpeta de Imágenes", command=self.select_folder)
        self.btn_select_folder.grid(row=0, column=0, sticky="ew", pady=5)
        self.lbl_folder = ttk.Label(sidebar, text="No se ha seleccionado carpeta", wraplength=150)
        self.lbl_folder.grid(row=1, column=0, sticky="ew", pady=5)
        
        param_frame = ttk.LabelFrame(sidebar, text="Parámetros de Entrenamiento", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Label(param_frame, text="Épocas:").grid(row=0, column=0, sticky="w")
        self.epochs_var = tk.StringVar(value="10")
        self.entry_epochs = ttk.Entry(param_frame, textvariable=self.epochs_var, width=10)
        self.entry_epochs.grid(row=0, column=1, sticky="e", padx=5, pady=2)
        ttk.Label(param_frame, text="Learning Rate:").grid(row=1, column=0, sticky="w")
        self.lr_var = tk.StringVar(value="0.00001")
        self.entry_lr = ttk.Entry(param_frame, textvariable=self.lr_var, width=10)
        self.entry_lr.grid(row=1, column=1, sticky="e", padx=5, pady=2)
        ttk.Label(param_frame, text="Batch Size:").grid(row=2, column=0, sticky="w")
        self.batch_var = tk.StringVar(value="32")
        self.entry_batch = ttk.Entry(param_frame, textvariable=self.batch_var, width=10)
        self.entry_batch.grid(row=2, column=1, sticky="e", padx=5, pady=2)        
        ttk.Label(param_frame, text="Número de K-Folds:").grid(row=3, column=0, sticky="w")
        self.kfolds_var = tk.StringVar(value="11")
        self.entry_kfolds = ttk.Entry(param_frame, textvariable=self.kfolds_var, width=10)
        self.entry_kfolds.grid(row=3, column=1, sticky="e", padx=5, pady=2)
        
        self.use_kfold_var = tk.BooleanVar(value=False)
        self.check_kfold = ttk.Checkbutton(param_frame, text="Usar validación cruzada", 
                                           variable=self.use_kfold_var)
        self.check_kfold.grid(row=4, column=0, columnspan=2, sticky="w", pady=5)

        self.btn_train_model = ttk.Button(sidebar, text="Entrenar Modelo", command=self.train_model_handler, state="disabled")
        self.btn_train_model.grid(row=3, column=0, sticky="ew", pady=10)
        self.progress = ttk.Progressbar(sidebar, orient="horizontal", mode="determinate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=5)
                
        self.lbl_progress = ttk.Label(sidebar, text="")
        self.lbl_progress.grid(row=5, column=0, sticky="ew", pady=5)
                
        self.btn_save_model = ttk.Button(sidebar, text="Guardar Mejor Modelo", 
                                       command=self.save_best_model, state="disabled")
        self.btn_save_model.grid(row=6, column=0, sticky="ew", pady=10)

        self.btn_load_model = ttk.Button(sidebar, text="Cargar Modelo", 
                                       command=self.load_model)
        self.btn_load_model.grid(row=7, column=0, sticky="ew", pady=5)
        
        ttk.Separator(sidebar, orient="horizontal").grid(row=9, column=0, sticky="ew", pady=10)
        
        info_label = ttk.Label(sidebar, text="Clasificador CNN\nv1.0", 
                              font=("Arial", 8), justify="center")
        info_label.grid(row=10, column=0, sticky="ew", pady=5)
        
        self.tab_dataset = ttk.Frame(notebook)
        notebook.add(self.tab_dataset, text="Datos")
        self.setup_tab_dataset(self.tab_dataset)
        self.tab_metrics = ttk.Frame(notebook)
        notebook.add(self.tab_metrics, text="Métricas")
        self.setup_tab_metrics(self.tab_metrics)
                
        self.tab_kfold = ttk.Frame(notebook)
        notebook.add(self.tab_kfold, text="Validación Cruzada")
        self.setup_tab_kfold(self.tab_kfold)

    def setup_tab_dataset(self, parent):        
        self.txt_dataset = tk.Text(parent, height=15, wrap="none", font=("Arial", 10))
        self.txt_dataset.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_tab_metrics(self, parent):        
        self.fig_cm = Figure(figsize=(5, 4), dpi=100)
        self.ax_cm = self.fig_cm.add_subplot(111)
        self.canvas_cm = FigureCanvasTkAgg(self.fig_cm, master=parent)
        self.canvas_cm.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.text_report = tk.Text(parent, height=8, wrap="none", font=("Courier New", 10))
        self.text_report.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def setup_tab_kfold(self, parent):        
        self.fig_kfold = Figure(figsize=(5, 4), dpi=100)
        self.ax_kfold = self.fig_kfold.add_subplot(111)
        self.canvas_kfold = FigureCanvasTkAgg(self.fig_kfold, master=parent)
        self.canvas_kfold.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.text_kfold = tk.Text(parent, height=8, wrap="word", font=("Arial", 10))
        self.text_kfold.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Modelos de Keras", "*.h5"), ("Todos los archivos", "*.*")],
            title="Cargar modelo entrenado"
        )

        if file_path:
            try:                
                self.best_model = keras.models.load_model(file_path)
                
                class_file = os.path.splitext(file_path)[0] + "_classes.txt"
                if os.path.exists(class_file):
                    with open(class_file, 'r') as f:
                        self.class_names = f.read().splitlines()
                else:                    
                    class_names = simpledialog.askstring(
                        "Nombres de clases", 
                        "Introduzca los nombres de las clases separados por comas:",
                        initialvalue="clase0,clase1,clase2,clase3,clase4,clase5"
                    )
                    if class_names:
                        self.class_names = [name.strip() for name in class_names.split(",")]
                    else:
                        messagebox.showerror("Error", "Se necesitan los nombres de las clases.")
                        return

                messagebox.showinfo("Éxito", f"Modelo cargado desde {file_path}")
                self.btn_predict.config(state="normal")
                
                summary = []
                self.best_model.summary(print_fn=lambda x: summary.append(x))
                self.txt_dataset.delete("1.0", tk.END)
                self.txt_dataset.insert(tk.END, "Modelo cargado:\n" + "\n".join(summary))
                self.txt_dataset.insert(tk.END, f"\n\nClases cargadas:\n" + "\n".join([f"{i}: {name}" for i, name in enumerate(self.class_names)]))

            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{str(e)}")

    def display_prediction_results(self, image_path, result_text, predictions):        
        result_window = tk.Toplevel(self.root)
        result_window.title("Resultados de la clasificación")
        result_window.geometry("600x500")
                
        img_frame = ttk.Frame(result_window, padding=10)
        img_frame.pack(fill=tk.BOTH, expand=True)
                
        img = Image.open(image_path)        
        width, height = img.size
        if width > 400 or height > 400:
            ratio = min(400/width, 400/height)
            width = int(width * ratio)
            height = int(height * ratio)
            img = img.resize((width, height), Image.LANCZOS)
                
        img_tk = ImageTk.PhotoImage(img)
        img_label = ttk.Label(img_frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack(pady=10)
                
        result_frame = ttk.Frame(result_window, padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        result_text_widget = tk.Text(result_frame, height=10, wrap="word", font=("Arial", 10))
        result_text_widget.pack(fill=tk.BOTH, expand=True, pady=5)
        result_text_widget.insert(tk.END, result_text)
                
        fig = Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        
        y_pos = np.arange(len(self.class_names))
        ax.barh(y_pos, predictions, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(self.class_names)
        ax.invert_yaxis()
        ax.set_xlabel('Probabilidad')
        ax.set_title('Probabilidades por clase')
        
        canvas = FigureCanvasTkAgg(fig, master=result_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
        ttk.Button(result_window, text="Cerrar", command=result_window.destroy).pack(pady=10)

    def select_folder(self):
        folder_path = filedialog.askdirectory(title="Seleccionar carpeta raíz con imágenes")
        if folder_path:
            self.lbl_folder.config(text=folder_path)
            try:                
                train_ds, val_ds, class_names = load_images(folder_path, image_size=(24,24), batch_size=int(self.batch_var.get()))
                self.train_ds = train_ds
                self.val_ds = val_ds
                self.class_names = class_names
                                
                X, y, _ = load_all_images(folder_path, image_size=(24,24), batch_size=int(self.batch_var.get()))
                self.X = X
                self.y = y
                                
                info = "Clases encontradas:\n" + "\n".join([f"{i}: {name}" for i, name in enumerate(class_names)])
                info += f"\n\nTotal de imágenes para validación cruzada: {len(self.y) if self.y is not None else 0}"
                class_counts = np.bincount(self.y) if self.y is not None else []
                for i, count in enumerate(class_counts):
                    info += f"\n  Clase {i} ({class_names[i]}): {count} imágenes"
                
                self.txt_dataset.delete("1.0", tk.END)
                self.txt_dataset.insert(tk.END, info)
                self.btn_train_model.config(state="normal")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudieron cargar las imágenes:\n{str(e)}")
                self.btn_train_model.config(state="disabled")

    def save_best_model(self):
        if self.best_model is None:
            messagebox.showerror("Error", "No hay un modelo entrenado para guardar.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("Modelos de Keras", "*.h5"), ("Todos los archivos", "*.*")],
            title="Guardar el mejor modelo"
        )

        if file_path:
            try:
                self.best_model.save(file_path)                
                class_file = os.path.splitext(file_path)[0] + "_classes.txt"
                with open(class_file, 'w') as f:
                    f.write('\n'.join(self.class_names))

                messagebox.showinfo("Éxito", f"Modelo guardado en {file_path}\nClases guardadas en {class_file}")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar el modelo:\n{str(e)}")

    def train_model_handler(self):
        if (self.train_ds is None or self.val_ds is None) and not self.use_kfold_var.get():
            messagebox.showerror("Error", "Primero debe seleccionar una carpeta con imágenes.")
            return
        if (self.X is None or self.y is None) and self.use_kfold_var.get():
            messagebox.showerror("Error", "No se pudieron cargar las imágenes para validación cruzada.")
            return

        try:
            epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            batch_size = int(self.batch_var.get())
            if self.use_kfold_var.get():
                k_folds = int(self.kfolds_var.get())
                if k_folds < 2:
                    messagebox.showerror("Error", "El número de K-Folds debe ser al menos 2.")
                    return
        except Exception as e:
            messagebox.showerror("Error", "Verifique los parámetros de entrenamiento.")
            return

        self.progress.config(maximum=100, value=0)
        self.btn_train_model.config(state="disabled")
        self.btn_save_model.config(state="disabled")

        if self.use_kfold_var.get():
            thread = threading.Thread(target=self.run_kfold_training, args=(epochs, lr, batch_size, k_folds))
        else:
            thread = threading.Thread(target=self.run_training, args=(epochs, lr, batch_size))
        thread.start()

    def run_kfold_training(self, epochs, lr, batch_size, k_folds): 
        if self.X is None or self.y is None:
            messagebox.showerror("Error", "No se pudieron cargar las imágenes para validación cruzada.")
            return

        num_classes = len(self.class_names)
        input_shape = self.X.shape[1:]

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

        fold_accuracies = []
        fold_precisions = []
        fold_recalls = []
        fold_f1s = []
        fold_train_errors = []
        fold_val_errors = []
        best_accuracy = 0
        best_model = None
        best_fold = 0
        all_y_true = []
        all_y_pred = []
        fold_models = []
        accumulated_cm = np.zeros((num_classes, num_classes), dtype=int)

        self.progress.config(value=0)
        total_steps = k_folds * epochs
        current_step = 0

        # Usar d2l_tf.Animator para rastrear el entrenamiento
        animator = d2l_tf.Animator(xlabel='epoch', xlim=[1, epochs],
                                   legend=['train loss', 'val loss', 'train acc', 'val acc'])

        for fold, (train_idx, val_idx) in enumerate(kfold.split(self.X, self.y)):
            self.lbl_progress.config(text=f"Entrenando fold {fold+1}/{k_folds}...")
            X_train, y_train = self.X[train_idx], self.y[train_idx]
            X_val, y_val = self.X[val_idx], self.y[val_idx]

            # Crear datasets de TensorFlow usando d2l
            train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
            val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

            # Aplicar preprocesamiento de d2l
            train_ds = d2l_tf.preprocess_data(train_ds, batch_size)
            val_ds = d2l_tf.preprocess_data(val_ds, batch_size)

            # Crear modelo usando d2l
            model = build_cnn_model(input_shape, num_classes, lr)

            # Callback para actualizar la barra de progreso
            class FoldProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_bar, epochs, fold, k_folds, parent, current_step, total_steps):
                    self.progress_bar = progress_bar
                    self.epochs = epochs
                    self.fold = fold
                    self.k_folds = k_folds
                    self.parent = parent
                    self.current_step = current_step
                    self.total_steps = total_steps

                def on_epoch_end(self, epoch, logs=None):
                    self.current_step += 1
                    progress = int(self.current_step / self.total_steps * 100)
                    self.parent.root.after(0, lambda: self.progress_bar.config(value=progress))
                    self.parent.root.after(0, lambda: self.parent.lbl_progress.config(
                        text=f"Fold {self.fold+1}/{self.k_folds} - Epoch {epoch+1}/{self.epochs} - "
                             f"acc: {logs.get('accuracy'):.4f} - val_acc: {logs.get('val_accuracy'):.4f}"))

            callback = FoldProgressCallback(self.progress, epochs, fold, k_folds, self, current_step, total_steps)

            # Entrenar el modelo usando d2l
            # Utilizando la interfaz de d2l_tf para mantener la compatibilidad con TensorFlow
            d2l_trainer = d2l_tf.Trainer(max_epochs=epochs)
            history = model.fit(
                train_ds, 
                validation_data=val_ds, 
                epochs=epochs, 
                verbose=0, 
                callbacks=[callback]
            )

            current_step += epochs

            # Extraer métricas
            train_loss = history.history['loss'][-1]
            val_loss = history.history['val_loss'][-1]
            fold_train_errors.append(train_loss)
            fold_val_errors.append(val_loss)

            # Hacer predicciones
            val_pred = np.argmax(model.predict(X_val), axis=1)
            fold_acc = accuracy_score(y_val, val_pred)
            fold_prec = precision_score(y_val, val_pred, average="macro", zero_division=0)
            fold_rec = recall_score(y_val, val_pred, average="macro", zero_division=0)
            fold_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)

            fold_accuracies.append(fold_acc)
            fold_precisions.append(fold_prec)
            fold_recalls.append(fold_rec)
            fold_f1s.append(fold_f1)
            fold_models.append(model)

            if fold_acc > best_accuracy:
                best_accuracy = fold_acc
                best_model = model
                best_fold = fold + 1

            all_y_true.extend(y_val)
            all_y_pred.extend(val_pred)

            fold_cm = confusion_matrix(y_val, val_pred, labels=np.arange(num_classes))
            print(f"Matriz de confusión del fold {fold+1}:\n", fold_cm)
            accumulated_cm += fold_cm

            print(f"Fold {fold+1} - Accuracy: {fold_acc:.4f}, Precision: {fold_prec:.4f}, "
                  f"Recall: {fold_rec:.4f}, F1: {fold_f1:.4f}")
            print(f"Matriz de confusión del fold {fold+1}:\n", fold_cm)

            # Actualizar el animador de d2l
            d2l_tf.update_animator(animator, epoch=epochs, 
                                train_loss=train_loss, test_loss=val_loss,
                                train_acc=history.history['accuracy'][-1], 
                                test_acc=history.history['val_accuracy'][-1])

        self.best_model = best_model
        cm = confusion_matrix(all_y_true, all_y_pred, labels=np.arange(num_classes))
        y_true_counts = cm.sum(axis=1)
        y_pred_counts = cm.sum(axis=0)
        acc = np.trace(cm) / np.sum(cm)
        prec = np.mean(fold_precisions)
        rec = np.mean(fold_recalls)
        f1 = np.mean(fold_f1s)

        avg_train_error = np.mean(fold_train_errors)
        avg_val_error = np.mean(fold_val_errors)
        total_error = (avg_train_error + avg_val_error) / 2

        best_train_error = fold_train_errors[best_fold - 1]
        best_val_error = fold_val_errors[best_fold - 1]
        best_total_error = (best_train_error + best_val_error) / 2

        # Usar d2l para generar un informe mejorado
        report_str = f"Informe de Clasificación (Promedio de {k_folds} folds):\n\n"
        report_str += f"              precisión  recall  f1-score  support\n"
        for i in range(num_classes):
            class_precision = cm[i, i] / y_pred_counts[i] if y_pred_counts[i] > 0 else 0
            class_recall = cm[i, i] / y_true_counts[i] if y_true_counts[i] > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            report_str += f"{self.class_names[i]:<14} {class_precision:.2f}     {class_recall:.2f}     {class_f1:.2f}      {y_true_counts[i]}\n"

        report_str += f"\naccuracy                          {acc:.2f}      {np.sum(cm)}\n"
        report_str += f"macro avg       {prec:.2f}     {rec:.2f}     {f1:.2f}      {np.sum(cm)}\n"

        report_str += f"\nError global de entrenamiento promedio: {avg_train_error:.4f}"
        report_str += f"\nError global de validación promedio: {avg_val_error:.4f}"
        report_str += f"\nError global total: {total_error:.4f}\n"

        report_str += f"\n--- Mejor Modelo (Fold {best_fold}) ---"
        report_str += f"\nError de entrenamiento: {best_train_error:.4f}"
        report_str += f"\nError de validación: {best_val_error:.4f}"
        report_str += f"\nError total: {best_total_error:.4f}\n"

        self.metrics_report = {
            "Exactitud": acc,
            "Precisión": prec,
            "Sensibilidad": rec,
            "Puntuación F": f1,
            "Matriz": cm,
            "Report": report_str,
            "Fold_Accuracies": fold_accuracies,
            "Fold_Precisions": fold_precisions,
            "Fold_Recalls": fold_recalls,
            "Fold_F1s": fold_f1s,
            "Fold_Train_Errors": fold_train_errors,
            "Fold_Val_Errors": fold_val_errors,
            "Avg_Train_Error": avg_train_error,
            "Avg_Val_Error": avg_val_error,
            "Total_Error": total_error,
            "Best_Train_Error": best_train_error,
            "Best_Val_Error": best_val_error,
            "Best_Total_Error": best_total_error,
            "Best_Fold": best_fold,
            "Best_Accuracy": best_accuracy,
            "Fold_Models": fold_models
        }

        print("Matriz de Confusión Global Acumulada:\n", cm)
        print("Diagonal (aciertos):", np.diag(cm))
        print(f"Mejor modelo: Fold {best_fold} con exactitud {best_accuracy:.4f}")

        # Generar visualizaciones de d2l
        d2l_tf.save_animator(animator, f'kfold_training_results.png')

        self.root.after(0, lambda: self.display_kfold_results())

    def display_kfold_results(self):
        cm = self.metrics_report["Matriz"]
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)
        self.ax_cm.clear()

        # Usar visualizaciones mejoradas de d2l para matriz de confusión
        d2l.plt.rcParams['figure.figsize'] = (10, 8)
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=self.ax_cm)
        self.ax_cm.set_xlabel("Predichos")
        self.ax_cm.set_ylabel("Reales")
        self.ax_cm.set_title("Matriz de Confusión (Todos los Folds)")
        self.canvas_cm.draw()

        self.text_report.delete("1.0", tk.END)
        self.text_report.insert(tk.END, self.metrics_report["Report"])

        # Usar gráficos de d2l para métricas por fold
        self.ax_kfold.clear()
        fold_indices = list(range(1, len(self.metrics_report["Fold_Accuracies"]) + 1))

        # Mejorar la visualización con estilos de d2l
        d2l.plt.style.use('ggplot')
        self.ax_kfold.plot(fold_indices, self.metrics_report["Fold_Accuracies"], 'o-', label='Exactitud')
        self.ax_kfold.plot(fold_indices, self.metrics_report["Fold_Precisions"], 's-', label='Precisión')
        self.ax_kfold.plot(fold_indices, self.metrics_report["Fold_Recalls"], '^-', label='Sensibilidad')
        self.ax_kfold.plot(fold_indices, self.metrics_report["Fold_F1s"], 'd-', label='F1')

        best_fold = self.metrics_report["Best_Fold"]
        best_accuracy = self.metrics_report["Best_Accuracy"]
        self.ax_kfold.plot(best_fold, best_accuracy, 'ro', markersize=10, label=f'Mejor Fold ({best_fold})')

        self.ax_kfold.set_xlabel('Número de Fold')
        self.ax_kfold.set_ylabel('Puntuación')
        self.ax_kfold.set_title('Métricas por Fold')
        self.ax_kfold.legend(loc='lower left')
        self.ax_kfold.grid(True)
        self.ax_kfold.set_xticks(fold_indices)
        self.canvas_kfold.draw()

        self.text_kfold.delete("1.0", tk.END)
        summary = f"Resultados de la Validación Cruzada ({len(fold_indices)}-Folds):\n\n"

        for i, (acc, prec, rec, f1) in enumerate(zip(
            self.metrics_report["Fold_Accuracies"],
            self.metrics_report["Fold_Precisions"],
            self.metrics_report["Fold_Recalls"],
            self.metrics_report["Fold_F1s"]
        )):
            summary += f"Fold {i+1}:\n"
            summary += f"  - Exactitud: {acc:.4f}\n"
            summary += f"  - Precisión: {prec:.4f}\n"
            summary += f"  - Sensibilidad: {rec:.4f}\n"
            summary += f"  - F1: {f1:.4f}\n\n"

        avg_acc = np.mean(self.metrics_report["Fold_Accuracies"])
        avg_prec = np.mean(self.metrics_report["Fold_Precisions"])
        avg_rec = np.mean(self.metrics_report["Fold_Recalls"])
        avg_f1 = np.mean(self.metrics_report["Fold_F1s"])

        summary += "Promedio de todos los folds:\n"
        summary += f"  - Exactitud: {avg_acc:.4f}\n"
        summary += f"  - Precisión: {avg_prec:.4f}\n"
        summary += f"  - Sensibilidad: {avg_rec:.4f}\n"
        summary += f"  - F1: {avg_f1:.4f}\n\n"

        summary += f"Mejor modelo: Fold {best_fold} con exactitud {best_accuracy:.4f}\n"
        summary += "(Este modelo será el guardado cuando presione 'Guardar Mejor Modelo')"

        self.text_kfold.insert(tk.END, summary)

        self.btn_train_model.config(state="normal")
        self.btn_save_model.config(state="normal")
        self.lbl_progress.config(text="Validación cruzada completada")

    # Método para ejecutar entrenamiento regular
    def run_training(self, epochs, lr, batch_size):
        num_classes = len(self.class_names)

        # Obtener la forma de entrada del dataset
        for x, y in self.train_ds:
            input_shape = x.shape[1:]
            break
        
        # Construir el modelo usando d2l con TensorFlow
        model = build_cnn_model(input_shape, num_classes, lr)

        # Configurar callback para la barra de progreso
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar, epochs, parent):
                self.progress_bar = progress_bar
                self.epochs = epochs
                self.parent = parent

            def on_epoch_end(self, epoch, logs=None):
                progress = int((epoch + 1) / self.epochs * 100)
                self.parent.root.after(0, lambda: self.progress_bar.config(value=progress))
                self.parent.root.after(0, lambda: self.parent.lbl_progress.config(
                    text=f"Epoch {epoch+1}/{self.epochs} - "
                         f"acc: {logs.get('accuracy'):.4f} - val_acc: {logs.get('val_accuracy'):.4f}"))

        # Crear un animador de d2l para seguimiento visual
        animator = d2l_tf.Animator(xlabel='epoch', xlim=[1, epochs],
                                 legend=['train loss', 'val loss', 'train acc', 'val acc'])

        # Entrenar el modelo con d2l y TensorFlow
        callback = ProgressCallback(self.progress, epochs, self)
        history = model.fit(
            self.train_ds, 
            validation_data=self.val_ds, 
            epochs=epochs, 
            verbose=0, 
            callbacks=[callback]
        )

        # Evaluar el modelo
        y_true = []
        y_pred = []

        # Recopilar etiquetas reales y predicciones
        for x_batch, y_batch in self.val_ds:
            batch_pred = np.argmax(model.predict(x_batch), axis=1)
            y_pred.extend(batch_pred)
            y_true.extend(y_batch.numpy())

        # Calcular métricas
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

        # Guardar el modelo
        self.best_model = model

        # Generar informe
        y_true_counts = cm.sum(axis=1)
        y_pred_counts = cm.sum(axis=0)

        report_str = f"Informe de Clasificación:\n\n"
        report_str += f"              precisión  recall  f1-score  support\n"
        for i in range(num_classes):
            class_precision = cm[i, i] / y_pred_counts[i] if y_pred_counts[i] > 0 else 0
            class_recall = cm[i, i] / y_true_counts[i] if y_true_counts[i] > 0 else 0
            class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
            report_str += f"{self.class_names[i]:<14} {class_precision:.2f}     {class_recall:.2f}     {class_f1:.2f}      {y_true_counts[i]}\n"

        report_str += f"\naccuracy                          {acc:.2f}      {np.sum(cm)}\n"
        report_str += f"macro avg       {prec:.2f}     {rec:.2f}     {f1:.2f}      {np.sum(cm)}\n"

        # Guardar métricas en formato d2l
        self.metrics_report = {
            "Exactitud": acc,
            "Precisión": prec,
            "Sensibilidad": rec,
            "Puntuación F": f1,
            "Matriz": cm,
            "Report": report_str,
            "History": history.history
        }

        # Guardar animación de d2l
        d2l_tf.save_animator(animator, 'training_results.png')

        # Actualizar la interfaz
        self.root.after(0, lambda: self.display_results())

    def display_results(self):
        # Mostrar resultados del entrenamiento usando visualizaciones de d2l
        cm = self.metrics_report["Matriz"]
        cm_df = pd.DataFrame(cm, index=self.class_names, columns=self.class_names)

        self.ax_cm.clear()
        d2l.plt.rcParams['figure.figsize'] = (10, 8)
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=self.ax_cm)
        self.ax_cm.set_xlabel("Predichos")
        self.ax_cm.set_ylabel("Reales")
        self.ax_cm.set_title("Matriz de Confusión")
        self.canvas_cm.draw()

        # Mostrar informe de clasificación
        self.text_report.delete("1.0", tk.END)
        self.text_report.insert(tk.END, self.metrics_report["Report"])

        # Actualizar estado de botones
        self.btn_train_model.config(state="normal")
        self.btn_save_model.config(state="normal")
        self.lbl_progress.config(text="Entrenamiento completado")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = NeuralNetworkGUI()
    app.run()