import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


PRIMARY_COLOR = "#1832a2"
SECONDARY_COLOR = "#aa1717"
BACKGROUND_COLOR = "#d9d9d9"
TEXT_COLOR = "#333333"
ACCENT_COLOR = "#467be3"

root = tk.Tk()
root.title("Red Neuronal - Clasificación")
root.geometry("1000x800")
root.configure(bg=BACKGROUND_COLOR)

style = ttk.Style(root)
style.theme_use('clam')
style.configure('.', background=BACKGROUND_COLOR, foreground=TEXT_COLOR, font=('Segoe UI', 10))
style.configure('TButton', background=PRIMARY_COLOR, foreground='white', font=('Segoe UI', 10, 'bold'))
style.map('TButton', background=[('active', ACCENT_COLOR)])

main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill=tk.BOTH, expand=True)

config_frame = ttk.LabelFrame(main_frame, text="Configuración de Entrenamiento", padding=15)
config_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

epochs_var = tk.IntVar(value=100)
folds_var = tk.IntVar(value=5)
learning_rate_var = tk.DoubleVar(value=0.01)
data = None
results_data = [] # To store results for Treeview
confusion_matrix_figure = None # To store the confusion matrix figure
confusion_window = None # To store the confusion matrix window


file_label = ttk.Label(config_frame, text="Cargar dataset (CSV):")
file_label.grid(row=0, column=0, padx=5, pady=5, sticky="e")

status_label = ttk.Label(config_frame, text="Archivo no seleccionado")
status_label.grid(row=0, column=3, padx=5, pady=5, sticky="w")

def load_csv_file():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("Archivos CSV", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path, header=None, delimiter=";") # Assuming no header and semicolon delimiter as per original description
            status_label.config(text="Dataset cargado exitosamente", foreground="green")
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar el archivo: {e}")
            status_label.config(text="Error al cargar el archivo", foreground="red")
            data = None # Reset data in case of error
    else:
        status_label.config(text="Archivo no seleccionado", foreground="red")
        data = None # Reset data if no file selected


def train_model():
    global data, results_data, confusion_matrix_figure, confusion_window

    if data is None:
        messagebox.showerror("Error", "No se ha cargado un dataset. Por favor, carga un archivo CSV.")
        return

    results_data = [] # Clear previous results
    results_tree.delete(*results_tree.get_children()) # Clear Treeview

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Codificación de etiquetas si es necesario
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)

    epochs = epochs_var.get()
    folds = folds_var.get()
    learning_rate = learning_rate_var.get()

    kf = KFold(n_splits=folds, shuffle=True, random_state=42) # Good practice to set a random_state for reproducibility

    fold_number = 1
    all_y_true = []
    all_y_pred = []

    for train_index, test_index in kf.split(X, y_encoded):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Definir modelo de red neuronal (reset model for each fold for independent training)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax') # Use num_classes for output layer
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) # Use learning rate from GUI
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Entrenar modelo
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0) # Reduced batch_size and silent training

        # Evaluación
        y_pred_probs = model.predict(X_test, verbose=0) # Silent prediction
        y_pred = np.argmax(y_pred_probs, axis=1)

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0) # Handle zero division
        accuracy = report['accuracy']
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1_score = report['macro avg']['f1-score']

        results_data.append({
            "Fold": fold_number,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1_score
        })
        results_tree.insert("", "end", values=(fold_number, f"{accuracy:.4f}", f"{precision:.4f}", f"{recall:.4f}", f"{f1_score:.4f}"))

        fold_number += 1

    # Calculate confusion matrix after all folds
    cm = confusion_matrix(all_y_true, all_y_pred)

    # Display Confusion Matrix in a new Toplevel window
    if confusion_window: # Check if window exists and destroy if it does to create a new one
        confusion_window.destroy()
    confusion_window = Toplevel(root) # Create a new Toplevel window
    confusion_window.title("Matriz de Confusión")

    confusion_matrix_figure = plt.figure(figsize=(8, 6)) # Create a new figure
    ax = confusion_matrix_figure.add_subplot(111)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm) # No class names needed for numerical labels
    cm_display.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_xlabel("Y Calculada") # Set X axis label to "Y Calculada"
    ax.set_ylabel("Y Deseada") # Set Y axis label to "Y Deseada"
    ax.set_title("Matriz de Confusión (Promedio)")

    canvas_cm = FigureCanvasTkAgg(confusion_matrix_figure, master=confusion_window) # Embed in confusion_window
    canvas_cm_widget = canvas_cm.get_tk_widget()
    canvas_cm_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
    canvas_cm.draw() # Draw the plot

    messagebox.showinfo("Éxito", "Entrenamiento completado y resultados mostrados. La matriz de confusión se muestra en una ventana separada.")


load_button = ttk.Button(config_frame, text="Cargar dataset", command=load_csv_file)
load_button.grid(row=0, column=1, padx=10, pady=5)

epochs_label = ttk.Label(config_frame, text="Épocas:")
epochs_label.grid(row=1, column=0, padx=5, pady=5, sticky="e")
epochs_entry = ttk.Entry(config_frame, textvariable=epochs_var, width=10)
epochs_entry.grid(row=1, column=1, padx=5, pady=5)

folds_label = ttk.Label(config_frame, text="Folds (K-Fold CV):")
folds_label.grid(row=2, column=0, padx=5, pady=5, sticky="e")
folds_entry = ttk.Entry(config_frame, textvariable=folds_var, width=10)
folds_entry.grid(row=2, column=1, padx=5, pady=5)

learning_rate_label = ttk.Label(config_frame, text="Tasa de aprendizaje:")
learning_rate_label.grid(row=3, column=0, padx=5, pady=5, sticky="e")
learning_rate_entry = ttk.Entry(config_frame, textvariable=learning_rate_var, width=10)
learning_rate_entry.grid(row=3, column=1, padx=5, pady=5)

train_button = ttk.Button(config_frame, text="Iniciar Entrenamiento", command=train_model) # Corrected command
train_button.grid(row=1, column=2, padx=10, pady=5, rowspan=3)

results_frame = ttk.LabelFrame(main_frame, text="Resultados del Entrenamiento", padding=15)
results_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)

results_tree = ttk.Treeview(results_frame, columns=("Fold", "Accuracy", "Precision", "Recall", "F1-Score"), show="headings")
for col in ["Fold", "Accuracy", "Precision", "Recall", "F1-Score"]:
    results_tree.heading(col, text=col)
    results_tree.column(col, width=100, anchor='center')
results_tree.pack(fill=tk.BOTH, expand=True, pady=(0, 10)) # Add some padding below the treeview

root.mainloop()