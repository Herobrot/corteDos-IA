import csv
import random

def generar_csv_confusion_matrix(nombre_archivo, n_muestras=100):
    # Lista de etiquetas a utilizar
    etiquetas = ["Perro", "Gato", "Perico", "León", "Tigre", "Paloma", "Gusano", "Hormiga"]

    # Abrir (o crear) el archivo CSV en modo escritura
    with open(nombre_archivo, mode='w', newline='', encoding='utf-8') as archivo:
        writer = csv.writer(archivo)
        # Escribir la cabecera del archivo
        writer.writerow(["Etiqueta_Real", "Etiqueta_Predicha"])

        for _ in range(n_muestras):
            # Seleccionar una etiqueta real aleatoria
            etiqueta_real = random.choice(etiquetas)
            # Con probabilidad del 70% la predicción es correcta
            if random.random() < 0.7:
                etiqueta_predicha = etiqueta_real
            else:
                # En caso de error, se escoge una etiqueta diferente a la real
                etiqueta_predicha = random.choice([et for et in etiquetas if et != etiqueta_real])
            # Escribir la fila en el CSV
            writer.writerow([etiqueta_real, etiqueta_predicha])

if __name__ == "__main__":
    generar_csv_confusion_matrix("confusion_matrix_test.csv", n_muestras=100)
    print("Archivo 'confusion_matrix_test.csv' generado exitosamente.")
