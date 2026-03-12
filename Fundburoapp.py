import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_model():
    # Anstatt tf.keras.models.load_model('model.h5') zu nutzen,
    # definieren wir hier die Architektur manuell.
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(10,)), # Beispiel-Input-Größe: 10
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid') # Für binäre Klassifikation
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 1. Modell initialisieren
model = create_model()
print("Modell wurde erfolgreich im Speicher erstellt (ohne .h5 Datei).")

# 2. Beispiel-Daten (Platzhalter für deine echten Daten)
# Falls du Vorhersagen machen willst, braucht das Modell Gewichte.
# Ohne .h5 Datei sind die Gewichte zufällig, es sei denn, du trainierst kurz:
dummy_data = np.random.random((1, 10))

# 3. Vorhersage oder Training
prediction = model.predict(dummy_data)
print(f"Test-Vorhersage: {prediction}")
