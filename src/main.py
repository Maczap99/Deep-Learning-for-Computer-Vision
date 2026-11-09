from pathlib import Path

import matplotlib.pyplot as plt

ordner_basis = (
    Path(__file__).resolve().parent.parent
    / "Assets"
    / "Fruits-360-Station-1"
    / "Training"
)

bild_endungen = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

gewuenschte_klassen = ["Apfel", "Birne", "Orange", "Mandarine", "Clementine", "Pfirsich", "Nektarine", "Banane"]
ordner_klassen = [ordner_basis / name for name in gewuenschte_klassen]
_, flaechen = plt.subplots(1, 7)

for flaeche, ordner in zip(flaechen, ordner_klassen):
    bild_dateien = sorted(
        datei
        for datei in ordner.iterdir()
        if datei.is_file() and datei.suffix.lower() in bild_endungen
    )
    anzahl = len(bild_dateien)
    print(f"Anzahl der Bilder in '{ordner.name}': {anzahl}")
    if bild_dateien:
        bildpfad = bild_dateien[0]
        print(f"Dateiname: {bildpfad.name}, Label: {bildpfad.parent.name}")
        flaeche.imshow(plt.imread(bildpfad))
        flaeche.set_title(bildpfad.parent.name)
        flaeche.axis("off")

plt.show()