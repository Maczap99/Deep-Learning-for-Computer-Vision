import os
from pathlib import Path

training = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "Fruits-360-Station-1"
    / "Training"
)

klassen = ["Apfel", "Birne", "Orange"]
pfade = ["Training/Apfel", "Training/Birne", "Training/Orange"]
bild_endungen = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

for klasse in klassen:
    ordner = training / klasse
    
    if not ordner.is_dir():
        print(f"{klasse}: Ordner nicht gefunden.")
    else:
        bilder = [
            datei
            for datei in ordner.iterdir()
            if datei.is_file() and datei.suffix.lower() in bild_endungen
        ]

        print(f"{klasse}: {len(bilder)} Bilder")

        if bilder:  
            datei = bilder[0]  # Bild auswählen 

            print("Bild:", datei.name)
            print("Label:", datei.parent.name)

            os.startfile(datei)
