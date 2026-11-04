from pathlib import Path

ordner = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "Fruits-360-Station-1"
    / "Training"
    / "Apfel"
)

bild_endungen = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

if not ordner.is_dir():
	print("Der angegebene Pfad ist kein gültiger Ordner.")
else:
	anzahl = sum(
		1
		for datei in ordner.iterdir()
		if datei.is_file() and datei.suffix.lower() in bild_endungen
	)
	print(f"Anzahl der Bilder im Ordner: {anzahl}")
