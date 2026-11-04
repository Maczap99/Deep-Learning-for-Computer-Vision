from pathlib import Path

obst_liste = ["Apfel", "Birne", "Orange"]

ordner_basis = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "Fruits-360-Station-1"
    / "Training"
)

bild_endungen = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

for obst in obst_liste:
	ordner = ordner_basis / obst
	if not ordner.is_dir():
		print(f"Der Ordner '{obst}' ist kein gültiger Ordner.")
	else:
		anzahl = sum(
			1
			for datei in ordner.iterdir()
			if datei.is_file() and datei.suffix.lower() in bild_endungen
		)
		print(f"Anzahl der Bilder in '{obst}': {anzahl}")
