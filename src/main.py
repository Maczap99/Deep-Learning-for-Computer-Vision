from pathlib import Path

ordner_basis = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "Fruits-360-Station-1"
    / "Training"
)

bild_endungen = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}

for ordner in sorted(ordner_basis.iterdir()):
	if ordner.is_dir():
		anzahl = sum(
			1
			for datei in ordner.iterdir()
			if datei.is_file() and datei.suffix.lower() in bild_endungen
		)
		print(f"Anzahl der Bilder in '{ordner.name}': {anzahl}")
