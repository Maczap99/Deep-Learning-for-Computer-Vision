from pathlib import Path

import matplotlib.pyplot as plt

repo_ordner = Path(__file__).resolve().parent.parent

validierung_zeilen = (
    repo_ordner / "configs" / "validierung.txt"
).read_text(encoding="utf-8").splitlines()
training_zeilen = (
    repo_ordner / "configs" / "training.txt"
).read_text(encoding="utf-8").splitlines()

orange_zeile = next(
    zeile
    for zeile in validierung_zeilen
    if Path(zeile).parent.name == "Orange"
)
bildpfad = repo_ordner / orange_zeile
print(f"Dateiname: {bildpfad.name}, Label: {bildpfad.parent.name}")
_, flaeche = plt.subplots()
flaeche.imshow(plt.imread(bildpfad))
flaeche.set_title(bildpfad.parent.name)
flaeche.axis("off")

plt.show()

gewuenschte_klassen = ["Orange", "Mandarine", "Clementine", "Apfel", "Pfirsich", "Nektarine", "Birne", "Banane"]

for dateiname, zeilen in [
    ("validierung.txt", validierung_zeilen),
    ("training.txt", training_zeilen),
]:
    bildpfade = [
        repo_ordner / zeile
        for zeile in zeilen
        if Path(zeile).parent.name in gewuenschte_klassen
    ]
    print(f"\n{dateiname}:")
    for klasse in gewuenschte_klassen:
        anzahl = sum(bildpfad.parent.name == klasse for bildpfad in bildpfade)
        print(f"{klasse}: {anzahl}")
    print(f"Gesamt für die ausgewählten Klassen: {len(bildpfade)}")
