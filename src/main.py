from pathlib import Path

import matplotlib.pyplot as plt

repo_ordner = Path(__file__).resolve().parent.parent

validierung_zeilen = (
    repo_ordner / "configs" / "validierung.txt"
).read_text(encoding="utf-8").splitlines()

bildpfad = repo_ordner / (
    "Assets/Fruits-360-Station-1/Training/Orange/Orange 1__r_100_100.jpg"
)
print(f"Dateiname: {bildpfad.name}, Label: {bildpfad.parent.name}")
_, flaeche = plt.subplots()
flaeche.imshow(plt.imread(bildpfad))
flaeche.set_title(bildpfad.parent.name)
flaeche.axis("off")

plt.show()

gewuenschte_klassen = ["Orange", "Mandarine", "Clementine", "Apfel", "Pfirsich", "Nektarine", "Birne", "Banane"]
ordner_klassen = [ordner_basis / name for name in gewuenschte_klassen]
_, flaechen = plt.subplots(1, len(ordner_klassen), figsize=(15, 5))

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
    print(f"Gesamt für die drei Startklassen: {len(bildpfade)}")
