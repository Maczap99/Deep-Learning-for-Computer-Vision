from pathlib import Path
import time

import timm
import torch

from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "resnet18.tv_in1k"

SEED = 32427
batchgroesse = 8

EPOCHEN = 10
LERNRATEN = (0.001,)
# Zusatzversuch vor dem CNN-Kopf (Ergebnisse unter models/Training/cnn_d)
# CNN_DROPOUT = 0.05
ERWARTETE_TRAININGSBILDER = 35713
ERWARTETE_VALIDIERUNGSBILDER = 6309

KLASSEN_ZU_LABELS = {
    "Orange": 0,
    "Mandarine": 1,
    "Clementine": 2,
    "Apfel": 3,
    "Pfirsich": 4,
    "Nektarine": 5,
    "Birne": 6,
    "Banane": 7,
}


class BildDataset(Dataset):
    def __init__(self, bildpfade, labels, vorverarbeitung):
        if len(bildpfade) != len(labels):
            raise ValueError(
                f"Anzahl Bildpfade ({len(bildpfade)}) und Labels "
                f"({len(labels)}) stimmt nicht ueberein."
            )
        self.bildpfade = bildpfade
        self.labels = labels
        self.vorverarbeitung = vorverarbeitung

    def __len__(self):
        return len(self.bildpfade)

    def __getitem__(self, index):
        with Image.open(self.bildpfade[index]) as bild:
            bild_tensor = self.vorverarbeitung(bild.convert("RGB"))
        return bild_tensor, self.labels[index]


LABELS_ZU_KLASSEN = {
    label: klasse for klasse, label in KLASSEN_ZU_LABELS.items()
}


training_zeilen = (
    repo_ordner / "configs" / "training.txt"
).read_text(encoding="utf-8").splitlines()
bildpfade_pro_klasse = {klasse: [] for klasse in KLASSEN_ZU_LABELS}
for zeile in training_zeilen:
    klasse = Path(zeile).parent.name
    if klasse in KLASSEN_ZU_LABELS:
        bildpfad = repo_ordner / zeile
        if bildpfad not in bildpfade_pro_klasse[klasse]:
            bildpfade_pro_klasse[klasse].append(bildpfad)

fehlende_klassen = [
    klasse
    for klasse, bildpfade in bildpfade_pro_klasse.items()
    if not bildpfade
]
if fehlende_klassen:
    raise ValueError(
        f"Keine Trainingsbilder fuer folgende Klassen gefunden: {fehlende_klassen}"
    )

bildpfade = []
labels = []
for bild_index in range(max(map(len, bildpfade_pro_klasse.values()))):
    for klasse, label in KLASSEN_ZU_LABELS.items():
        if bild_index < len(bildpfade_pro_klasse[klasse]):
            bildpfade.append(bildpfade_pro_klasse[klasse][bild_index])
            labels.append(label)

if len(bildpfade) != ERWARTETE_TRAININGSBILDER:
    raise RuntimeError(
        f"Erwartet wurden {ERWARTETE_TRAININGSBILDER} Trainingsbilder, "
        f"gefunden wurden {len(bildpfade)}."
    )

validierung_zeilen = (
    repo_ordner / "configs" / "validierung.txt"
).read_text(encoding="utf-8").splitlines()
validierungspfade_pro_klasse = {klasse: [] for klasse in KLASSEN_ZU_LABELS}
for zeile in validierung_zeilen:
    klasse = Path(zeile).parent.name
    if klasse in KLASSEN_ZU_LABELS:
        bildpfad = repo_ordner / zeile
        if bildpfad not in validierungspfade_pro_klasse[klasse]:
            validierungspfade_pro_klasse[klasse].append(bildpfad)

fehlende_validierungsklassen = [
    klasse
    for klasse, bildpfade in validierungspfade_pro_klasse.items()
    if not bildpfade
]
if fehlende_validierungsklassen:
    raise ValueError(
        "Keine Validierungsbilder fuer folgende Klassen gefunden: "
        f"{fehlende_validierungsklassen}"
    )

validierung_bildpfade = []
validierung_labels = []
for bild_index in range(max(map(len, validierungspfade_pro_klasse.values()))):
    for klasse, label in KLASSEN_ZU_LABELS.items():
        if bild_index < len(validierungspfade_pro_klasse[klasse]):
            validierung_bildpfade.append(
                validierungspfade_pro_klasse[klasse][bild_index]
            )
            validierung_labels.append(label)

if len(validierung_bildpfade) != ERWARTETE_VALIDIERUNGSBILDER:
    raise RuntimeError(
        f"Erwartet wurden {ERWARTETE_VALIDIERUNGSBILDER} Validierungsbilder, "
        f"gefunden wurden {len(validierung_bildpfade)}."
    )

modell_fuer_vorverarbeitung = timm.create_model(
    CHECKPOINT_ID,
    pretrained=True,
)
vorverarbeitung_config = resolve_model_data_config(
    modell_fuer_vorverarbeitung
)
vorverarbeitung = create_transform(
    **vorverarbeitung_config,
    is_training=False,
)

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

trainings_dataset = BildDataset(bildpfade, labels, vorverarbeitung)
validierungs_dataset = BildDataset(
    validierung_bildpfade,
    validierung_labels,
    vorverarbeitung,
)
trainings_dataloader = DataLoader(
    trainings_dataset,
    batch_size=batchgroesse,
    shuffle=True,
)
validierungs_dataloader = DataLoader(
    validierungs_dataset,
    batch_size=batchgroesse,
    shuffle=False,
)

if torch.cuda.is_available():
    geraete_index = next(
        (
            index
            for index in range(torch.cuda.device_count())
            if any(
                gesucht in torch.cuda.get_device_name(index)
                for gesucht in ("RX 6700 XT", "GTX 1070")
            )
        ),
        0,
    )
    geraet = torch.device(f"cuda:{geraete_index}")
    geraetename = torch.cuda.get_device_name(geraete_index).rstrip("\x00")
else:
    geraet = torch.device("cpu")
    geraetename = "CPU"

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

modell = modell_fuer_vorverarbeitung.to(geraet)
modell.fc = torch.nn.Linear(
    modell.fc.in_features,
    len(KLASSEN_ZU_LABELS),
).to(geraet)
for parameter in modell.parameters():
    parameter.requires_grad = False
for parameter in modell.fc.parameters():
    parameter.requires_grad = True
trainierbare_parameter = sum(
    parameter.numel()
    for parameter in modell.parameters()
    if parameter.requires_grad
)
if trainierbare_parameter != 4104:
    raise RuntimeError(
        f"Unerwartete Anzahl trainierbarer Parameter: {trainierbare_parameter}; "
        "erwartet werden 4104."
    )
modell_initialzustand = {
    name: parameter.detach().cpu().clone()
    for name, parameter in modell.state_dict().items()
}
print(f"Modellgerät: {geraetename}")

trainings_bilder, trainings_labels = next(iter(trainings_dataloader))
erwartete_batchform = (batchgroesse, 3, 224, 224)
if tuple(trainings_bilder.shape) != erwartete_batchform:
    raise RuntimeError(
        f"Unerwartete Bildform {tuple(trainings_bilder.shape)}; "
        f"erwartet wird {erwartete_batchform}."
    )
if tuple(trainings_labels.shape) != (batchgroesse,):
    raise RuntimeError(
        f"Unerwartete Label-Form {tuple(trainings_labels.shape)}; "
        f"erwartet wird {(batchgroesse,)}."
    )
if trainings_labels.dtype != torch.int64:
    raise RuntimeError(
        f"Unerwarteter Labeltyp {trainings_labels.dtype}; "
        f"erwartet wird {torch.int64}."
    )

modell.eval()
with torch.no_grad():
    logits = modell(trainings_bilder.to(geraet))

erwartete_ausgabeform = (batchgroesse, len(KLASSEN_ZU_LABELS))
if tuple(logits.shape) != erwartete_ausgabeform:
    raise RuntimeError(
        f"Unerwartete Ausgabeform {tuple(logits.shape)}; "
        f"erwartet wird {erwartete_ausgabeform}."
    )
if not torch.isfinite(logits).all():
    raise RuntimeError("Das Modell hat nicht endliche Klassenscores erzeugt.")

print("Trainingsbilder:", len(trainings_dataset))
print("Validierungsbilder:", len(validierungs_dataset))
print("Trainingsbatch:", tuple(trainings_bilder.shape))
print("Trainingslabels:", tuple(trainings_labels.shape), trainings_labels.dtype)
print("Modellausgabe:", tuple(logits.shape))

if __name__ == "__main__":
    ausgabeordner = repo_ordner / "models" / "cnn"
    laufordner_pro_lernrate = {
        lernrate: ausgabeordner / f"lernrate_{lernrate:g}"
        for lernrate in LERNRATEN
    }
    vorhandene_laufordner = [
        laufordner
        for laufordner in laufordner_pro_lernrate.values()
        if laufordner.exists()
    ]
    if vorhandene_laufordner:
        raise FileExistsError(
            "Ausgabeordner existieren bereits; Ueberschreiben verhindert: "
            f"{vorhandene_laufordner}"
        )
    ausgabeordner.mkdir(parents=True, exist_ok=True)

    verlustfunktion = torch.nn.CrossEntropyLoss()

    for lernrate in LERNRATEN:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        modell.load_state_dict(modell_initialzustand)
        optimierer = torch.optim.AdamW(
            (modell.fc.weight, modell.fc.bias),
            lr=lernrate,
        )
        laufordner = laufordner_pro_lernrate[lernrate]
        laufordner.mkdir(exist_ok=False)

        for epoche in range(1, EPOCHEN + 1):
            epoche_start = time.perf_counter()
            modell.eval()
            modell.fc.train()
            trainingsverlust_summe = 0.0
            training_richtig = 0
            training_anzahl = 0

            for trainings_bilder, trainings_labels in trainings_dataloader:
                trainings_bilder = trainings_bilder.to(geraet)
                trainings_labels = trainings_labels.to(geraet)

                optimierer.zero_grad(set_to_none=True)
                # Variante aus dem Dropout-Versuch:
                # merkmale = modell.forward_features(trainings_bilder)
                # merkmale = modell.forward_head(merkmale, pre_logits=True)
                # merkmale = torch.nn.functional.dropout(
                #     merkmale,
                #     p=CNN_DROPOUT,
                #     training=True,
                # )
                # logits = modell.fc(merkmale)
                logits = modell(trainings_bilder)
                verlust = verlustfunktion(logits, trainings_labels)
                if not torch.isfinite(verlust):
                    raise RuntimeError(f"Ungueltiger Trainingsverlust: {verlust}")

                verlust.backward()
                optimierer.step()

                anzahl = trainings_labels.size(0)
                trainingsverlust_summe += verlust.item() * anzahl
                training_richtig += (
                    logits.argmax(dim=1) == trainings_labels
                ).sum().item()
                training_anzahl += anzahl

            modell.eval()
            validierungsverlust_summe = 0.0
            validierung_richtig = 0
            validierung_anzahl = 0
            validierung_anzahl_pro_klasse = {
                klasse: 0 for klasse in KLASSEN_ZU_LABELS
            }
            validierung_richtig_pro_klasse = {
                klasse: 0 for klasse in KLASSEN_ZU_LABELS
            }

            with torch.no_grad():
                for validierungs_bilder, validierungs_labels in validierungs_dataloader:
                    validierungs_bilder = validierungs_bilder.to(geraet)
                    validierungs_labels = validierungs_labels.to(geraet)
                    logits = modell(validierungs_bilder)
                    verlust = verlustfunktion(logits, validierungs_labels)
                    if not torch.isfinite(verlust):
                        raise RuntimeError(
                            f"Ungueltiger Validierungsverlust: {verlust}"
                        )

                    vorhergesagte_labels = logits.argmax(dim=1)
                    anzahl = validierungs_labels.size(0)
                    validierungsverlust_summe += verlust.item() * anzahl
                    validierung_richtig += (
                        vorhergesagte_labels == validierungs_labels
                    ).sum().item()
                    validierung_anzahl += anzahl

                    for label, vorhersage in zip(
                        validierungs_labels.tolist(),
                        vorhergesagte_labels.tolist(),
                    ):
                        klasse = LABELS_ZU_KLASSEN[label]
                        validierung_anzahl_pro_klasse[klasse] += 1
                        if label == vorhersage:
                            validierung_richtig_pro_klasse[klasse] += 1

            laufzeit = time.perf_counter() - epoche_start
            trainings_verlust = trainingsverlust_summe / training_anzahl
            trainingstrefferquote = training_richtig / training_anzahl
            validierungs_verlust = (
                validierungsverlust_summe / validierung_anzahl
            )
            validierungstrefferquote = (
                validierung_richtig / validierung_anzahl
            )
            validierung_trefferquote_pro_klasse = {
                klasse: (
                    validierung_richtig_pro_klasse[klasse]
                    / validierung_anzahl_pro_klasse[klasse]
                )
                for klasse in KLASSEN_ZU_LABELS
            }

            epoche_ausgabeordner = laufordner / f"epoche_{epoche}"
            epoche_ausgabeordner.mkdir(exist_ok=False)
            torch.save(
                {
                    "klassen_zu_labels": KLASSEN_ZU_LABELS,
                    "epoche": epoche,
                    "lernrate": lernrate,
                    # "cnn_dropout": CNN_DROPOUT,
                    "seed": SEED,
                    "batchgroesse": batchgroesse,
                    "modell_state_dict": {
                        name: parameter.detach().cpu()
                        for name, parameter in modell.state_dict().items()
                    },
                    "optimizer_state_dict": optimierer.state_dict(),
                    "vorverarbeitung": vorverarbeitung_config,
                    "trainingsverlust": trainings_verlust,
                    "trainingstrefferquote": trainingstrefferquote,
                    "training_richtige_vorhersagen": training_richtig,
                    "training_anzahl": training_anzahl,
                    "validierungsverlust": validierungs_verlust,
                    "validierungstrefferquote": validierungstrefferquote,
                    "validierung_richtige_vorhersagen": validierung_richtig,
                    "validierung_anzahl": validierung_anzahl,
                    "validierung_anzahl_pro_klasse": validierung_anzahl_pro_klasse,
                    "validierung_richtige_vorhersagen_pro_klasse": validierung_richtig_pro_klasse,
                    "validierung_trefferquote_pro_klasse": validierung_trefferquote_pro_klasse,
                    "laufzeit_sekunden": laufzeit,
                },
                epoche_ausgabeordner / "trainingsstand.pt",
            )

            epoche_ergebnis = (
                f"Epoche: {epoche}\n"
                f"Lernrate: {lernrate}\n"
                # f"CNN-Dropout: {CNN_DROPOUT}\n"
                f"Trainingsverlust: {trainings_verlust:.6f}\n"
                f"Trainingsgenauigkeit: {trainingstrefferquote:.2%}\n"
                f"Trainingsrichtige Vorhersagen: {training_richtig}/{training_anzahl}\n"
                f"Validierungsverlust: {validierungs_verlust:.6f}\n"
                f"Validierungsgenauigkeit: {validierungstrefferquote:.2%}\n"
                f"Validierungsrichtige Vorhersagen: {validierung_richtig}/{validierung_anzahl}\n"
                f"Validierungstrefferquote je Klasse: {validierung_trefferquote_pro_klasse}\n"
                f"Laufzeit: {laufzeit:.2f} s\n"
            )
            (epoche_ausgabeordner / "ergebnis.txt").write_text(
                epoche_ergebnis,
                encoding="utf-8",
            )

            print(
                f"Epoche {epoche} | "
                f"Lernrate: {lernrate:g} | "
                f"Trainingsverlust: {trainings_verlust:.6f} | "
                f"Trainingsgenauigkeit: {trainingstrefferquote:.2%} | "
                f"Trainingsrichtige Vorhersagen: "
                f"{training_richtig}/{training_anzahl} | "
                f"Validierungsverlust: {validierungs_verlust:.6f} | "
                f"Validierungsgenauigkeit: {validierungstrefferquote:.2%} | "
                f"Validierungsrichtige Vorhersagen: "
                f"{validierung_richtig}/{validierung_anzahl} | "
                f"Laufzeit: {laufzeit:.2f} s"
            )
