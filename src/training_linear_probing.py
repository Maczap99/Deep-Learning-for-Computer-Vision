from pathlib import Path
import time

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"
SEED = 32427
EPOCHEN = 10
LERNRATEN = (0.01,)
batchgroesse = 8

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

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
modell = timm.create_model(
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    pretrained=True,
    pretrained_cfg_overlay={
        "hf_hub_id": f"{CHECKPOINT_ID}@{CHECKPOINT_REVISION}",
    },
)
for parameter in modell.parameters():
    parameter.requires_grad = False
modell.reset_classifier(num_classes=len(KLASSEN_ZU_LABELS))
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
print(f"Modellgerät: {geraetename}")
modell = modell.to(geraet)
head = modell.get_classifier()
if not isinstance(head, torch.nn.Linear):
    raise TypeError(f"Unerwarteter Klassifikationskopf: {type(head)}")
for parameter in head.parameters():
    parameter.requires_grad = True
trainierbare_parameter = [
    (name, parameter)
    for name, parameter in modell.named_parameters()
    if parameter.requires_grad
]
if [name for name, _ in trainierbare_parameter] != ["head.weight", "head.bias"]:
    raise RuntimeError("Unerwartete trainierbare Parameter.")
anzahl_trainierbarer_parameter = sum(
    parameter.numel() for _, parameter in trainierbare_parameter
)
if anzahl_trainierbarer_parameter != 6152:
    raise RuntimeError(
        f"Erwartet wurden 6152 trainierbare Parameter, gefunden wurden "
        f"{anzahl_trainierbarer_parameter}."
    )

vorverarbeitung_config = resolve_model_data_config(modell)
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)
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
verlustfunktion = torch.nn.CrossEntropyLoss()
modell_initialzustand = {
    name: parameter.detach().cpu().clone()
    for name, parameter in head.state_dict().items()
}

if __name__ == "__main__":
    ausgabeordner = repo_ordner / "models" / "linear_probing"
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

    for lernrate in LERNRATEN:
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        head.load_state_dict(modell_initialzustand)
        optimierer = torch.optim.SGD(
            (parameter for _, parameter in trainierbare_parameter),
            lr=lernrate,
        )
        laufordner = laufordner_pro_lernrate[lernrate]
        laufordner.mkdir(exist_ok=False)

        for epoche in range(1, EPOCHEN + 1):
            epoche_start = time.perf_counter()
            modell.eval()
            head.train()
            trainingsverlust_summe = 0.0
            training_richtig = 0
            training_anzahl = 0
            for trainings_bilder, trainings_labels in trainings_dataloader:
                trainings_bilder = trainings_bilder.to(geraet)
                trainings_labels = trainings_labels.to(geraet)
                optimierer.zero_grad(set_to_none=True)
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
            with torch.no_grad():
                for validierungs_bilder, validierungs_labels in validierungs_dataloader:
                    validierungs_bilder = validierungs_bilder.to(geraet)
                    validierungs_labels = validierungs_labels.to(geraet)
                    logits = modell(validierungs_bilder)
                    verlust = verlustfunktion(logits, validierungs_labels)
                    validierungsverlust_summe += (
                        verlust.item() * validierungs_labels.size(0)
                    )
                    validierung_richtig += (
                        logits.argmax(dim=1) == validierungs_labels
                    ).sum().item()
                    validierung_anzahl += validierungs_labels.size(0)

            laufzeit = time.perf_counter() - epoche_start
            trainings_verlust = trainingsverlust_summe / training_anzahl
            trainingstrefferquote = training_richtig / training_anzahl
            validierungs_verlust = validierungsverlust_summe / validierung_anzahl
            validierungstrefferquote = validierung_richtig / validierung_anzahl
            epoche_ausgabeordner = laufordner / f"epoche_{epoche}"
            epoche_ausgabeordner.mkdir(exist_ok=False)
            torch.save(
                {
                    "head_state_dict": {
                        name: parameter.detach().cpu()
                        for name, parameter in head.state_dict().items()
                    },
                    "optimizer_state_dict": optimierer.state_dict(),
                    "checkpoint_id": CHECKPOINT_ID,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                    "klassen_zu_labels": KLASSEN_ZU_LABELS,
                    "vorverarbeitung": vorverarbeitung_config,
                    "batchgroesse": batchgroesse,
                    "optimierer": "SGD",
                    "seed": SEED,
                    "epoche": epoche,
                    "lernrate": lernrate,
                    "trainingsverlust": trainings_verlust,
                    "trainingstrefferquote": trainingstrefferquote,
                    "training_richtige_vorhersagen": training_richtig,
                    "training_anzahl": training_anzahl,
                    "validierungsverlust": validierungs_verlust,
                    "validierungstrefferquote": validierungstrefferquote,
                    "validierung_richtige_vorhersagen": validierung_richtig,
                    "validierung_anzahl": validierung_anzahl,
                    "laufzeit_sekunden": laufzeit,
                },
                epoche_ausgabeordner / "trainingsstand.pt",
            )
            (epoche_ausgabeordner / "ergebnis.txt").write_text(
                f"Epoche: {epoche}\n"
                f"Lernrate: {lernrate}\n"
                f"Trainingsverlust: {trainings_verlust:.6f}\n"
                f"Trainingsgenauigkeit: {trainingstrefferquote:.2%}\n"
                f"Validierungsverlust: {validierungs_verlust:.6f}\n"
                f"Validierungsgenauigkeit: {validierungstrefferquote:.2%}\n"
                f"Laufzeit: {laufzeit:.2f} s\n",
                encoding="utf-8",
            )
            print(
                f"Epoche {epoche} | Lernrate: {lernrate:g} | "
                f"Trainingsverlust: {trainings_verlust:.6f} | "
                f"Validierungsverlust: {validierungs_verlust:.6f} | "
                f"Validierungsgenauigkeit: {validierungstrefferquote:.2%}"
            )
