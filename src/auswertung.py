import csv
from pathlib import Path

import timm
import torch
from peft import PeftModel
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset


repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"
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
LABELS_ZU_KLASSEN = {
    label: klasse for klasse, label in KLASSEN_ZU_LABELS.items()
}

MODELLPFADE = {
    "linear_probing": (
        repo_ordner
        / "models"
        / "linear_probing"
        / "lernrate_0.01"
        / "epoche_7"
    ),
    "lora": (
        repo_ordner
        / "models"
        / "lora"
        / "rang_16"
        / "lernrate_0.0001"
        / "epoche_8"
    ),
    "cnn": (
        repo_ordner
        / "models"
        / "cnn"
        / "lernrate_0.001"
        / "epoche_8"
    ),
}

ERWARTETE_VALIDIERUNGSBILDER = 6309
ERWARTETE_TESTBILDER = 13982


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
        return bild_tensor, self.labels[index], str(self.bildpfade[index])


if __name__ == "__main__":
    validierung_zeilen = (
        repo_ordner / "configs" / "validierung.txt"
    ).read_text(encoding="utf-8").splitlines()
    validierungspfade_pro_klasse = {
        klasse: [] for klasse in KLASSEN_ZU_LABELS
    }
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
    for bild_index in range(
        max(map(len, validierungspfade_pro_klasse.values()))
    ):
        for klasse, label in KLASSEN_ZU_LABELS.items():
            if bild_index < len(validierungspfade_pro_klasse[klasse]):
                validierung_bildpfade.append(
                    validierungspfade_pro_klasse[klasse][bild_index]
                )
                validierung_labels.append(label)

    if len(validierung_bildpfade) != ERWARTETE_VALIDIERUNGSBILDER:
        raise RuntimeError(
            f"Erwartet wurden {ERWARTETE_VALIDIERUNGSBILDER} "
            f"Validierungsbilder, gefunden wurden "
            f"{len(validierung_bildpfade)}."
        )

    testordner = (
        repo_ordner / "Assets" / "Fruits-360-Station-1" / "Test"
    )
    testpfade_pro_klasse = {klasse: [] for klasse in KLASSEN_ZU_LABELS}
    for klasse in KLASSEN_ZU_LABELS:
        klassenordner = testordner / klasse
        if not klassenordner.is_dir():
            raise FileNotFoundError(
                f"Klassenordner nicht gefunden: {klassenordner}"
            )
        testpfade_pro_klasse[klasse] = sorted(
            pfad for pfad in klassenordner.iterdir() if pfad.is_file()
        )

    test_bildpfade = []
    test_labels = []
    for bild_index in range(max(map(len, testpfade_pro_klasse.values()))):
        for klasse, label in KLASSEN_ZU_LABELS.items():
            if bild_index < len(testpfade_pro_klasse[klasse]):
                test_bildpfade.append(testpfade_pro_klasse[klasse][bild_index])
                test_labels.append(label)

    if len(test_bildpfade) != ERWARTETE_TESTBILDER:
        raise RuntimeError(
            f"Erwartet wurden {ERWARTETE_TESTBILDER} Testbilder, "
            f"gefunden wurden {len(test_bildpfade)}."
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
        geraetename = torch.cuda.get_device_name(
            geraete_index
        ).rstrip("\x00")
    else:
        geraet = torch.device("cpu")
        geraetename = "CPU"
    print(f"Modellgerät: {geraetename}")

    for modellname in MODELLPFADE:
        speicherpfad = MODELLPFADE[modellname]
        trainingsstand = speicherpfad / "trainingsstand.pt"
        if not trainingsstand.is_file():
            raise FileNotFoundError(
                f"Checkpoint nicht gefunden: {trainingsstand}"
            )
        gespeichert = torch.load(
            trainingsstand,
            map_location="cpu",
            weights_only=True,
        )

        if modellname != "cnn":
            if gespeichert.get("checkpoint_id") != CHECKPOINT_ID:
                raise RuntimeError("Checkpoint-ID stimmt nicht ueberein.")
            if gespeichert.get("checkpoint_revision") != CHECKPOINT_REVISION:
                raise RuntimeError("Checkpoint-Revision stimmt nicht ueberein.")
        if gespeichert.get("klassen_zu_labels") != KLASSEN_ZU_LABELS:
            raise RuntimeError(
                "Klassen-Zuordnung im Checkpoint stimmt nicht mit dem "
                "Auswertungscode ueberein."
            )

        if modellname == "cnn":
            modell = timm.create_model(
                "resnet18.tv_in1k",
                pretrained=True,
            )
            modell.fc = torch.nn.Linear(
                modell.fc.in_features,
                len(KLASSEN_ZU_LABELS),
            )
            modell.load_state_dict(gespeichert["modell_state_dict"])
        else:
            modell = timm.create_model(
                "vit_base_patch16_224.augreg_in21k_ft_in1k",
                pretrained=True,
                pretrained_cfg_overlay={
                    "hf_hub_id": f"{CHECKPOINT_ID}@{CHECKPOINT_REVISION}",
                },
            )
            modell.reset_classifier(
                num_classes=len(KLASSEN_ZU_LABELS)
            )

            if modellname == "linear_probing":
                head = modell.get_classifier()
                head.load_state_dict(gespeichert["head_state_dict"])
            else:
                modell = PeftModel.from_pretrained(
                    modell,
                    speicherpfad,
                    is_trainable=False,
                )

        vorverarbeitung_config = resolve_model_data_config(modell)
        if vorverarbeitung_config != gespeichert["vorverarbeitung"]:
            raise RuntimeError(
                f"Vorverarbeitung fuer {modellname} stimmt nicht ueberein."
            )
        vorverarbeitung = create_transform(
            **vorverarbeitung_config,
            is_training=False,
        )
        modell = modell.to(geraet)
        modell.eval()

        for datenname, bildpfade, labels in (
            ("validierung", validierung_bildpfade, validierung_labels),
            ("test", test_bildpfade, test_labels),
        ):
            dataset = BildDataset(bildpfade, labels, vorverarbeitung)
            dataloader = DataLoader(
                dataset,
                batch_size=gespeichert.get("batchgroesse", batchgroesse),
                shuffle=False,
            )

            anzahl_klassen = len(KLASSEN_ZU_LABELS)
            bildzahl_pro_klasse = [0] * anzahl_klassen
            richtige_vorhersagen_pro_klasse = [0] * anzahl_klassen
            verwechslungsmatrix = [
                [0] * anzahl_klassen for _ in range(anzahl_klassen)
            ]
            fehlerliste = []

            with torch.no_grad():
                for batch, batch_labels, batch_pfade in dataloader:
                    batch = batch.to(geraet)
                    batch_labels = batch_labels.to(geraet)
                    logits = modell(batch)
                    vorhergesagte_labels = logits.argmax(dim=1)

                    for tatsaechliches_label, vorhergesagtes_label, bildpfad in zip(
                        batch_labels.cpu().tolist(),
                        vorhergesagte_labels.cpu().tolist(),
                        batch_pfade,
                    ):
                        bildzahl_pro_klasse[tatsaechliches_label] += 1
                        verwechslungsmatrix[tatsaechliches_label][
                            vorhergesagtes_label
                        ] += 1
                        if tatsaechliches_label == vorhergesagtes_label:
                            richtige_vorhersagen_pro_klasse[
                                tatsaechliches_label
                            ] += 1
                        else:
                            fehlerliste.append(
                                {
                                    "bildpfad": Path(bildpfad)
                                    .relative_to(repo_ordner)
                                    .as_posix(),
                                    "tatsaechlich": LABELS_ZU_KLASSEN[
                                        tatsaechliches_label
                                    ],
                                    "vorhergesagt": LABELS_ZU_KLASSEN[
                                        vorhergesagtes_label
                                    ],
                                }
                            )

            daten_anzahl = sum(bildzahl_pro_klasse)
            richtig = sum(richtige_vorhersagen_pro_klasse)

            print(f"\n{modellname} - {datenname}")
            print("Checkpoint:", speicherpfad)
            print("Bilder:", daten_anzahl)
            print("Richtige Vorhersagen:", richtig)
            print("Fehler:", len(fehlerliste))
            print(f"Trefferquote: {richtig / daten_anzahl:.2%}")

            ergebnisordner = (
                repo_ordner / "results" / datenname / modellname
            )
            ergebnisordner.mkdir(parents=True, exist_ok=True)

            fehlerdatei = ergebnisordner / "fehlerbilder.csv"
            with fehlerdatei.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as datei:
                writer = csv.writer(datei)
                writer.writerow(
                    ["bildpfad", "tatsaechlich", "vorhergesagt"]
                )
                for fehler in fehlerliste:
                    writer.writerow(
                        [
                            fehler["bildpfad"],
                            fehler["tatsaechlich"],
                            fehler["vorhergesagt"],
                        ]
                    )

            matrixdatei = ergebnisordner / "verwechslungsmatrix.csv"
            with matrixdatei.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as datei:
                writer = csv.writer(datei)
                writer.writerow(
                    ["Original / Vorhergesagt", *LABELS_ZU_KLASSEN.values()]
                )
                for label, zeile in enumerate(verwechslungsmatrix):
                    writer.writerow(
                        [LABELS_ZU_KLASSEN[label], *zeile]
                    )

            print("Ergebnisse gespeichert:")
            print(fehlerdatei)
            print(matrixdatei)
