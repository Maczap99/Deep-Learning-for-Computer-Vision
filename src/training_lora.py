from pathlib import Path
import time

import timm
import torch

from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from timm.data import create_transform, resolve_model_data_config
from torch.utils.data import DataLoader, Dataset

repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"

SEED = 32427
EPOCHEN = 10
LERNRATEN = (0.0001,)
ALTER_STAND = ""
batchgroesse = 8

LORA_RANG = 16
LORA_ALPHA = 2 * LORA_RANG
LORA_DROPOUT = 0.05
LORA_ZIELMODULE = ["attn.qkv", "attn.proj"]

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

modell = timm.create_model(
    "vit_base_patch16_224.augreg_in21k_ft_in1k",
    pretrained=False,
)
vorverarbeitung_config = resolve_model_data_config(modell)
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)
del modell

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

lora_config = LoraConfig(
    r=LORA_RANG,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_ZIELMODULE,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    modules_to_save=["head"]
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
print(f"Modellgerät: {geraetename}")

verlustfunktion = torch.nn.CrossEntropyLoss()

if __name__ == "__main__":
    ausgabeordner = (
        repo_ordner
        / "models"
        / "lora"
        / f"rang_{LORA_RANG}"
    )
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
        modell = timm.create_model(
            "vit_base_patch16_224.augreg_in21k_ft_in1k",
            pretrained=True,
            pretrained_cfg_overlay={
                "hf_hub_id": f"{CHECKPOINT_ID}@{CHECKPOINT_REVISION}",
            },
        )

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        modell.reset_classifier(num_classes=len(KLASSEN_ZU_LABELS))

        anzahl_lora_parameter_pro_rang = sum(
            modul.in_features + modul.out_features
            for name, modul in modell.named_modules()
            if isinstance(modul, torch.nn.Linear)
            and any(name.endswith(zielmodul) for zielmodul in LORA_ZIELMODULE)
        )
        anzahl_kopfparameter = sum(
            parameter.numel()
            for parameter in modell.get_classifier().parameters()
        )
        erwartete_anzahl_trainierbarer_parameter = (
            LORA_RANG * anzahl_lora_parameter_pro_rang
            + anzahl_kopfparameter
        )

        if ALTER_STAND:
            alter_standpfad = repo_ordner / "models" / ALTER_STAND
            if not alter_standpfad.is_dir():
                raise FileNotFoundError(
                    f"Alter Stand nicht gefunden: {alter_standpfad}"
                )

            trainingsstandspfad = alter_standpfad / "trainingsstand.pt"
            if not trainingsstandspfad.is_file():
                raise FileNotFoundError(
                    f"Trainingsstand nicht gefunden: {trainingsstandspfad}"
                )

            gespeichert = torch.load(
                trainingsstandspfad,
                map_location="cpu",
                weights_only=True,
            )
            if gespeichert["checkpoint_id"] != CHECKPOINT_ID:
                raise RuntimeError(
                    "Checkpoint-ID stimmt nicht mit dem Fortsetzungsstand ueberein."
                )
            if gespeichert["checkpoint_revision"] != CHECKPOINT_REVISION:
                raise RuntimeError(
                    "Checkpoint-Revision stimmt nicht mit dem Fortsetzungsstand ueberein."
                )
            if gespeichert["lora_rang"] != LORA_RANG:
                raise RuntimeError(
                    "LoRA-Rang stimmt nicht mit dem Fortsetzungsstand ueberein."
                )
            if gespeichert["lernrate"] != lernrate:
                raise RuntimeError(
                    "Lernrate stimmt nicht mit dem Fortsetzungsstand ueberein."
                )
            if gespeichert["klassen_zu_labels"] != KLASSEN_ZU_LABELS:
                raise RuntimeError(
                    "Klassen-Zuordnung stimmt nicht mit dem Fortsetzungsstand ueberein."
                )

            print("Geladener Trainingsstand:", trainingsstandspfad)
            modell = PeftModel.from_pretrained(
                modell,
                alter_standpfad,
                is_trainable=True,
            )
            startepoche = gespeichert["epoche"] + 1
        else:
            modell = get_peft_model(modell, lora_config)
            startepoche = 1

        modell = modell.to(geraet)

        lora_modulnamen = []
        for name, modul in modell.named_modules():
            if hasattr(modul, "lora_A"):
                lora_modulnamen.append(name)

        print(f"Anzahl LoRA Adaptermodule: {len(lora_modulnamen)}")
        if len(lora_modulnamen) != 24:
            raise RuntimeError(
                f"Erwartet wurden 24 LoRA Adaptermodule, gefunden wurden "
                f"{len(lora_modulnamen)}."
            )
        if any("patch_embed.proj" in name for name in lora_modulnamen):
            raise RuntimeError(
                "patch_embed.proj darf keinen LoRA Adapter enthalten."
            )

        for block in range(12):
            for zielmodul in ("qkv", "proj"):
                suffix = f"blocks.{block}.attn.{zielmodul}"
                treffer = [
                    name
                    for name in lora_modulnamen
                    if name.endswith(suffix)
                ]
                if len(treffer) != 1:
                    raise RuntimeError(
                        f"Erwartet wurde genau ein Adapter fuer {suffix}, "
                        f"gefunden wurden {len(treffer)}: {treffer}"
                    )

        trainierbare_parameter = [
            (name, parameter)
            for name, parameter in modell.named_parameters()
            if parameter.requires_grad
        ]
        anzahl_trainierbarer_parameter = sum(
            parameter.numel()
            for _, parameter in trainierbare_parameter
        )

        print(
            f"Anzahl trainierbarer Parameter: "
            f"{anzahl_trainierbarer_parameter}"
        )
        if (
            anzahl_trainierbarer_parameter
            != erwartete_anzahl_trainierbarer_parameter
        ):
            raise RuntimeError(
                "Erwartet wurden "
                f"{erwartete_anzahl_trainierbarer_parameter} "
                f"trainierbare Parameter, gefunden wurden "
                f"{anzahl_trainierbarer_parameter}."
            )

        optimierer = torch.optim.AdamW(
            [parameter for _, parameter in trainierbare_parameter],
            lr=lernrate,
        )

        if ALTER_STAND:
            optimierer.load_state_dict(
                gespeichert["optimizer_state_dict"]
            )
            if "torch_rng_state" in gespeichert:
                torch.set_rng_state(gespeichert["torch_rng_state"].cpu())
            if "cuda_rng_state" in gespeichert and torch.cuda.is_available():
                torch.cuda.set_rng_state(
                    gespeichert["cuda_rng_state"].cpu(),
                    geraete_index,
                )

        laufordner = laufordner_pro_lernrate[lernrate]
        laufordner.mkdir(exist_ok=False)

        for epoche in range(startepoche, EPOCHEN + 1):
            epoche_start = time.perf_counter()
            modell.train()
            trainingsverlust_summe = 0.0
            training_richtig = 0
            training_anzahl = 0

            for trainings_bilder, trainings_labels in trainings_dataloader:
                trainings_bilder = trainings_bilder.to(geraet)
                trainings_labels = trainings_labels.to(geraet)

                optimierer.zero_grad(set_to_none=True)
                logits = modell(trainings_bilder)
                verlust = verlustfunktion(logits, trainings_labels)
                verlust.backward()
                optimierer.step()

                trainingsverlust_summe += (
                    verlust.item() * trainings_labels.size(0)
                )
                training_richtig += (
                    logits.argmax(dim=1) == trainings_labels
                ).sum().item()
                training_anzahl += trainings_labels.size(0)

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
                    vorhergesagte_labels = logits.argmax(dim=1)

                    validierungsverlust_summe += (
                        verlust.item() * validierungs_labels.size(0)
                    )
                    validierung_richtig += (
                        vorhergesagte_labels == validierungs_labels
                    ).sum().item()
                    validierung_anzahl += validierungs_labels.size(0)

            laufzeit = time.perf_counter() - epoche_start
            trainings_verlust = (
                trainingsverlust_summe / training_anzahl
            )
            trainingstrefferquote = (
                training_richtig / training_anzahl
            )
            validierungs_verlust = (
                validierungsverlust_summe / validierung_anzahl
            )
            validierungstrefferquote = (
                validierung_richtig / validierung_anzahl
            )

            epoche_ausgabeordner = laufordner / f"epoche_{epoche}"
            epoche_ausgabeordner.mkdir(exist_ok=False)
            modell.save_pretrained(epoche_ausgabeordner)

            zufallszustaende = {
                "torch_rng_state": torch.get_rng_state(),
            }
            if torch.cuda.is_available():
                zufallszustaende["cuda_rng_state"] = (
                    torch.cuda.get_rng_state(geraete_index)
                )

            torch.save(
                {
                    "checkpoint_id": CHECKPOINT_ID,
                    "checkpoint_revision": CHECKPOINT_REVISION,
                    "klassen_zu_labels": KLASSEN_ZU_LABELS,
                    "lora_rang": LORA_RANG,
                    "lora_alpha": LORA_ALPHA,
                    "lora_dropout": LORA_DROPOUT,
                    "seed": SEED,
                    "batchgroesse": batchgroesse,
                    "epoche": epoche,
                    "lernrate": lernrate,
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
                    "laufzeit_sekunden": laufzeit,
                    **zufallszustaende,
                },
                epoche_ausgabeordner / "trainingsstand.pt",
            )

            epoche_ergebnis = (
                f"Epoche: {epoche}\n"
                f"Lernrate: {lernrate}\n"
                f"Trainingsverlust: {trainings_verlust:.6f}\n"
                f"Trainingsgenauigkeit: {trainingstrefferquote:.2%}\n"
                f"Trainingsrichtige Vorhersagen: "
                f"{training_richtig}/{training_anzahl}\n"
                f"Trainingsbildanzahl: {training_anzahl}\n"
                f"Validierungsverlust: {validierungs_verlust:.6f}\n"
                f"Validierungsgenauigkeit: {validierungstrefferquote:.2%}\n"
                f"Validierungsrichtige Vorhersagen: "
                f"{validierung_richtig}/{validierung_anzahl}\n"
                f"Validierungsbildanzahl: {validierung_anzahl}\n"
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
