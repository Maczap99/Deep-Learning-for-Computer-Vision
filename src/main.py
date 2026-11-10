import random
from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"
ZUFALLSSTARTWERT = 42
BILDER_PRO_KLASSE = 10

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
    if len(bildpfade) < BILDER_PRO_KLASSE
]
if fehlende_klassen:
    raise ValueError(
        f"Zu wenige Trainingsbilder fuer folgende Klassen: {fehlende_klassen}"
    )

zufallsgenerator = random.Random(ZUFALLSSTARTWERT)
ausgewaehlte_pfade = {
    klasse: zufallsgenerator.sample(bildpfade, BILDER_PRO_KLASSE)
    for klasse, bildpfade in bildpfade_pro_klasse.items()
}
anzahl_ausgewaehlter_pfade = sum(
    len(bildpfade) for bildpfade in ausgewaehlte_pfade.values()
)
anzahl_eindeutiger_pfade = len(
    {pfad for bildpfade in ausgewaehlte_pfade.values() for pfad in bildpfade}
)
if anzahl_ausgewaehlter_pfade != 80 or anzahl_eindeutiger_pfade != 80:
    raise RuntimeError(
        f"Erwartet wurden 80 verschiedene Pfade, gefunden wurden "
        f"{anzahl_ausgewaehlter_pfade} ({anzahl_eindeutiger_pfade} eindeutig)."
    )

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
head = modell.get_classifier()
if not isinstance(head, torch.nn.Linear):
    raise TypeError(f"Erwartet wurde ein linearer Klassifikationskopf, erhalten: {type(head)}")
if (head.in_features, head.out_features, head.bias is not None) != (768, 8, True):
    raise RuntimeError(
        "Unerwartete Kopfkonfiguration: "
        f"in_features={head.in_features}, out_features={head.out_features}, "
        f"bias={head.bias is not None}"
    )

trainierbare_parameter = [
    (name, parameter)
    for name, parameter in modell.named_parameters()
    if parameter.requires_grad
]
trainierbare_namen = [name for name, _ in trainierbare_parameter]
anzahl_trainierbarer_parameter = sum(
    parameter.numel() for _, parameter in trainierbare_parameter
)
if trainierbare_namen != ["head.weight", "head.bias"]:
    raise RuntimeError(
        f"Unerwartete trainierbare Parameter: {trainierbare_namen}"
    )
if anzahl_trainierbarer_parameter != 6152:
    raise RuntimeError(
        f"Erwartet wurden 6152 trainierbare Parameter, "
        f"gefunden wurden {anzahl_trainierbarer_parameter}."
    )

optimierer = torch.optim.SGD(
    (parameter for _, parameter in trainierbare_parameter),
    lr=0.01,
)
verlustfunktion = torch.nn.CrossEntropyLoss()

modell.eval()

vorverarbeitung_config = resolve_model_data_config(modell)
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)

bild_tensoren = []
labels = []
for bild_index in range(BILDER_PRO_KLASSE):
    for klasse, label in KLASSEN_ZU_LABELS.items():
        with Image.open(ausgewaehlte_pfade[klasse][bild_index]) as bild:
            rgb_bild = bild.convert("RGB")
            bild_tensoren.append(vorverarbeitung(rgb_bild))
            labels.append(label)

if len(bild_tensoren) != len(labels):
    raise RuntimeError(
        f"Anzahl Bilder ({len(bild_tensoren)}) und Labels ({len(labels)}) stimmt nicht ueberein."
    )

bild_batch = torch.stack(bild_tensoren)
labels_batch = torch.tensor(labels, dtype=torch.long)
erwartete_form = (
    BILDER_PRO_KLASSE * len(KLASSEN_ZU_LABELS),
    *vorverarbeitung_config["input_size"],
)
if tuple(bild_batch.shape) != erwartete_form or erwartete_form != (80, 3, 224, 224):
    raise RuntimeError(
        f"Unerwartete Tensorform {tuple(bild_batch.shape)}; "
        "erwartet wird [80, 3, 224, 224]."
    )

head_vorher = {
    name: parameter.detach().clone()
    for name, parameter in trainierbare_parameter
}
batchgroesse = len(KLASSEN_ZU_LABELS)
verluste = []
for batch_start in range(0, len(bild_batch), batchgroesse):
    batch = bild_batch[batch_start : batch_start + batchgroesse]
    batch_labels = labels_batch[batch_start : batch_start + batchgroesse]

    optimierer.zero_grad(set_to_none=True)
    logits = modell(batch)
    erwartete_ausgabeform = (len(batch), len(KLASSEN_ZU_LABELS))
    if tuple(logits.shape) != erwartete_ausgabeform:
        raise RuntimeError(
            f"Unerwartete Ausgabeform {tuple(logits.shape)}; "
            f"erwartet wird {erwartete_ausgabeform}."
        )

    verlust = verlustfunktion(logits, batch_labels)
    if verlust.ndim != 0 or not torch.isfinite(verlust):
        raise RuntimeError(f"Ungueltiger Verlust: {verlust}")

    verlust.backward()
    if any(parameter.grad is None for _, parameter in trainierbare_parameter):
        raise RuntimeError(
            "Mindestens ein trainierbarer Kopfparameter hat keinen Gradienten."
        )
    if any(
        parameter.grad is not None
        for parameter in modell.parameters()
        if not parameter.requires_grad
    ):
        raise RuntimeError(
            "Ein eingefrorener ViT-Parameter hat unerwartet einen Gradienten."
        )

    optimierer.step()
    verluste.append(verlust.item())

head_danach = {
    name: not torch.equal(head_vorher[name], parameter.detach())
    for name, parameter in trainierbare_parameter
}
if not all(head_danach.values()):
    raise RuntimeError(f"Nicht alle Headparameter wurden aktualisiert: {head_danach}")

print(f"Bilder: {len(bild_batch)}")
print(f"Batches: {len(verluste)}")
print(f"Batchgroesse: {batchgroesse}")
print(f"Ausgabe pro Batch: {tuple(logits.shape)}")
print(f"Mittlerer Verlust: {sum(verluste) / len(verluste):.4f}")
print(f"Trainierbare Parameter: {anzahl_trainierbarer_parameter}")
print(f"Head aktualisiert: {all(head_danach.values())}")