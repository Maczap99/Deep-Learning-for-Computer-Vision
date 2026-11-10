from pathlib import Path

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config

repo_ordner = Path(__file__).resolve().parent.parent

CHECKPOINT_ID = "timm/vit_base_patch16_224.augreg_in21k_ft_in1k"
CHECKPOINT_REVISION = "2ec9fb3d7bb664aac471ac44582c94d18de33780"

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
bildpfade_pro_klasse = {}
for zeile in training_zeilen:
    klasse = Path(zeile).parent.name
    if klasse in KLASSEN_ZU_LABELS:
        bildpfade_pro_klasse.setdefault(klasse, repo_ordner / zeile)

fehlende_klassen = KLASSEN_ZU_LABELS.keys() - bildpfade_pro_klasse.keys()
if fehlende_klassen:
    raise ValueError(
        f"Keine Trainingsbilder für folgende Klassen gefunden: {sorted(fehlende_klassen)}"
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

modell.eval()

vorverarbeitung_config = resolve_model_data_config(modell)
vorverarbeitung = create_transform(**vorverarbeitung_config, is_training=False)

bild_tensoren = []
for klasse in KLASSEN_ZU_LABELS:
    with Image.open(bildpfade_pro_klasse[klasse]) as bild:
        rgb_bild = bild.convert("RGB")
        bild_tensoren.append(vorverarbeitung(rgb_bild))

batch = torch.stack(bild_tensoren)
erwartete_form = (len(KLASSEN_ZU_LABELS), *vorverarbeitung_config["input_size"])
if tuple(batch.shape) != erwartete_form or erwartete_form != (8, 3, 224, 224):
    raise RuntimeError(
        f"Unerwartete Tensorform {tuple(batch.shape)}; erwartet wird [8, 3, 224, 224]."
    )

with torch.no_grad():
    logits = modell(batch)

erwartete_ausgabeform = (len(KLASSEN_ZU_LABELS), len(KLASSEN_ZU_LABELS))
if tuple(logits.shape) != erwartete_ausgabeform:
    raise RuntimeError(
        f"Unerwartete Ausgabeform {tuple(logits.shape)}; erwartet wird [8, 8]."
    )

# Test Ausgabe:
# print(f"Checkpoint: {CHECKPOINT_ID}")
# print(f"Geladene Revision: {CHECKPOINT_REVISION}")
# print(f"Eingabegroesse: {vorverarbeitung_config['input_size']}")
# print(f"Normalisierung: Mittelwert={vorverarbeitung_config['mean']}, "f"Standardabweichung={vorverarbeitung_config['std']}")
# print(f"Form des Bild-Batches: {list(batch.shape)}")
# print(f"Trainierbare Parameter: {trainierbare_namen}")
# print(f"Anzahl trainierbarer Parameter: {anzahl_trainierbarer_parameter}")
print(f"Form der Modell-Ausgabe: {list(logits.shape)}")