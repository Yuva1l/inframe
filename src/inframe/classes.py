"""Object classes the detector knows: the COCO category ids the RF-DETR COCO checkpoints use (91 slots, 80 of them named; the ids have gaps),
`--target class:NAME` parsing, and which model size a class needs.

The table is the official COCO one. tests/test_classes.py checks every row against the table the rfdetr package ships (when it is
installed); against the real model the logit slot of the named class fires on the dog and car clips, the person slot does not."""
from __future__ import annotations

PERSON = 1

COCO_91 = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
    10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
    19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella",
    31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
    46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch",
    64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}
NAME_TO_ID = {n: i for i, n in COCO_91.items()}
ALIASES = {"people": "person", "human": "person", "bike": "bicycle", "motorbike": "motorcycle", "plane": "airplane", "aeroplane": "airplane",
           "ball": "sports ball", "phone": "cell phone", "mobile phone": "cell phone", "television": "tv", "sofa": "couch",
           "plant": "potted plant", "table": "dining table", "hairdryer": "hair drier", "hair dryer": "hair drier", "puppy": "dog",
           "kitten": "cat", "lorry": "truck"}


class UnknownClass(ValueError):
    pass


def normalise(name: str) -> str:
    return " ".join(name.strip().lower().replace("_", " ").replace("-", " ").split())


def class_id(name: str) -> int:
    """COCO id of a class name (case, '_' and '-' do not matter; a few aliases such as bike, phone, sofa work)."""
    n = normalise(name)
    n = ALIASES.get(n, n)
    if n in NAME_TO_ID:
        return NAME_TO_ID[n]
    near = sorted(c for c in NAME_TO_ID if n and (c.startswith(n[:3]) or n in c))[:6]
    raise UnknownClass(f"unknown class {name!r}" + (f"; did you mean {', '.join(near)}?" if near else "")
                       + "; the 80 COCO classes are listed in src/inframe/classes.py")


def class_name(cid: int) -> str:
    return COCO_91[cid]


def parse_class_spec(spec: str) -> tuple[int, str]:
    """`dog` (the part of `class:dog` after the colon) -> (18, 'dog')."""
    cid = class_id(spec)
    return cid, COCO_91[cid]


# ---------------------------------------------------------------- model size
SIZES = ("nano", "small", "base")  # detect.MODELS order, smallest first
INPUT = {"nano": 384, "small": 512, "base": 560}  # detector input side in pixels
SMALL_PX = 80  # nano misses anything under about 80 px (the roadmap's figure, not measured here): an object smaller than this in a model's input goes to the next size


def model_for(preset_model: str, cls: int, height_frac: float | None = None, have: tuple[str, ...] = SIZES) -> tuple[str, str | None]:
    """(model size, why it was changed or None) for a run that looks for class `cls`.
    A person keeps the preset's model (the person runs of the regression gate depend on it). Any other class never runs on nano: its object sizes are
    not known and a dog or a ball is smaller in the frame than a person. When `height_frac` (the object's box height as a fraction of the
    frame height, from a tap) says the object is under SMALL_PX pixels high in the model's input, the next larger model is used,
    as far as `have` lists one."""
    if cls == PERSON:
        return preset_model, None
    size, why = preset_model, None
    if size == "nano" and ("small" in have or not have):  # nothing listed: load_detector says what is missing
        size, why = "small", "class targets do not run on nano (it misses small objects)"
    elif size == "nano" and "base" in have:  # no small package, but base is installed: step up past nano
        size, why = "base", "class targets do not run on nano (it misses small objects); small is not installed, using base"
    elif size == "nano" and "nano" in have:  # only nano is installed: run on it, but say that small objects may be missed
        why = "only the nano model is installed and a class target wants small (nano misses small objects); install the small model for better results"
    while height_frac is not None and height_frac * INPUT[size] < SMALL_PX and SIZES.index(size) + 1 < len(SIZES):
        nxt = SIZES[SIZES.index(size) + 1]
        if nxt not in have:
            break
        why = f"the object is about {height_frac * INPUT[size]:.0f} px high in the {size} input (under {SMALL_PX}); using {nxt}"
        size = nxt
    return size, why
