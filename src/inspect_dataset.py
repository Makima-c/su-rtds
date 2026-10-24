# Downloading and inspecting the LoCoMo dataset


import json
import re
from pathlib import Path
from urllib.request import urlretrieve

DATASET_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)

DATASET_PATH = Path(__file__).resolve().parent / "data" / "locomo10.json"
DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

# Download dataset once
if not DATASET_PATH.exists():
    print("Downloading LoCoMo dataset...")
    urlretrieve(DATASET_URL, DATASET_PATH)

# Load dataset
with DATASET_PATH.open("r", encoding="utf-8") as f:
    data = json.load(f)

print("Total conversations:", len(data))

sample = data[0]

print("\nSample ID:", sample.get("sample_id"))
print("Top-level keys:", list(sample.keys()))
print("Number of QA pairs:", len(sample["qa"]))

# Inspect conversation sessions
conversation = sample["conversation"]

session_keys = sorted(
    [key for key in conversation if re.fullmatch(r"session_\d+", key)],
    key=lambda key: int(key.split("_")[1]),
)

print("\nNumber of sessions:", len(session_keys))

if session_keys:
    first_session = session_keys[0]
    print("First session:", first_session)

    print("\nExample dialogue turn:")
    print(conversation[first_session][0])

print("\nExample QA:")
print(sample["qa"][0])
