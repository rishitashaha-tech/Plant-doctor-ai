
from pathlib import Path

dataset_dir = Path("PlantVillage")

if not dataset_dir.exists():
    print("ERROR: PlantVillage folder nahi mila!")
else:
    print("Plant folders and disease categories:\n")

    for folder in sorted(dataset_dir.iterdir()):
        if folder.is_dir() and any(
            plant in folder.name.lower()
            for plant in ["tomato", "potato", "pepper"]
        ):
            image_count = sum(
                1 for file in folder.iterdir()
                if file.is_file()
                and file.suffix.lower() in [".jpg", ".jpeg", ".png"]
            )

            print(f"{folder.name}: {image_count} images")

