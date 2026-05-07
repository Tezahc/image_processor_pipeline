import random
import shutil
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Pairing factory  (à passer comme pairing_function au ProcessingStep)
# ──────────────────────────────────────────────────────────────────────────────

def make_split_pairing(
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
    split_names: Tuple[str, str] = ("train", "val"),
):
    """Retourne une pairing_function pour le mode 'custom' qui découpe les
    fichiers d'entrée en deux partitions nommées (par défaut train/val).

    La logique de shuffle et de ratio est encapsulée dans la closure :
    elle n'a pas besoin de transiter par process_kwargs, qui reste réservé
    aux arguments de la process_function elle-même.

    Parameters
    ----------
    train_ratio : float, default=0.8
        Part des fichiers affectée à la première partition.
    seed : Optional[int], default=None
        Graine pour random.shuffle. None = non reproductible.
    split_names : Tuple[str, str], default=('train', 'val')
        Noms des deux partitions yielded comme 2e élément de chaque tuple.
        Doit correspondre aux composants des output_dirs déclarés dans le step
        (ex. "train/images" contient "train" dans ses parts).

    Returns
    -------
    Callable[[List[List[Path]]], Iterator[Tuple[Path, str]]]
        Fonction compatible avec l'interface pairing_function de ProcessingStep.
        Yields : (image_path, split_name)

    Composition avec sample_k
    -------------------------
    sample_k est appliqué par ProcessingStep *avant* l'appel à cette fonction.
    Exemple : sample_k=500 + train_ratio=0.8 → 500 images tirées aléatoirement,
    dont 400 en train et 100 en val.

    Example
    -------
    >>> step = ProcessingStep(
    ...     pairing_method='custom',
    ...     pairing_function=make_split_pairing(train_ratio=0.8, seed=42),
    ...     ...
    ... )
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"'train_ratio' doit être dans ]0, 1[, reçu : {train_ratio}")
    if len(split_names) != 2:
        raise ValueError(f"'split_names' doit contenir exactement 2 chaînes. reçu : {split_names}")

    def pairing(input_file_lists: List[List[Path]]) -> Iterator[Tuple[Path, str]]:
        if len(input_file_lists) != 1:
            raise ValueError("make_split_pairing attend exactement 1 dossier d'entrée.")
        files = list(input_file_lists[0])
        if seed is not None:
            random.seed(seed)
        random.shuffle(files)
        cut = int(len(files) * train_ratio)

        # chaque yield est consommé complètement avant de passer au suivant
        yield from ((f, split_names[0]) for f in files[:cut])
        yield from ((f, split_names[1]) for f in files[cut:])

    return pairing


# ──────────────────────────────────────────────────────────────────────────────
# Process function
# ──────────────────────────────────────────────────────────────────────────────

def reorganize_for_yolo(
    image_path: Path,
    split: str,
    output_dirs: List[Path],
    **options,
) -> Optional[Path]:
    """Copie une image et son label YOLO dans les dossiers du bon split.

    Conçue pour être utilisée avec pairing_method='custom' et
    make_split_pairing() : le 2e argument positionnel `split` ('train' ou 'val')
    est fourni par la pairing_function.

    Le routage vers le bon dossier de sortie utilise les parts des chemins :
    la fonction cherche dans output_dirs celui dont le nom contient à la fois
    `split` et 'images' (resp. 'labels'). Cela évite toute dépendance à l'ordre
    de la liste et reste robuste si d'autres dossiers sont ajoutés.

    Parameters
    ----------
    image_path : Path
        Chemin vers l'image (1er arg positionnel fourni par ProcessingStep).
    split : str
        Nom de la partition, ex. 'train' ou 'val' (2e arg, fourni par la
        pairing_function).
    output_dirs : List[Path]
        Les 4 dossiers résolus par le pipeline, ex. :
            [.../train/images, .../train/labels, .../val/images, .../val/labels]
        Créés automatiquement par ProcessingStep.run() avant le premier appel.
    **options :
        label_dir : Path, optional
            Dossier des labels. Par défaut : dossier `labels/` voisin de
            `images/` dans la même arborescence (convention YOLO standard).

    Returns
    -------
    Optional[Path]
        Chemin de l'image copiée, ou None si le label est introuvable.
    """
    # ── Résolution du label ───────────────────────────────────────────────────
    label_dir: Path = options.get(
        "label_dir",
        image_path.parent.parent / "labels",  # .../all/images/../labels
    )
    label_path = label_dir / image_path.with_suffix(".txt").name

    if not label_path.exists():
        print(f"  [SKIP] Label introuvable pour '{image_path.name}' → {label_path}")
        return None

    # ── Routage vers le bon dossier de sortie ─────────────────────────────────
    # On filtre output_dirs par les parts du chemin plutôt que par index :
    # robuste à tout réordonnancement de la liste déclarée dans le step.
    try:
        dest_images = next(d for d in output_dirs if split in d.parts and "images" in d.parts)
        dest_labels = next(d for d in output_dirs if split in d.parts and "labels" in d.parts)
    except StopIteration:
        raise ValueError(
            f"Aucun dossier de sortie trouvé pour split='{split}' dans output_dirs.\n"
            f"Vérifier que output_dirs contient des chemins avec '{split}/images' "
            f"et '{split}/labels' comme composants.\n"
            f"output_dirs reçu : {output_dirs}"
        )

    # Copie
    shutil.copy(image_path, dest_images / image_path.name)
    shutil.copy(label_path, dest_labels / label_path.name)

    return dest_images / image_path.name


# ──────────────────────────────────────────────────────────────────────────────
# Instanciation du ProcessingStep
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ..pipeline import ProcessingStep

    DATASET_ROOT = Path("C:/Users/GuillaumeChazet/Documents/ICUREsearch/DeepCath/temp/ds_test")
    IMAGES_DIR   = DATASET_ROOT / "all" / "images"   # input résolu en absolu direct

    reorganize_step = ProcessingStep(
        name="reorganize_yolo_dataset",
        process_function=reorganize_for_yolo,
        input_dirs=[IMAGES_DIR],
        output_dirs=[                          # chemins relatifs → résolus via root_dir
            "train/images",
            "train/labels",
            "val/images",
            "val/labels",
        ],
        root_dir=DATASET_ROOT,                 # racine commune pour les output_dirs
        pairing_method="custom",
        pairing_function=make_split_pairing(
            train_ratio=0.8,
            seed=42,                           # retirer pour un split non reproductible
        ),
        workers=1,
        save_log=True,
        # options={} intentionnellement vide ou absent :
        # process_kwargs ne contient que ce dont reorganize_for_yolo a besoin.
        # label_dir n'est utile que si la convention images/labels n'est pas respectée.
    )

    reorganize_step.run()
