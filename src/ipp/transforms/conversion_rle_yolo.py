import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from .conversion_yolo import YoloSegmentationConverter
from ipp.utils.artifact import Artifact
from ipp.utils import utils


logger = logging.getLogger(__name__)


# =============================================================================
#  Factory — point d'entrée public
# =============================================================================

def make_convert_fn(
    df: pd.DataFrame,
    filename_col: str = "image",
    brushlabels_col: str = "brushlabels",
    rle_col: str = "rle",
) -> Callable:
    """Fabrique une ``process_function`` compatible avec ``DataFrameImportStep``,
    en capturant le DataFrame dans une closure.

    Le DataFrame est une **source de données**, pas un paramètre de configuration :
    il n'a pas sa place dans ``options`` (non-hashable, non-sérialisable, trop volumineux).
    Le capturer ici garantit qu'il n'apparaîtra jamais dans ``process_kwargs`` ni dans
    le champ ``options_used`` des logs du pipeline.

    Les paramètres *sérialisables* (``label_map``, ``tol``, ``simplify``) restent dans
    ``options={}`` de l'étape et sont donc tracés dans les logs — c'est voulu.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame complet des annotations. Capturé par la closure, invisible au pipeline.
    filename_col : str, default ``"image"``
        Colonne contenant les noms de fichiers dans ``df``.
    brushlabels_col : str, default ``"brushlabels"``
        Colonne contenant le label de classe de chaque région annotée.
    rle_col : str, default ``"rle"``
        Colonne contenant les données RLE brutes.

    Returns
    -------
    Callable
        Fonction ``_convert(image_path, output_dirs, **options)`` prête à être passée
        à ``DataFrameImportStep(process_function=...)``.

    Examples
    --------
    >>> step = DataFrameImportStep(
    ...     name="rle_to_yolo",
    ...     source=df_full,
    ...     image_pool=Path("images/"),
    ...     filename_col="image",
    ...     process_function=make_convert_fn(df_full),  # df capturé ici
    ...     output_dirs=[Path("dataset/labels")],
    ...     save_log=True,   # active la génération du manifest JSON
    ...     options={
    ...         "label_map": {"catheter": 1, "wire": 2},
    ...         "tol": 0.002,
    ...         "simplify": True,
    ...     },
    ... )
    """

    def _convert(
        image_path: Path,
        output_dirs: List[Path],
        *,
        label_map: Dict[str, int],
        tol: float = 0.001,
        simplify: bool = True,
        **kwargs: Any,
    ) -> Optional[List[Artifact]]:
        """Convertit les annotations RLE d'une image en fichier de labels YOLO segmentation.

        Conçue pour être utilisée dans une ``DataFrameImportStep`` en mode ``one_input``.
        Reçoit un chemin image par appel, filtre le DataFrame capturé sur ce fichier,
        puis exécute la chaîne ``YoloSegmentationConverter`` pour chaque région annotée.

        **Manifest de reconstruction**

        Cette fonction retourne une liste contenant un seul ``Artifact`` par image traitée.
        Son champ ``params`` embarque *toutes* les données nécessaires à une reconstruction
        à l'identique : ``tol``, ``simplify``, et pour chaque région son ``brushlabels``,
        ``class_id`` et ``rle`` brut.

        Lorsque l'étape est configurée avec ``save_log=True``, le pipeline sérialise
        automatiquement l'ensemble des ``Artifact`` en un seul fichier JSON (nommé d'après
        l'étape et placé dans le dossier parent du premier ``output_dirs``). Ce fichier
        constitue le **manifest global** de l'exécution — il n'y a aucun fichier sidecar
        par image.

        Structure d'une entrée dans le manifest :

        .. code-block:: json

            {
              "inputs": ["images/img_001.jpg"],
              "outputs": [
                {
                  "image_path": "dataset/labels/img_001.txt",
                  "transformation": "rle_to_yolo",
                  "params": {
                    "tol": 0.001,
                    "simplify": true,
                    "regions": [
                      { "brushlabels": "catheter", "class_id": 1, "rle": [...] },
                      { "brushlabels": "wire",     "class_id": 2, "rle": [...] }
                    ]
                  }
                }
              ],
              "status": "Success"
            }

        Parameters
        ----------
        image_path : Path
            Chemin absolu vers l'image courante, fourni par l'orchestrateur.
            Seul le ``.name`` sert au filtre dans le DataFrame.
        output_dirs : List[Path]
            Dossiers de sortie fournis par le pipeline. ``output_dirs[0]`` reçoit le ``.txt``.
        label_map : Dict[str, int]
            Correspondance ``brushlabels → class_id`` YOLO, passé via ``options``.
            Les libellés absents reçoivent l'id ``0``.
        tol : float, default ``0.001``
            Tolérance de simplification polygonale (epsilon relatif au périmètre).
        simplify : bool, default ``True``
            Si ``False``, l'étape ``simplify_polygons`` est sautée.
        **kwargs
            Arguments supplémentaires ignorés (tolérance aux options génériques du pipeline).

        Returns
        -------
        List[Artifact]
            Liste avec un seul ``Artifact`` dont ``image_path`` pointe vers le ``.txt`` YOLO
            et ``params`` contient toutes les données de reconstruction.
        None
            Si aucune annotation n'est trouvée pour cette image, ou si toutes les
            régions échouent à la conversion.
        """
        output_dir = utils._validate_dirs(output_dirs, 1)
        image_name = image_path.name  # seul le nom sert au filtre, comme dans le notebook

        # --- Filtre sur l'image courante ---
        rows: pd.DataFrame = df[df[filename_col] == image_name]

        if rows.empty:
            logger.warning(f"[{image_name}] Aucune annotation trouvée dans le DataFrame. Image ignorée.")
            return None

        # --- Initialisation du converter ---
        try:
            converter = YoloSegmentationConverter(image_path)
        except Exception as e:
            logger.error(f"[{image_name}] Impossible de charger l'image : {e}")
            return None

        # Régions converties avec succès — embarquées dans l'Artifact pour le manifest
        converted_regions = []

        for _, row in rows.iterrows():
            rle_data = row[rle_col]
            brushlabel = row[brushlabels_col]
            class_id = label_map.get(brushlabel, 0)

            try:
                converter.convert(rle_data=rle_data, class_id=class_id, tol=tol, simplify=simplify)
            except Exception as e:
                # Une région défectueuse ne bloque pas les autres
                logger.error(f"[{image_name}] Échec région '{brushlabel}' : {e}")
                continue

            # Accumulé seulement si la conversion a réussi
            converted_regions.append({
                "brushlabels": brushlabel,
                "class_id": class_id,
                "rle": rle_data,
            })

        if not converter.yolo_lines:
            logger.warning(f"[{image_name}] Aucune ligne YOLO générée (toutes les régions ont échoué).")
            return None

        # --- Écriture du fichier de labels YOLO ---
        label_path = output_dir / Path(image_name).with_suffix(".txt")
        converter.write_yolo_lines(label_path)

        # --- Construction de l'Artifact ---
        # params contient tout ce qui est nécessaire à la reconstruction à l'identique.
        # Il sera sérialisé par _build_log via Artifact.to_dict() et finira dans le
        # manifest JSON unique produit par save_log=True — pas de fichier sidecar.
        artifact = Artifact(
            image_path=label_path,
            transformation="rle_to_yolo",
            params={
                "tol": tol,
                "simplify": simplify,
                "regions": converted_regions,
            },
        )

        return [artifact]

    # Nom lisible pour les logs et tracebacks
    _convert.__name__ = "convert_rle_to_yolo_labels"
    _convert.__qualname__ = "make_convert_fn.<locals>.convert_rle_to_yolo_labels"

    return _convert