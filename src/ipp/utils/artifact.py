from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Artifact:
    image_path: Path
    transformation: str
    params: Dict[str, Any]

    label_path: Optional[Path] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "image_path": self.image_path,
            "transformation": self.transformation,
            "params": self.params
        }

        if self.label_path is not None:
            data["label_path"] = self.label_path
        if self.extra:
            data["extra"] = self.extra
        
        return data