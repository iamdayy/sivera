from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from api.core.config import settings
from api.schemas.prediction import PREDICTION_OUTPUT_TYPES


class AutoPredictConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")

    enabled: bool = False
    interval_sec: int = Field(settings.DEFAULT_AUTO_PREDICT_INTERVAL_SEC, ge=5, le=3600)
    source_type: Literal["rtsp_frame"] = "rtsp_frame"
    seg_model: str | None = Field(settings.DEFAULT_AUTO_PREDICT_SEG_MODEL, max_length=100)
    seg_conf: float = Field(settings.DEFAULT_AUTO_PREDICT_SEG_CONF, ge=0, le=1)
    seg_classes: list[str] | None = Field(None, max_length=100)
    volume_cell_m: float = Field(settings.DEFAULT_AUTO_PREDICT_VOLUME_CELL_M, gt=0, le=1)
    volume_min_pts_per_cell: int = Field(settings.DEFAULT_AUTO_PREDICT_VOLUME_MIN_PTS, ge=1, le=10000)
    return_outputs: list[str] | None = Field(default_factory=lambda: list(settings.DEFAULT_AUTO_PREDICT_RETURN_OUTPUTS), max_length=13)
    max_frame_age_sec: int = Field(settings.DEFAULT_AUTO_PREDICT_MAX_FRAME_AGE_SEC, ge=1, le=300)

    @model_validator(mode="after")
    def validate_options(self):
        if self.seg_model and (Path(self.seg_model).name != self.seg_model or "\\" in self.seg_model):
            raise ValueError("prediction_invalid_model")
        if self.seg_classes is not None and any(not name or len(name) > 100 for name in self.seg_classes):
            raise ValueError("prediction_invalid_parameters")
        if self.return_outputs is not None and not set(self.return_outputs).issubset(PREDICTION_OUTPUT_TYPES):
            raise ValueError("prediction_invalid_outputs")
        return self


class AutoPredictConfigUpdate(AutoPredictConfig):
    """Only explicitly supplied fields are merged into the saved configuration."""
