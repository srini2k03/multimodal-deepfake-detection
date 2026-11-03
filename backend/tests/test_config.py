from pathlib import Path

import pytest

from deepfake_detector.config import DEFAULT_CONFIG_PATH, Config, apply_env_overrides, load_config


def test_default_yaml_loads_and_matches_dataclass_defaults():
    cfg = load_config(DEFAULT_CONFIG_PATH, env=False)
    assert cfg.model.backbones == ["xception", "inception_v3"]
    assert cfg.data.image_size == 299
    assert cfg.to_dict()["api"]["demo_mode"] == "auto"


def test_smoke_yaml_loads():
    cfg = load_config(Path(DEFAULT_CONFIG_PATH).parent / "smoke.yaml", env=False)
    assert cfg.data.image_size == 96
    assert cfg.model.temporal_pooling == "attention"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("model:\n  backbone_typo: [xception]\n")
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(p, env=False)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: setattr(c.model, "backbones", ["resnet"]),
        lambda c: setattr(c.model, "ensemble_weights", [1.0]),
        lambda c: setattr(c.fusion, "threshold", 1.5),
        lambda c: setattr(c.data, "image_size", 32),
        lambda c: setattr(c.api, "frame_aggregation", "mode"),
    ],
)
def test_validation_errors(mutate):
    cfg = Config()
    mutate(cfg)
    with pytest.raises(ValueError):
        cfg.validate()


def test_env_overrides():
    cfg = apply_env_overrides(
        Config(),
        {
            "DFD_API__MAX_UPLOAD_MB": "7",
            "DFD_TRAIN__MIXED_PRECISION": "false",
            "DFD_MODEL__ENSEMBLE_WEIGHTS": "0.3,0.7",
            "DFD_API__CORS_ORIGINS": "http://a,http://b",
        },
    )
    assert cfg.api.max_upload_mb == 7
    assert cfg.train.mixed_precision is False
    assert cfg.model.ensemble_weights == [0.3, 0.7]
    assert cfg.api.cors_origins == ["http://a", "http://b"]
