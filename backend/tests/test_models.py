"""Model construction tests with tiny inputs and random weights (no downloads)."""

import numpy as np
import pytest

pytestmark = pytest.mark.tf

SIZE = 75  # smallest input accepted by both backbones


@pytest.fixture(scope="module")
def keras():
    tf = pytest.importorskip("tensorflow")
    tf.random.set_seed(0)
    import keras as k

    k.mixed_precision.set_global_policy("float32")
    return k


@pytest.fixture(scope="module")
def xception_clf(keras):
    from deepfake_detector.models import build_frame_classifier

    return build_frame_classifier("xception", SIZE, weights=None, dropout=0.2)


def test_frame_classifiers_output_probabilities(keras, xception_clf):
    from deepfake_detector.models import build_frame_classifier

    x = np.random.default_rng(0).uniform(-1, 1, (2, SIZE, SIZE, 3)).astype("float32")
    for model in (xception_clf, build_frame_classifier("inception_v3", SIZE, weights=None)):
        y = model.predict(x, verbose=0)
        assert y.shape == (2, 1)
        assert ((y >= 0) & (y <= 1)).all()


def test_invalid_backbone_and_size(keras):
    from deepfake_detector.models import build_backbone

    with pytest.raises(ValueError):
        build_backbone("resnet", SIZE, None)
    with pytest.raises(ValueError):
        build_backbone("inception_v3", 64, None)


def test_two_stage_trainability(keras, xception_clf):
    from deepfake_detector.models import get_backbone, set_trainable

    backbone = get_backbone(xception_clf)
    assert set_trainable(xception_clf, "head") == 0
    assert not backbone.trainable
    head_params = sum(int(np.prod(w.shape)) for w in xception_clf.trainable_weights)
    n = set_trainable(xception_clf, "finetune", 0.2)
    assert 0 < n < len(backbone.layers)
    assert all(not layer.trainable for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization))
    assert sum(int(np.prod(w.shape)) for w in xception_clf.trainable_weights) > head_params
    with pytest.raises(ValueError):
        set_trainable(xception_clf, "warmup")
    set_trainable(xception_clf, "head")


def test_ensemble_weighted_average(keras):
    from deepfake_detector.models import build_ensemble

    def const_model(value):
        inp = keras.Input((SIZE, SIZE, 3))
        out = keras.layers.Lambda(lambda t: keras.ops.ones((keras.ops.shape(t)[0], 1)) * value)(inp)
        return keras.Model(inp, out)

    ens = build_ensemble({"a": const_model(0.2), "b": const_model(0.8)}, [3.0, 1.0])
    out = ens.predict(np.zeros((2, SIZE, SIZE, 3), "float32"), verbose=0)
    assert set(out) == {"ensemble", "a", "b"}
    np.testing.assert_allclose(out["ensemble"], 0.35, rtol=1e-5)


def test_temporal_models_and_serialization(keras, xception_clf, tmp_path):
    from deepfake_detector.models import AttentionPooling, build_temporal_model
    from deepfake_detector.models.backbones import set_trainable

    x = np.random.default_rng(1).uniform(-1, 1, (2, 3, SIZE, SIZE, 3)).astype("float32")
    for pooling in ("mean", "attention"):
        m = build_temporal_model("xception", 3, SIZE, pooling=pooling, weights=None, frame_classifier=xception_clf)
        assert m.predict(x, verbose=0).shape == (2, 1)
    assert set_trainable(m, "finetune", 0.1) > 0  # backbone found inside TimeDistributed
    set_trainable(m, "head")
    path = tmp_path / "temporal.keras"
    m.save(path)
    restored = keras.models.load_model(path, compile=False)
    assert isinstance(restored.get_layer("temporal_pool"), AttentionPooling)
    np.testing.assert_allclose(restored.predict(x, verbose=0), m.predict(x, verbose=0), rtol=1e-4, atol=1e-5)

    with pytest.raises(ValueError):
        build_temporal_model("xception", 3, SIZE, pooling="max", weights=None)


def test_attention_weights_sum_to_one(keras):
    from deepfake_detector.models import AttentionPooling

    layer = AttentionPooling(8)
    feats = np.random.default_rng(2).standard_normal((2, 5, 16)).astype("float32")
    out = layer(feats)
    alpha = keras.ops.convert_to_numpy(layer.attention_weights(feats))
    assert out.shape == (2, 16)
    np.testing.assert_allclose(alpha.sum(axis=1), 1.0, rtol=1e-5)


def test_audio_cnn(keras):
    from deepfake_detector.models import build_audio_cnn

    m = build_audio_cnn(32, 50, filters=(8, 16))
    y = m.predict(np.zeros((3, 32, 50, 1), "float32"), verbose=0)
    assert y.shape == (3, 1)


def test_gradcam_heatmap(keras, xception_clf):
    from deepfake_detector.explain.gradcam import compute_gradcam, encode_png_base64, overlay_heatmap

    img = np.random.default_rng(3).uniform(-1, 1, (SIZE, SIZE, 3)).astype("float32")
    heat = compute_gradcam(xception_clf, img)
    assert heat.shape == (SIZE, SIZE)
    assert heat.min() >= 0.0 and heat.max() <= 1.0
    overlay = overlay_heatmap(((img + 1) * 127.5).astype("uint8"), heat)
    assert encode_png_base64(overlay).startswith("data:image/png;base64,")


def test_keras_demo_detector_end_to_end(keras, cfg, image_file):
    from deepfake_detector.inference.predictor import DeepfakeDetector

    cfg.data.image_size = SIZE
    cfg.api.demo_backend = "keras"
    cfg.audio.enabled = False
    det = DeepfakeDetector(cfg)
    assert det.mode == "demo"
    assert {s.kind for s in det.frame_scorers.values()} == {"random_weights"}
    res = det.predict_file(image_file)
    assert res["mode"] == "demo" and res["warning"]
    assert res["gradcam"]["method"] == "grad-cam"


def test_trained_weights_are_loaded(keras, cfg, image_file, xception_clf):
    from pathlib import Path

    from deepfake_detector.inference.predictor import DeepfakeDetector

    cfg.data.image_size = SIZE
    cfg.model.backbones = ["xception"]
    cfg.model.ensemble_weights = [1.0]
    cfg.audio.enabled = False
    Path(cfg.api.weights_dir).mkdir(parents=True)
    xception_clf.save(Path(cfg.api.weights_dir) / "xception.keras")
    det = DeepfakeDetector(cfg)
    assert det.mode == "trained"
    res = det.predict_file(image_file)
    assert res["mode"] == "trained" and res["warning"] is None
    assert res["modalities"]["visual"]["per_model"]["xception"]["status"] == "trained"
