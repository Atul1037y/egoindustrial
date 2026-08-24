"""Tests for data module."""

import torch


def test_imports():
    """Test that all modules can be imported."""
    from egoindustrial.data import (
        Assembly101Dataset,
        ConcatDatasetWithDomain,
        EpicKitchensDataset,
        HoloAssistDataset,
        build_dataloader,
        build_datasets,
    )
    from egoindustrial.inference import create_app, run_server
    from egoindustrial.models import (
        InternVideo2,
        MViTv2,
        SlowFast,
        VideoMAEv2,
        get_model,
    )

    assert EpicKitchensDataset is not None
    assert Assembly101Dataset is not None
    assert HoloAssistDataset is not None
    assert ConcatDatasetWithDomain is not None
    assert build_datasets is not None
    assert build_dataloader is not None
    assert VideoMAEv2 is not None
    assert MViTv2 is not None
    assert SlowFast is not None
    assert InternVideo2 is not None
    assert get_model is not None
    assert create_app is not None
    assert run_server is not None

    # TensorRTEngine is optional (requires tensorrt)
    from egoindustrial.inference import TensorRTEngine
    if TensorRTEngine is not None:
        assert TensorRTEngine is not None


def test_model_registry():
    """Test model registry."""
    from egoindustrial.models import get_model, list_models

    models = list_models()
    assert "videomaev2" in models
    assert "mvitv2" in models
    assert "slowfast" in models
    assert "internvideo2" in models

    # Test getting a model
    model = get_model("videomaev2", num_verb_classes=10, num_noun_classes=20, num_action_classes=30, pretrained=False)
    assert model is not None


def test_multi_task_head():
    """Test multi-task head."""
    from egoindustrial.models.head import MultiTaskHead

    head = MultiTaskHead(
        embed_dim=768,
        num_verb_classes=97,
        num_noun_classes=300,
        num_action_classes=3806,
    )

    x = torch.randn(2, 768)
    out = head(x)

    assert "verb" in out
    assert "noun" in out
    assert "action" in out
    assert out["verb"].shape == (2, 97)
    assert out["noun"].shape == (2, 300)
    assert out["action"].shape == (2, 3806)


def test_losses():
    """Test loss functions."""
    from egoindustrial.training.losses import FocalLoss, MultiTaskLoss

    loss_fn = MultiTaskLoss()
    preds = {
        "verb": torch.randn(4, 97),
        "noun": torch.randn(4, 300),
        "action": torch.randn(4, 3806),
    }
    targets = {
        "verb_label": torch.randint(0, 97, (4,)),
        "noun_label": torch.randint(0, 300, (4,)),
        "action_label": torch.randint(0, 3806, (4,)),
    }

    losses = loss_fn(preds, targets)
    assert "verb" in losses
    assert "noun" in losses
    assert "action" in losses
    assert "total" in losses
    assert losses["total"].item() > 0

    # Test FocalLoss
    focal = FocalLoss()
    loss = focal(torch.randn(4, 10), torch.randint(0, 10, (4,)))
    assert loss.item() > 0


def test_metrics():
    """Test metrics."""
    from egoindustrial.training.metrics import get_metrics

    metrics = get_metrics(10, 20, 30)
    assert metrics is not None
