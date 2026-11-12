import pytest
from PIL import Image

import hoi_adapter


@pytest.fixture
def mock_inference(monkeypatch):
    def fake_get_resources():
        return (
            object(),
            {},
            object(),
            {"epoch": 1900},
            None,
            {},
        )

    def fake_run_inference(**kwargs):
        image = kwargs["image"]
        kwargs["output_path"].parent.mkdir(parents=True, exist_ok=True)
        image.save(kwargs["output_path"], format="JPEG")

        return {
            "ranked_predictions": [
                {
                    "rank": 1,
                    "verb": "hold",
                    "object": "cell phone",
                    "gnn_score": 0.8345,
                    "qpic_score": 0.5541,
                    "object_score": 0.9123,
                }
            ],
            "score_note": "Test inference",
        }

    monkeypatch.setattr(
        hoi_adapter, "get_model_resources", fake_get_resources
    )
    monkeypatch.setattr(
        hoi_adapter, "run_inference", fake_run_inference
    )


def test_predict_image_returns_expected_structure(mock_inference):
    image = Image.new("RGB", (640, 480), color="white")

    result = hoi_adapter.predict_image(image)

    assert isinstance(result, dict)
    assert "annotated_image" in result
    assert "predictions" in result
    assert result["is_mock"] is False


def test_annotated_image_is_valid_pil_image(mock_inference):
    image = Image.new("RGB", (640, 480), color="white")

    result = hoi_adapter.predict_image(image)

    assert isinstance(result["annotated_image"], Image.Image)
    assert result["annotated_image"].size == image.size
    assert result["annotated_image"].mode == "RGB"


def test_prediction_contains_real_score_fields(mock_inference):
    image = Image.new("RGB", (640, 480), color="white")

    result = hoi_adapter.predict_image(image)
    prediction = result["predictions"][0]

    assert prediction["subject"] == "person"
    assert prediction["action"] == "hold"
    assert prediction["object"] == "cell phone"

    assert {
        "gnn_score",
        "qpic_score",
        "object_score",
    }.issubset(prediction.keys())


def test_invalid_input_raises_type_error():
    with pytest.raises(TypeError):
        hoi_adapter.predict_image("not an image")


@pytest.mark.parametrize(
    "size",
    [
        (1, 1),
        (0, 10),
        (10, 0),
    ],
)
def test_invalid_image_dimensions_raise_value_error(size):
    image = Image.new("RGB", size)

    with pytest.raises(ValueError):
        hoi_adapter.predict_image(image)


def test_invalid_threshold_raises_value_error():
    image = Image.new("RGB", (640, 480))

    with pytest.raises(ValueError):
        hoi_adapter.predict_image(image, object_threshold=1.5)


def test_invalid_candidate_count_raises_value_error():
    image = Image.new("RGB", (640, 480))

    with pytest.raises(ValueError):
        hoi_adapter.predict_image(image, max_candidates=0)
