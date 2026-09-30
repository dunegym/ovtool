"""imagegen result handling: tensor/array results to PNG files."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import openvino as ov
import pytest
from PIL import Image

from ovtool.imagegen import _save_images, ov_genai_tensor


def _arr(n=1, h=4, w=6, c=3):
    a = np.zeros((n, h, w, c), np.uint8)
    for i in range(n):
        a[i] = (i * 40) % 255
    return a


@pytest.mark.parametrize("n", [1, 2])
def test_save_images_from_array_data(tmp_path, n):
    result = SimpleNamespace(data=_arr(n))
    saved = _save_images(result, tmp_path, "t2i")
    assert len(saved) == n
    for i, p in enumerate(saved):
        assert p.endswith(f"t2i_{i}.png")
        with Image.open(p) as im:
            assert im.size == (6, 4)
            assert im.getpixel((0, 0))[0] == (i * 40) % 255


def test_save_images_from_3d_array_adds_batch(tmp_path):
    saved = _save_images(SimpleNamespace(data=_arr(1)[0]), tmp_path, "t2i")
    assert len(saved) == 1


def test_save_images_from_ov_tensor(tmp_path):
    saved = _save_images(ov.Tensor(_arr(2)), tmp_path, "i2i")
    assert len(saved) == 2


def test_save_images_from_images_list(tmp_path):
    ims = [SimpleNamespace(data=_arr(1)[0]), _arr(2)[1]]
    saved = _save_images(SimpleNamespace(images=ims), tmp_path, "x")
    assert len(saved) == 2


def test_ov_genai_tensor_is_nhwc_uint8():
    img = Image.fromarray(np.full((8, 6, 3), 127, np.uint8))
    t = ov_genai_tensor(img)
    assert list(t.shape) == [1, 8, 6, 3]
    assert t.element_type.to_string() == "u8"
