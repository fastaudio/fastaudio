import torch
from fastai.data.all import test_eq as _test_eq
from unittest.mock import patch

from fastaudio.augment.functional import region_mask


class TestCreateRegionMask:
    def test_shape(self):
        _test_eq(region_mask(1, 5, 7, 10).shape, (1, 10))
        _test_eq(region_mask(2, 3, 7, 12).shape, (2, 12))
        _test_eq(region_mask(4, 0, 3, 3).shape, (4, 3))

    def test_max(self):
        # Test max size
        with patch(
            "torch.rand",
            side_effect=[
                torch.Tensor([[[[1.0]]]]),
                torch.Tensor([[[[0.0]]]]),
            ],
        ):
            _test_eq(
                region_mask(1, 4, 6, 10),
                torch.BoolTensor([[[[1] * 6 + [0] * 4]]]),
            )

    def test_min(self):
        # Test min size
        with patch(
            "torch.rand",
            side_effect=[
                torch.Tensor([0.0]),
                # Test start middle start here too
                torch.Tensor([0.5]),
            ],
        ):
            _test_eq(
                region_mask(1, 4, 6, 10),
                torch.BoolTensor([0] * 3 + [1] * 4 + [0] * 3),
            )

    def test_multiple_masks(self):
        # Test multiple masks
        with patch(
            "torch.rand",
            side_effect=[
                torch.Tensor([[1.0], [0.0]]),
                torch.Tensor([[0.0], [0.5]]),
            ],
        ):
            _test_eq(
                region_mask(2, 4, 6, 10),
                torch.BoolTensor([[1] * 6 + [0] * 4, [0] * 3 + [1] * 4 + [0] * 3]),
            )
