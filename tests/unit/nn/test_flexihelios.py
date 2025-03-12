"""Unit tests for the flexihelios module."""

import logging

import pytest
import torch
from einops import repeat

from helios.data.constants import Modality, ModalitySpec
from helios.nn.flexihelios import (Encoder, EncoderConfig, FlexiHeliosBase,
                                   FlexiHeliosCompositeEncodings, PoolingType,
                                   Predictor, PredictorConfig, TokensAndMasks)
from helios.train.masking import MaskValue

logger = logging.getLogger(__name__)


class TestFlexiHeliosCompositeEncodings:
    """Unit tests for the FlexiHeliosCompositeEncodings class."""

    @pytest.fixture
    def flexi_helios_composite_encodings(
        self,
    ) -> FlexiHeliosCompositeEncodings:
        """Create composite encoder fixture for testing."""
        flexi_helios_composite_encodings = FlexiHeliosCompositeEncodings(
            embedding_size=16,
            supported_modalities=[
                Modality.SENTINEL2_L2A,
                Modality.LATLON,
                Modality.WORLDCOVER,
            ],
            max_sequence_length=12,
            use_channel_embs=True,
            random_channel_embs=True,
        )
        return flexi_helios_composite_encodings

    def test_apply_encodings_per_modality_latlon(
        self,
        flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings,
    ) -> None:
        """Test applying encodings to different modalities."""
        B, D = 4, 16
        patch_size = 4
        input_res = 10
        latlon_tokens = torch.randn(B, 1, D)
        ll_enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "latlon", latlon_tokens, None, patch_size, input_res
        )
        assert not (ll_enc == 0).all()
        assert not (ll_enc == latlon_tokens).all()
        assert latlon_tokens.shape == ll_enc.shape

    def test_apply_encodings_per_modality_sentinel2_l2a(
        self, flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, T, C, D = 4, 4, 4, 3, 3, 16
        patch_size = 4
        input_res = 10
        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        )
        timestamps = repeat(timestamps, "... -> b ...", b=B)
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, C, D)
        enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "sentinel2_l2a", sentinel2_l2a_tokens, timestamps, patch_size, input_res
        )
        assert not (enc == 0).all()

    def test_apply_encodings_per_modality_worldcover(
        self,
        flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings,
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, C, D = 4, 4, 4, 1, 16
        patch_size = 4
        input_res = 10
        worldcover_tokens = torch.randn(B, H, W, C, D)
        wc_enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "worldcover", worldcover_tokens, None, patch_size, input_res
        )
        assert not (wc_enc == 0).all()
        assert not (wc_enc == worldcover_tokens).all()
        assert worldcover_tokens.shape == wc_enc.shape

    def test_apply_encodings_per_modality_grad(
        self, flexi_helios_composite_encodings: FlexiHeliosCompositeEncodings
    ) -> None:
        """Test applying encodings to different modalities."""
        B, H, W, T, C, D = 4, 4, 4, 3, 3, 16
        patch_size = 4
        input_res = 10
        timestamps = torch.tensor(
            [[15, 7, 2023], [15, 8, 2023], [15, 9, 2023]], dtype=torch.long
        )
        timestamps = repeat(timestamps, "... -> b ...", b=B)
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, C, D)
        assert (
            flexi_helios_composite_encodings.per_modality_channel_embeddings[
                "sentinel2_l2a"
            ].grad
            is None
        )
        enc = flexi_helios_composite_encodings._apply_encodings_per_modality(
            "sentinel2_l2a", sentinel2_l2a_tokens, timestamps, patch_size, input_res
        )
        loss = enc.sum()
        loss.backward()
        assert (
            flexi_helios_composite_encodings.per_modality_channel_embeddings[
                "sentinel2_l2a"
            ].grad
            is not None
        )


# TODO: Add tests for when the inputs are completely masked or different dims or something
class TestFlexiHeliosBase:
    """Unit tests for the FlexiHeliosBase class."""

    @pytest.fixture
    def flexi_helios_base(
        self, supported_modalities: list[ModalitySpec]
    ) -> FlexiHeliosBase:
        """Create encoder fixture for testing."""
        flexi_helios_base = FlexiHeliosBase(
            embedding_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            use_channel_embs=True,
        )
        return flexi_helios_base

    def test_collapse_and_combine_hwtc(
        self, flexi_helios_base: FlexiHeliosBase
    ) -> None:
        """Test collapsing tokens from different modalities into single tensor."""
        B, D = 2, 4
        sentinel2_l2a_tokens = torch.randn(B, 2, 1, 1, 2, D)
        sentinel2_l2a_mask = torch.randint(0, 2, (B, 2, 1, 1, 2)).float()
        latlon = torch.randn(B, 1, D)
        latlon_mask = torch.randint(0, 2, (B, 1)).float()
        x = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        tokens, masks = flexi_helios_base.collapse_and_combine_hwtc(x)
        assert tokens.shape == (B, 5, D)
        assert masks.shape == (B, 5)

    def test_split_and_expand_per_modality(self) -> None:
        """Test splitting combined tensor back into per-modality tensors."""
        B, D = 2, 4  # Batch size and embedding dimension
        modality_1_channel_groups = 3
        modality_2_channel_groups = 5
        modalities_to_dims_dict: dict[str, tuple[int, ...]] = {
            "modality1": (B, 2, 2, 1, modality_1_channel_groups, D),
            "modality2": (B, 1, 1, 2, modality_2_channel_groups, D),
        }

        modality1_data = torch.randn(B, 4 * modality_1_channel_groups, D)
        modality2_data = torch.randn(B, 4 * modality_2_channel_groups, D)

        x = torch.cat([modality1_data, modality2_data], dim=1)

        # Now call the function
        modality_tokens_dict = FlexiHeliosBase.split_and_expand_per_modality(
            x, modalities_to_dims_dict
        )

        modality1_tokens = modality_tokens_dict["modality1"]
        modality2_tokens = modality_tokens_dict["modality2"]
        assert list(modality1_tokens.shape) == [
            2,
            2,
            2,
            1,
            3,
            4,
        ], f"Incorrect shape for modality1 tokens: {modality1_tokens.shape}"
        assert list(modality2_tokens.shape) == [
            2,
            1,
            1,
            2,
            5,
            4,
        ], f"Incorrect shape for modality2 tokens: {modality2_tokens.shape}"


class TestEncoder:
    """Unit tests for the Encoder class."""

    @pytest.fixture
    def encoder(self, supported_modalities: list[ModalitySpec]) -> Encoder:
        """Create encoder fixture for testing.

        Returns:
            Encoder: Test encoder instance with small test config
        """
        return Encoder(
            embedding_size=8,
            num_heads=2,
            mlp_ratio=4.0,
            depth=2,
            drop_path=0.1,
            supported_modalities=supported_modalities,
            max_sequence_length=12,
            max_patch_size=8,
            use_channel_embs=True,
        )

    def test_create_token_exit_ids_normal_usage(self, encoder: Encoder) -> None:
        """Test creating exit IDs for early token exiting - normal usage.

        Tests normal usage with full token_exit_cfg.
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, D)
        latlon_tokens = torch.randn(B, 1, D)
        x = {"sentinel2_l2a": sentinel2_l2a_tokens, "latlon": latlon_tokens}

        token_exit_cfg = {"sentinel2_l2a": 1, "latlon": 2}
        exit_ids_dict = encoder.create_token_exit_ids(x, token_exit_cfg)
        assert (
            "sentinel2_l2a" in exit_ids_dict
        ), "Expected 'sentinel2_l2a' key in the result dict"
        sentinel2_l2a_exit_ids = exit_ids_dict["sentinel2_l2a"]
        assert (
            sentinel2_l2a_exit_ids.shape == sentinel2_l2a_tokens.shape
        ), "Shape of exit IDs should match the shape of the modality tokens."

        assert (exit_ids_dict["sentinel2_l2a"] == 1).all()
        assert (exit_ids_dict["latlon"] == 2).all()

    def test_create_token_exit_ids_missing_exit_cfg_band_group(
        self, encoder: Encoder
    ) -> None:
        """Test creating exit IDs for early token exiting - error cases.

        Tests error handling for:
        - Missing band group in token_exit_cfg (KeyError)
        """
        B, H, W, T, D = 1, 2, 2, 2, 4
        sentinel2_l2a_tokens = torch.zeros(B, H, W, T, D)
        x = {"sentinel2_l2a": sentinel2_l2a_tokens}

        with pytest.raises(KeyError):
            incomplete_exit_cfg = {"rgb": 1}  # Missing the "nir" key
            encoder.create_token_exit_ids(x, incomplete_exit_cfg)

    def test_remove_masked_tokens(self) -> None:
        """Test removing masked tokens and tracking indices."""
        d = 2
        x = torch.tensor([[0, 1, 0], [1, 0, 1]]).float()
        x = repeat(x, "b n -> b n d", d=d)
        print(f"x shape: {x.shape}")
        mask = torch.tensor([[1, 0, 1], [0, 1, 0]]).float()

        expected_tokens = torch.tensor(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ]
        )
        num_tokens_to_keep = torch.sum(~mask.bool())
        expected_indices = torch.tensor([[1, 0, 2], [0, 2, 1]])
        expected_updated_mask = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
        tokens, indices, updated_mask = Encoder.remove_masked_tokens(x, mask)
        kept_unmasked_tokens = torch.sum(~updated_mask.bool())
        assert torch.equal(tokens, expected_tokens)
        assert torch.equal(indices, expected_indices)
        assert torch.equal(updated_mask, expected_updated_mask)
        assert kept_unmasked_tokens == num_tokens_to_keep

    @pytest.mark.parametrize(
        "block_idx,exit_after,expected",
        [
            (0, None, False),
            (0, 1, False),
            (1, 1, True),
            (1, 2, False),
            (2, 1, True),
        ],
    )
    def test_should_exit(
        self, block_idx: int, exit_after: int | None, expected: bool
    ) -> None:
        """Test exit condition logic.

        Args:
            block_idx: Current block index
            exit_after: Number of layers after which to exit, or None
            expected: Expected output
        """
        assert Encoder.should_exit(block_idx, exit_after) is expected

    def test_add_removed_tokens(self) -> None:
        """Test adding removed tokens back into tensor."""
        partial_tokens = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0]],
                [[5.0, 55.0], [6.0, 66.0]],
            ]
        )
        indices = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 2],
            ]
        )
        partial_mask = torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 1.0],
            ]
        )

        expected_out = torch.tensor(
            [
                [[1.0, 11.0], [2.0, 22.0], [0.0, 0.0]],
                [[0.0, 0.0], [5.0, 55.0], [0.0, 0.0]],
            ]
        )
        expected_mask = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )

        out, full_mask = Encoder.add_removed_tokens(
            partial_tokens, indices, partial_mask
        )
        assert torch.equal(out, expected_out)
        assert torch.equal(full_mask, expected_mask)

    def test_encoder_config(self, supported_modalities: list[ModalitySpec]) -> None:
        """Tests we can build with default args."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = EncoderConfig(supported_modality_names)
        _ = config.build()


class TestPredictor:
    """Unit tests for the Predictor class."""

    @pytest.fixture
    def predictor(self, supported_modalities: list[ModalitySpec]) -> Predictor:
        """Create predictor fixture for testing."""
        return Predictor(
            supported_modalities=supported_modalities,
            encoder_embedding_size=8,
            decoder_embedding_size=8,
            depth=2,
            mlp_ratio=4.0,
            num_heads=2,
            max_sequence_length=12,
            drop_path=0.1,
            learnable_channel_embeddings=True,
            output_embedding_size=8,
        )

    def test_add_masks(self, predictor: Predictor) -> None:
        """Test adding masks to tokens."""
        B, H, W, T, C, D = (
            1,
            2,
            2,
            1,
            2,
            8,
        )  # Changed D from 16 to 8 to match predictor's embedding size
        sentinel2_l2a_tokens = torch.randn(B, H, W, T, C, D)
        sentinel2_l2a_mask = torch.zeros(B, H, W, T, C, dtype=torch.float32)
        # Set one pixel to be decoded (mask value 2)
        sentinel2_l2a_mask[0, 0, 0, 0, 0] = MaskValue.DECODER.value

        latlon = torch.randn(B, 2, D)
        latlon_mask = torch.zeros(B, 2, dtype=torch.float32)

        tokens_and_masks = {
            "sentinel2_l2a": sentinel2_l2a_tokens,
            "sentinel2_l2a_mask": sentinel2_l2a_mask,
            "latlon": latlon,
            "latlon_mask": latlon_mask,
        }
        replaced_dict = predictor.add_masks(tokens_and_masks)

        # We expect replaced_dict to have the key "sentinel2_l2a"
        assert (
            "sentinel2_l2a" in replaced_dict
        ), "Expected replaced_dict to have key 'sentinel2_l2a'"

        replaced_sentinel2_l2a = replaced_dict["sentinel2_l2a"]
        assert replaced_sentinel2_l2a.shape == sentinel2_l2a_tokens.shape, (
            f"Expected shape {sentinel2_l2a_tokens.shape}, "
            f"got {replaced_sentinel2_l2a.shape}"
        )

        # Check the single pixel we set to be decoded
        replaced_location = replaced_sentinel2_l2a[0, 0, 0, 0, 0, :]

        # Check an unchanged location
        unchanged_location = replaced_sentinel2_l2a[0, 0, 0, 0, 1, :]

        assert torch.allclose(
            replaced_location, predictor.mask_token, atol=1e-6
        ), "Tokens at masked location should be replaced with mask token."
        assert torch.allclose(
            unchanged_location, sentinel2_l2a_tokens[0, 0, 0, 0, 1, :], atol=1e-6
        ), "Tokens at non-masked location should remain the same."

    def test_split_x_y(self) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups."""
        tokens = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13, 14, 15, 16, 17, 18]]
        ).unsqueeze(-1)
        # should we handle target encoder values here?
        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
            ]
        )

        # Add some missing tokens (value MISSING)
        mask[0, 0] = MaskValue.MISSING.value  # First token in first batch is missing
        mask[1, 1] = MaskValue.MISSING.value  # Second token in second batch is missing

        (
            unmasked_tokens,
            tokens_to_decode,
            missing_tokens,
            unmasked_tokens_mask,
            tokens_to_decode_mask,
            missing_tokens_mask,
            indices,
        ) = Predictor.split_x_y(tokens, mask)

        # Check shapes
        assert unmasked_tokens.shape == (2, 6, 1)
        assert tokens_to_decode.shape == (2, 3, 1)
        assert missing_tokens.shape == (2, 1, 1)

        assert missing_tokens[0, 0, 0].item() == 1
        assert missing_tokens[1, 0, 0].item() == 11

        assert torch.equal(missing_tokens_mask, torch.tensor([[1], [1]]))

        expected_unmasked_tokens = torch.tensor(
            [[2, 3, 4, 5, 6, 0], [10, 12, 13, 14, 15, 16]]
        )
        assert torch.equal(unmasked_tokens.squeeze(-1), expected_unmasked_tokens)
        assert torch.equal(
            unmasked_tokens_mask, torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])
        )

        expected_tokens_to_decode = torch.tensor([[7, 8, 9], [17, 18, 0]])
        assert torch.equal(tokens_to_decode.squeeze(-1), expected_tokens_to_decode)
        assert torch.equal(tokens_to_decode_mask, torch.tensor([[1, 1, 1], [1, 1, 0]]))

    def test_split_and_recombine_with_missing_tokens(self) -> None:
        """Test splitting the tokens into decoded, unmasked, and missing groups with missing tokens."""
        # Create a batch with two samples, with tokens in a non-sorted order
        # and different numbers of missing tokens per batch
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],  # First batch
                [10, 11, 12, 13, 14, 15, 16, 17, 18],  # Second batch
            ]
        ).unsqueeze(-1)

        # Create masks with different patterns of missing tokens
        # First batch: 1 missing token, 3 decoder tokens, 5 encoder tokens
        # Second batch: 3 missing tokens, 2 decoder tokens, 4 encoder tokens
        # The tokens are intentionally not sorted by mask value
        mask = torch.tensor(
            [
                [
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
                [
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                ],
            ]
        )

        (
            tokens_to_decode,
            unmasked_tokens,
            missing_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            missing_tokens_mask,
            indices,
        ) = Predictor.split_x_y(tokens, mask)

        # Check shapes
        assert tokens_to_decode.shape == (
            2,
            3,
            1,
        ), f"Expected decoded shape (2, 3, 1), got {tokens_to_decode.shape}"
        assert unmasked_tokens.shape == (
            2,
            5,
            1,
        ), f"Expected unmasked shape (2, 5, 1), got {unmasked_tokens.shape}"
        assert missing_tokens.shape == (
            2,
            3,
            1,
        ), f"Expected missing shape (2, 3, 1), got {missing_tokens.shape}"

        expected_tokens_to_decode = torch.tensor([[20, 60, 80], [33, 66, 0]])
        assert torch.equal(
            tokens_to_decode.squeeze(-1), expected_tokens_to_decode
        ), f"Expected decoded to be {expected_tokens_to_decode}, got {tokens_to_decode.squeeze(-1)}"

        expected_tokens_to_decode_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        assert torch.equal(
            tokens_to_decode_mask, expected_tokens_to_decode_mask
        ), f"Expected decoded_mask to be {expected_tokens_to_decode_mask}, got {tokens_to_decode_mask}"

        expected_unmasked_tokens = torch.tensor(
            [[10, 30, 50, 70, 90], [22, 55, 77, 99, 0]]
        )
        assert torch.equal(
            unmasked_tokens.squeeze(-1), expected_unmasked_tokens
        ), f"Expected unmasked to be {expected_unmasked_tokens}, got {unmasked_tokens.squeeze(-1)}"

        expected_unmasked_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
        assert torch.equal(
            unmasked_tokens_mask, expected_unmasked_mask
        ), f"Expected unmasked_mask to be {expected_unmasked_mask}, got {unmasked_tokens_mask}"

        expected_missing_tokens = torch.tensor([[40, 0, 0], [11, 44, 88]])
        assert torch.equal(
            missing_tokens.squeeze(-1), expected_missing_tokens
        ), f"Expected missing to be {expected_missing_tokens}, got {missing_tokens.squeeze(-1)}"

        expected_missing_mask = torch.tensor([[1, 0, 0], [1, 1, 1]])
        assert torch.equal(
            missing_tokens_mask, expected_missing_mask
        ), f"Expected missing_mask to be {expected_missing_mask}, got {missing_tokens_mask}"

        # Test that we can combine the tokens back correctly
        combined_tokens = Predictor.combine_x_y(
            tokens_to_decode,
            unmasked_tokens,
            missing_tokens,
            tokens_to_decode_mask,
            unmasked_tokens_mask,
            missing_tokens_mask,
            indices,
        )

        # Check the shape of the combined tokens
        assert (
            combined_tokens.shape == tokens.shape
        ), f"Expected combined shape {tokens.shape}, got {combined_tokens.shape}"

        # Check that the combined tokens match the original tokens
        # Note: Tokens with mask values TARGET_ENCODER_ONLY will be set to 0
        # in the combined tokens, so we need to zero those out in our comparison
        masked_tokens = tokens.clone()
        for b in range(tokens.shape[0]):
            for t in range(tokens.shape[1]):
                if mask[b, t] == MaskValue.TARGET_ENCODER_ONLY.value:
                    masked_tokens[b, t] = 0

        assert torch.equal(
            combined_tokens, masked_tokens
        ), "Expected combined tokens to match original tokens"

    def test_combine_x_y(self) -> None:
        """Test combining the decoded, unmasked, and missing groups back into the original tokens."""
        # Create a simple test case with known values
        # decoded is the query (i.e. the masked tokens to be decoded - DECODER)
        decoded = torch.tensor([[14, 15, 16], [15, 16, 0]]).unsqueeze(-1)
        # unmasked is the keys and values (i.e. the unmasked tokens - ONLINE_ENCODER)
        unmasked = torch.tensor([[6, 7, 8], [5, 7, 0]]).unsqueeze(-1)
        # missing is the missing tokens (MISSING)
        missing = torch.tensor([[5], [6]]).unsqueeze(-1)

        # Masks indicate which tokens are valid (1) or padding (0)
        decoded_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        unmasked_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        missing_mask = torch.tensor([[1], [1]])

        # Indices map from the sorted order back to the original order
        indices = torch.tensor(
            [[0, 6, 7, 8, 4, 5, 1, 2, 3], [1, 7, 8, 3, 4, 5, 6, 0, 2]]
        )

        tokens = Predictor.combine_x_y(
            decoded,
            unmasked,
            missing,
            decoded_mask,
            unmasked_mask,
            missing_mask,
            indices,
        )

        # Check the shape of the output
        expected_shape = (2, 9, 1)
        assert (
            tokens.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {tokens.shape}"

        # Check specific token positions based on the actual behavior
        expected_tokens_batch0 = torch.tensor([5, 6, 7, 8, 0, 0, 14, 15, 16]).unsqueeze(
            -1
        )
        expected_tokens_batch1 = torch.tensor([5, 6, 7, 0, 0, 0, 0, 15, 16]).unsqueeze(
            -1
        )

        assert torch.equal(tokens[0], expected_tokens_batch0), (
            f"Expected tokens[0] to be {expected_tokens_batch0.squeeze(-1)}, "
            f"got {tokens[0].squeeze(-1)}"
        )
        assert torch.equal(tokens[1], expected_tokens_batch1), (
            f"Expected tokens[1] to be {expected_tokens_batch1.squeeze(-1)}, "
            f"got {tokens[1].squeeze(-1)}"
        )
        # Need to add asserts that the masks are put back together correctly

    def test_predictor_config(self, supported_modalities: list[ModalitySpec]) -> None:
        """Tests we can build with default args."""
        supported_modality_names = [m.name for m in supported_modalities]
        config = PredictorConfig(supported_modality_names)
        _ = config.build()

    def test_split_x_y_different_missing_counts(self) -> None:
        """Test splitting tokens with different missing token counts per sample."""
        # Create a batch with two samples, each having different numbers of missing tokens
        tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [11, 12, 13, 14, 15, 16, 17, 18, 19],
            ]
        ).unsqueeze(-1)

        # First sample: 2 missing, 4 decoder, 3 encoder
        # Second sample: 3 missing, 3 decoder, 3 encoder
        mask = torch.tensor(
            [
                [
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.DECODER.value,
                ],
                [
                    MaskValue.DECODER.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.DECODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.ONLINE_ENCODER.value,
                    MaskValue.MISSING.value,
                    MaskValue.DECODER.value,
                ],
            ]
        )

        (
            decoded,
            unmasked,
            missing,
            decoded_mask,
            unmasked_mask,
            missing_mask,
            indices,
        ) = Predictor.split_x_y(tokens, mask)
        logger.info(f"decoded: {decoded}")
        logger.info(f"unmasked: {unmasked}")
        logger.info(f"missing: {missing}")
        logger.info(f"decoded_mask: {decoded_mask}")
        logger.info(f"unmasked_mask: {unmasked_mask}")
        logger.info(f"missing_mask: {missing_mask}")
        logger.info(f"indices: {indices}")
        # Check shapes
        assert decoded.shape == (
            2,
            4,
            1,
        ), f"Expected decoded shape (2, 4, 1), got {decoded.shape}"
        assert unmasked.shape == (
            2,
            3,
            1,
        ), f"Expected unmasked shape (2, 3, 1), got {unmasked.shape}"
        assert missing.shape == (
            2,
            3,
            1,
        ), f"Expected missing shape (2, 3, 1), got {missing.shape}"

        # Check decoded values (tokens to be decoded - mask value DECODER)
        expected_decoded = torch.tensor([[6, 7, 8, 9], [17, 18, 19, 0]])
        assert torch.equal(
            decoded.squeeze(-1), expected_decoded
        ), f"Expected decoded to be {expected_decoded}, got {decoded.squeeze(-1)}"

        # Check unmasked values (tokens for context - mask value ONLINE_ENCODER)
        expected_unmasked = torch.tensor([[3, 4, 5], [14, 15, 16]])
        assert torch.equal(
            unmasked.squeeze(-1), expected_unmasked
        ), f"Expected unmasked to be {expected_unmasked}, got {unmasked.squeeze(-1)}"

        # Check missing values (missing tokens - mask value MISSING)
        expected_missing = torch.tensor([[1, 2, 0], [11, 12, 13]])
        assert torch.equal(
            missing.squeeze(-1), expected_missing
        ), f"Expected missing to be {expected_missing}, got {missing.squeeze(-1)}"

        # Check masks
        assert torch.equal(decoded_mask, torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]]))
        assert torch.equal(unmasked_mask, torch.tensor([[1, 1, 1], [1, 1, 1]]))
        assert torch.equal(missing_mask, torch.tensor([[1, 1, 0], [1, 1, 1]]))

        # Combine the tokens back
        tokens_combined = Predictor.combine_x_y(
            decoded,
            unmasked,
            missing,
            decoded_mask,
            unmasked_mask,
            missing_mask,
            indices,
        )
        logger.info(f"tokens_combined: {tokens_combined}")
        logger.info(f"original tokens: {tokens}")
        # Check the combined tokens
        expected_tokens = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [11, 12, 13, 14, 15, 16, 17, 18, 19],
            ]
        ).unsqueeze(-1)

        assert torch.equal(tokens_combined, expected_tokens), (
            f"Expected combined tokens to be {expected_tokens.squeeze(-1)}, "
            f"got {tokens_combined.squeeze(-1)}"
        )

        # Check that the indices are correct
        expected_indices = torch.tensor(
            [[0, 1, 5, 6, 7, 8, 2, 3, 4], [0, 1, 2, 6, 7, 8, 3, 4, 5]]
        )

        assert torch.equal(
            indices, expected_indices
        ), f"Expected indices {expected_indices}, got {indices}"


class TestTokensAndMasks:
    """Test TestTokensAndMasks."""

    def test_flatten_tokens_and_masks(self) -> None:
        """Test TokensAndMasks.flatten_tokens_and_masks."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        sentinel_2 = torch.ones((b, h, w, t, d))
        sentinel_2[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m = TokensAndMasks(
            sentinel2_l2a=sentinel_2, sentinel2_l2a_mask=sentinel_2_mask
        )
        x, mask = t_and_m.flatten_tokens_and_masks()

        assert x.shape == (b, h * w * t, d)
        assert mask.shape == (b, h * w * t)
        assert (x[mask.bool()] == 0).all()
        assert (x[(1 - mask).bool()] == 1).all()

    def test_pool_unmasked_tokens(self) -> None:
        """Test TokensAndMasks.pool_unmasked_tokens."""
        b, h, w, t, d = 2, 4, 4, 3, 128
        # Setup for mean pooling
        sentinel_2_mean = torch.ones((b, h, w, t, d))
        sentinel_2_mean[0, 0, 0, 0, :] = 0  # set one "token" to 0s
        sentinel_2_mask_mean = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask_mean[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m_mean = TokensAndMasks(
            sentinel2_l2a=sentinel_2_mean, sentinel2_l2a_mask=sentinel_2_mask_mean
        )
        # Setup for max pooling
        sentinel_2_max = torch.ones((b, h, w, t, d)) * 2  # set all tokens to 2
        sentinel_2_max[0, 0, 0, 0, :] = 3  # set one "token" to 3s for max pooling
        sentinel_2_mask_max = torch.zeros((b, h, w, t)).long()
        sentinel_2_mask_max[0, 0, 0, 0] = 1  # set the same token's mask to 1
        t_and_m_max = TokensAndMasks(
            sentinel2_l2a=sentinel_2_max, sentinel2_l2a_mask=sentinel_2_mask_max
        )

        # Test max pooling
        pooled_max = t_and_m_max.pool_unmasked_tokens(PoolingType.MAX)
        assert pooled_max.shape == (b, d)
        assert (pooled_max == 2).all()  # check the 3 tokens have been ignored

        # Test mean pooling
        pooled_mean = t_and_m_mean.pool_unmasked_tokens(PoolingType.MEAN)
        assert pooled_mean.shape == (b, d)
        assert (pooled_mean == 1).all()  # check the 0 tokens have been ignored


# TODO: write a unit test for the FlexiPatchEmbeddings
