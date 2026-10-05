"""Train a binary gyaru head using the frozen SigLIP SO400M vision encoder."""

from pathlib import Path

from train_siglip import main


DEFAULT_ENCODER = "google/siglip-so400m-patch14-384"


if __name__ == "__main__":
    main(
        default_encoder=DEFAULT_ENCODER,
        default_output=Path("artifacts/gyaru_siglip_400_heads.safetensors"),
        head_usage_instruction="Start the app with --siglip-400-heads {output} to use it.",
    )
