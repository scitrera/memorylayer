"""Smoke test for ColPaliEmbeddingProvider with a real (tiny) model.

Marked ``slow`` and gated on ``colpali_engine`` + ``torch`` being importable.
Pulls ``ModernVBERT/colmodernvbert`` (the smallest supported model, MIT
licensed) and exercises ``embed_text_multivector`` + ``embed_image_multivector``
to prove the GPU/CPU code path works end-to-end.

Skipped by default in regular pytest runs. To execute::

    pytest -m slow oss/memorylayer-embed-server/tests/unit/test_colpali_provider_real.py
"""

from __future__ import annotations

import io

import pytest

# Heavy deps are optional — gate the whole module on them.
colpali_engine = pytest.importorskip("colpali_engine")
torch = pytest.importorskip("torch")
PIL_Image = pytest.importorskip("PIL.Image")


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def colpali_provider():
    """Module-scoped so the (slow) model load happens once."""
    from memorylayer_embed_server.services.embedding.colpali import (
        ColPaliEmbeddingProvider,
    )

    provider = ColPaliEmbeddingProvider(
        v=None,
        model_name="ModernVBERT/colmodernvbert",
        device="cpu",  # keep deterministic and avoid GPU requirements
        revision="main",
    )
    return provider


async def test_real_colpali_embed_text_multivector(colpali_provider):
    mv = await colpali_provider.embed_text_multivector("hello world")
    assert mv.num_vectors > 0
    assert mv.dimensions > 0
    # Vectors are floats and non-empty
    assert all(isinstance(v, float) for v in mv.vectors[0])


async def test_real_colpali_embed_image_multivector(colpali_provider):
    # Build a 32×32 RGB PNG entirely in memory — no fixtures on disk.
    img = PIL_Image.new("RGB", (32, 32), color=(127, 200, 64))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    mv = await colpali_provider.embed_image_multivector(image_bytes)
    assert mv.num_vectors > 0
    assert mv.dimensions > 0


def test_real_colpali_maxsim_self_score_is_max(colpali_provider):
    """A document scored against itself should rank above unrelated docs."""
    import asyncio

    async def _go():
        q = await colpali_provider.embed_text_multivector("apple orchards")
        d_same = await colpali_provider.embed_text_multivector("apple orchards")
        d_other = await colpali_provider.embed_text_multivector("quantum chromodynamics lecture")
        from memorylayer_server.services.embedding._maxsim import maxsim_score

        same = maxsim_score(q, d_same)
        other = maxsim_score(q, d_other)
        return same, other

    same, other = asyncio.run(_go())
    assert same > other
