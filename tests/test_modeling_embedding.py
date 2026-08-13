"""Tests for the SVD color embedding (modeling/embedding.py).

Fixture note: a naive "one isolated co-occurring pair" toy example
turns out to produce a mathematically real but unintuitive result —
the two colors end up with *orthogonal* embeddings despite having the
strongest possible direct PPMI, because a single off-diagonal pair in
an otherwise-zero symmetric matrix creates a +p/-p eigenvalue pair
that splits into two orthogonal directions. This was discovered
empirically while building this fixture, not assumed — see
`_richer_fixture` below for a fixture with enough connectivity that
cosine similarity actually tracks PPMI strength, which is what's
tested here. The isolated-pair behavior itself is a real, documented
property of this approach (see the module docstring), not a bug.
"""

import numpy as np
import pytest

from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.embedding import ColorEmbedding, fit_multiple


def _richer_fixture() -> CoOccurrenceModel:
    """5 colors: {0,1,2} mutually co-occur a lot, 3 co-occurs moderately
    with each of them, 4 never co-occurs with anything.
    ppmi(0,1) > ppmi(0,3) > ppmi(0,4) == 0, verified in-line below.
    """
    color_counts = np.array([10, 10, 10, 8, 10], dtype=np.int64)
    pair_counts = np.zeros((5, 5), dtype=np.int64)
    for i, j, c in [(0, 1, 7), (0, 2, 6), (1, 2, 6), (0, 3, 4), (1, 3, 3), (2, 3, 3)]:
        pair_counts[i, j] = pair_counts[j, i] = c
    return CoOccurrenceModel(vocab_size=5, n_artworks=30, color_counts=color_counts, pair_counts=pair_counts)


class TestFit:
    def test_vector_shape(self):
        model = _richer_fixture()
        embedding = ColorEmbedding.fit(model, n_components=3)
        assert embedding.vectors.shape == (5, 3)
        assert embedding.n_components == 3

    def test_clips_to_matrix_rank(self):
        model = _richer_fixture()
        embedding = ColorEmbedding.fit(model, n_components=100)
        assert embedding.n_components == 5  # can't exceed vocab_size

    def test_deterministic(self):
        model = _richer_fixture()
        e1 = ColorEmbedding.fit(model, n_components=3)
        e2 = ColorEmbedding.fit(model, n_components=3)
        assert np.array_equal(e1.vectors, e2.vectors)
        assert np.array_equal(e1.singular_values, e2.singular_values)

    def test_explained_variance_ratio_increases_with_components(self):
        model = _richer_fixture()
        ratios = [ColorEmbedding.fit(model, n_components=k).explained_variance_ratio for k in [1, 2, 3, 4, 5]]
        assert all(r2 >= r1 - 1e-9 for r1, r2 in zip(ratios, ratios[1:]))
        assert ratios[-1] == pytest.approx(1.0, abs=1e-9)  # full rank captures everything
        assert 0.0 <= ratios[0] <= 1.0


class TestSimilarity:
    def test_self_similarity_is_one(self):
        model = _richer_fixture()
        embedding = ColorEmbedding.fit(model, n_components=4)
        assert embedding.similarity(0, 0) == pytest.approx(1.0)

    def test_similarity_tracks_ppmi_strength_at_near_full_rank(self):
        model = _richer_fixture()
        assert model.ppmi(0, 1) > model.ppmi(0, 3) > model.ppmi(0, 4) == 0.0
        embedding = ColorEmbedding.fit(model, n_components=5)  # full rank
        assert embedding.similarity(0, 1) > embedding.similarity(0, 3) > embedding.similarity(0, 4)

    def test_isolated_color_has_zero_similarity_to_everything(self):
        # color 4 never co-occurs with anything -> its PPMI row is all
        # zero -> its embedding vector is the zero vector -> similarity
        # falls back to the documented 0.0 (division-by-zero guard)
        model = _richer_fixture()
        embedding = ColorEmbedding.fit(model, n_components=5)
        for other in range(4):
            assert embedding.similarity(4, other) == 0.0


class TestSaveLoad:
    def test_round_trip(self, tmp_path):
        model = _richer_fixture()
        embedding = ColorEmbedding.fit(model, n_components=3)
        path = tmp_path / "embedding.json"
        embedding.save(path)

        loaded = ColorEmbedding.load(path)

        assert loaded.vocab_size == embedding.vocab_size
        assert loaded.n_components == embedding.n_components
        assert np.allclose(loaded.vectors, embedding.vectors)
        assert np.allclose(loaded.singular_values, embedding.singular_values)
        assert loaded.explained_variance_ratio == pytest.approx(embedding.explained_variance_ratio)

    def test_loaded_embedding_gives_identical_similarities(self, tmp_path):
        model = _richer_fixture()
        embedding = ColorEmbedding.fit(model, n_components=3)
        path = tmp_path / "embedding.json"
        embedding.save(path)
        loaded = ColorEmbedding.load(path)
        assert loaded.similarity(0, 1) == pytest.approx(embedding.similarity(0, 1))


class TestFitMultiple:
    def test_returns_one_embedding_per_dimension(self):
        model = _richer_fixture()
        embeddings = fit_multiple(model, [2, 3, 4])
        assert set(embeddings.keys()) == {2, 3, 4}
        assert embeddings[2].n_components == 2
        assert embeddings[3].n_components == 3
        assert embeddings[4].n_components == 4

    def test_embeddings_are_nested_nested_prefixes(self):
        # truncating one SVD at different k must give genuinely nested
        # vectors — the first k columns of the d=4 embedding equal the
        # full d=2 embedding's vectors
        model = _richer_fixture()
        embeddings = fit_multiple(model, [2, 4])
        assert np.allclose(embeddings[4].vectors[:, :2], embeddings[2].vectors)

    def test_matches_individual_fit_calls(self):
        model = _richer_fixture()
        multi = fit_multiple(model, [3])[3]
        single = ColorEmbedding.fit(model, n_components=3)
        assert np.allclose(multi.vectors, single.vectors)
