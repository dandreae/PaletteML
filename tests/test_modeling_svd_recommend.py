"""Tests for the SVD embedding recommender (modeling/svd_recommend.py)."""

import numpy as np
import pytest

from paletteml.modeling.co_occurrence import CoOccurrenceModel
from paletteml.modeling.embedding import ColorEmbedding
from paletteml.modeling.svd_recommend import SvdEmbeddingRecommender
from paletteml.modeling.vocabulary import ColorVocabulary

RANDOM_STATE = 42

LAB_0 = [40.0, 55.0, 35.0]
LAB_1 = [42.0, 50.0, 32.0]
LAB_2 = [60.0, 20.0, 45.0]
LAB_3 = [50.0, -10.0, 20.0]
LAB_4 = [30.0, 10.0, -55.0]


def _vocabulary() -> ColorVocabulary:
    points = np.array([LAB_0, LAB_1, LAB_2, LAB_3, LAB_4] * 3, dtype=np.float64)
    return ColorVocabulary.fit(points, vocab_size=5, random_state=RANDOM_STATE)


def _co_occurrence(vocab: ColorVocabulary) -> tuple[CoOccurrenceModel, dict]:
    ids = {i: vocab.assign(np.array(lab)) for i, lab in enumerate([LAB_0, LAB_1, LAB_2, LAB_3, LAB_4])}
    color_counts = np.zeros(5, dtype=np.int64)
    for i, count in enumerate([10, 10, 10, 8, 10]):
        color_counts[ids[i]] = count
    pair_counts = np.zeros((5, 5), dtype=np.int64)

    def set_pair(a, b, count):
        pair_counts[ids[a], ids[b]] = count
        pair_counts[ids[b], ids[a]] = count

    set_pair(0, 1, 7)
    set_pair(0, 2, 6)
    set_pair(1, 2, 6)
    set_pair(0, 3, 4)
    set_pair(1, 3, 3)
    set_pair(2, 3, 3)
    # color 4 (ids[4]) never co-occurs with anything

    model = CoOccurrenceModel(vocab_size=5, n_artworks=30, color_counts=color_counts, pair_counts=pair_counts)
    return model, ids


@pytest.fixture
def setup():
    vocab = _vocabulary()
    co_occurrence, ids = _co_occurrence(vocab)
    embedding = ColorEmbedding.fit(co_occurrence, n_components=5)  # full rank, see embedding test notes
    return vocab, embedding, ids


def _hex_of(vocab, ids, i):
    return vocab.entries[ids[i]].hex


class TestRecommend:
    def test_ranks_by_cosine_similarity_not_raw_ppmi_order(self, setup):
        # Deliberately NOT asserting "top result == strongest PPMI
        # partner (color 1)". Verified directly: ppmi(0,1)=0.742 >
        # ppmi(0,2)=0.588, but sim(0,2)=0.304 > sim(0,1)=0.195 — cosine
        # similarity does not always preserve PPMI's pairwise ordering,
        # even at full rank, because normalizing by embedding norm
        # (which reflects a color's overall connectivity) can reorder
        # pairs. That's real, verified behavior of this approach, not a
        # bug — exactly the kind of difference from direct PPMI ranking
        # this experiment exists to surface.
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)

        results = recommender.recommend(_hex_of(vocab, ids, 0), top_n=4)

        assert len(results) == 4  # SVD always returns a full ranking (up to top_n)
        assert results[0].cluster_id == ids[2]  # highest cosine similarity, per direct computation
        assert [r.score for r in results] == sorted([r.score for r in results], reverse=True)

    def test_seed_excluded_from_results(self, setup):
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        results = recommender.recommend(_hex_of(vocab, ids, 0), top_n=4)
        assert ids[0] not in [r.cluster_id for r in results]

    def test_always_returns_top_n_unlike_co_occurrence(self, setup):
        # the isolated color (4) has zero PPMI with everything, so the
        # direct co-occurrence recommender would return nothing for it —
        # the SVD recommender should still produce a full dense ranking
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        results = recommender.recommend(_hex_of(vocab, ids, 4), top_n=4)
        assert len(results) == 4

    def test_evidence_present(self, setup):
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        results = recommender.recommend(_hex_of(vocab, ids, 0), top_n=1)
        top = results[0]
        assert len(top.evidence) == 1
        assert top.evidence[0].cosine_similarity == pytest.approx(top.score)


class TestMultiSeed:
    def test_combines_by_averaging_cosine_similarity(self, setup):
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)

        seed0, seed1 = _hex_of(vocab, ids, 0), _hex_of(vocab, ids, 1)
        results = recommender.recommend([seed0, seed1], top_n=3)

        by_id = {r.cluster_id: r for r in results}
        assert ids[2] in by_id
        expected = (embedding.similarity(ids[0], ids[2]) + embedding.similarity(ids[1], ids[2])) / 2
        assert by_id[ids[2]].score == pytest.approx(expected)

    def test_both_seeds_excluded(self, setup):
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        seed0, seed1 = _hex_of(vocab, ids, 0), _hex_of(vocab, ids, 1)
        results = recommender.recommend([seed0, seed1], top_n=3)
        result_ids = [r.cluster_id for r in results]
        assert ids[0] not in result_ids
        assert ids[1] not in result_ids

    def test_duplicate_seeds_not_double_counted(self, setup):
        vocab, embedding, ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        seed0 = _hex_of(vocab, ids, 0)
        single = recommender.recommend(seed0, top_n=4)
        duplicated = recommender.recommend([seed0, seed0], top_n=4)
        assert [r.score for r in single] == pytest.approx([r.score for r in duplicated])


class TestEdgeCases:
    def test_empty_seed_list_raises(self, setup):
        vocab, embedding, _ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        with pytest.raises(ValueError):
            recommender.recommend([], top_n=3)

    def test_vocab_size_mismatch_raises(self, setup):
        vocab, _embedding, _ids = setup
        mismatched = ColorEmbedding(
            vocab_size=3, n_components=2, vectors=np.zeros((3, 2)), singular_values=np.zeros(2),
            explained_variance_ratio=0.0,
        )
        with pytest.raises(ValueError):
            SvdEmbeddingRecommender(vocab, mismatched)

    def test_out_of_distribution_seed_does_not_crash(self, setup):
        vocab, embedding, _ids = setup
        recommender = SvdEmbeddingRecommender(vocab, embedding)
        results = recommender.recommend("#7f7f7f", top_n=3)
        assert isinstance(results, list)
