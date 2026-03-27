"""Property-based tests for GridSearchOrchestrator.

Property 9: 그리드 조합 생성 — 조합 수 = 각 값 리스트 길이의 곱.
"""

import math
from unittest.mock import MagicMock

from hypothesis import given, settings, strategies as st

from experiment.grid_search import GridSearchOrchestrator


# Feature: experiment-validation, Property 9: 그리드 조합 생성
class TestProperty9GridCombinations:
    @settings(max_examples=100, deadline=None)
    @given(
        grid=st.dictionaries(
            keys=st.text(min_size=1, max_size=10,
                         alphabet=st.characters(whitelist_categories=("L",))),
            values=st.lists(
                st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
                min_size=1, max_size=5,
            ),
            min_size=1, max_size=4,
        ),
    )
    def test_combination_count_equals_product(self, grid):
        """조합 수 = 각 값 리스트 길이의 곱."""
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations(grid)

        expected_count = math.prod(len(v) for v in grid.values())
        assert len(combos) == expected_count

    @settings(max_examples=100, deadline=None)
    @given(
        grid=st.dictionaries(
            keys=st.text(min_size=1, max_size=10,
                         alphabet=st.characters(whitelist_categories=("L",))),
            values=st.lists(
                st.integers(min_value=0, max_value=100),
                min_size=1, max_size=4,
            ),
            min_size=1, max_size=3,
        ),
    )
    def test_all_combinations_have_all_keys(self, grid):
        """모든 조합은 그리드의 모든 키를 포함."""
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations(grid)

        for combo in combos:
            assert set(combo.keys()) == set(grid.keys())

    @settings(max_examples=100, deadline=None)
    @given(
        grid=st.dictionaries(
            keys=st.text(min_size=1, max_size=8,
                         alphabet=st.characters(whitelist_categories=("L",))),
            values=st.lists(
                st.integers(min_value=0, max_value=50),
                min_size=1, max_size=4, unique=True,
            ),
            min_size=1, max_size=3,
        ),
    )
    def test_all_combinations_unique(self, grid):
        """모든 조합은 고유 (unique values일 때)."""
        orch = GridSearchOrchestrator(MagicMock())
        combos = orch._generate_combinations(grid)

        combo_tuples = [tuple(sorted(c.items())) for c in combos]
        assert len(combo_tuples) == len(set(combo_tuples))
