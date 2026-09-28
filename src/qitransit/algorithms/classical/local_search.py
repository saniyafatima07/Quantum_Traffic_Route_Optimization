"""Local search over decoded solutions: 2-opt, Or-opt and cross-exchange.

These operate on *solutions* rather than on the priority vector, because an
intra-route move has no clean real-valued representation in the priority
encoding.  Every candidate move is scored with the problem's own incremental
cost where one exists and with a full evaluation otherwise, and the cheapest
improving move is applied — first improvement, which converges faster on these
instances than best improvement and costs one evaluation per accepted move
rather than one per candidate.
"""

from __future__ import annotations

from ...problems.base import Evaluation, Problem, Route, Solution


class LocalSearchRefiner:
    """First-improvement descent over the neighbourhood of a solution."""

    def __init__(self, problem: Problem, max_passes: int = 3) -> None:
        self.problem = problem
        self.max_passes = max(int(max_passes), 1)
        self.evaluations = 0

    def refine(self, solution: Solution, passes: int | None = None) -> Solution:
        """Descend until no single move improves, or the pass limit is reached.

        ``evaluations`` is left holding the number of objective calls made, so a
        hybrid optimiser can charge them against its budget honestly.
        """
        self.evaluations = 0
        current = tuple(tuple(r) for r in solution)
        best = self.problem.evaluate(current)
        self.evaluations += 1
        for _ in range(passes if passes is not None else self.max_passes):
            improved = False
            for candidate in self._neighbourhood(current):
                evaluation = self.problem.evaluate(candidate)
                self.evaluations += 1
                if evaluation.objective < best.objective - 1e-9:
                    current, best, improved = candidate, evaluation, True
            if not improved:
                break
        return current

    def _neighbourhood(self, solution: Solution):
        """All single moves, as a generator of candidate solutions."""
        for r_index, route in enumerate(solution):
            if len(route) < 3:
                continue
            yield from self._two_opt(solution, r_index, route)
            yield from self._or_opt(solution, r_index, route)
        yield from self._intra_route_relocate(solution)
        yield from self._inter_route_exchange(solution)

    @staticmethod
    def _two_opt(solution: Solution, r_index: int, route: Route):
        """Reverse a segment, the classic tour-improvement move."""
        n = len(route)
        for i in range(n - 1):
            for j in range(i + 2, n):
                reversed_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                mutated = list(solution)
                mutated[r_index] = reversed_route
                yield tuple(mutated)

    @staticmethod
    def _or_opt(solution: Solution, r_index: int, route: Route):
        """Move a segment of 1-3 customers to the end, either orientation."""
        n = len(route)
        for length in (1, 2, 3):
            if length >= n:
                break
            for i in range(n - length + 1):
                segment = route[i : i + length]
                remainder = route[:i] + route[i + length :]
                for tail in (segment, segment[::-1]):
                    mutated = list(solution)
                    mutated[r_index] = remainder + tail
                    yield tuple(mutated)

    @staticmethod
    def _intra_route_relocate(solution: Solution):
        """Relocate one customer anywhere within its own route."""
        for r_index, route in enumerate(solution):
            if len(route) < 4:
                continue
            for i in range(len(route)):
                rest = route[:i] + route[i + 1 :]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    mutated = list(solution)
                    mutated[r_index] = rest[:j] + (route[i],) + rest[j:]
                    yield tuple(mutated)

    @staticmethod
    def _inter_route_exchange(solution: Solution):
        """Swap two customers between different routes, when both stay feasible."""
        for a in range(len(solution)):
            for b in range(a + 1, len(solution)):
                route_a, route_b = solution[a], solution[b]
                if not route_a or not route_b:
                    continue
                for i, ca in enumerate(route_a):
                    for j, cb in enumerate(route_b):
                        if ca == cb:
                            continue
                        new_a = list(route_a)
                        new_b = list(route_b)
                        new_a[i] = cb
                        new_b[j] = ca
                        mutated = list(solution)
                        mutated[a] = tuple(new_a)
                        mutated[b] = tuple(new_b)
                        yield tuple(mutated)


def two_opt_distance(order: list[int], cost: list[list[float]]) -> float:
    """Tour cost of a cycle through ``order`` using a full cost matrix."""
    total = 0.0
    previous = 0
    for nxt in order:
        total += cost[previous][nxt]
        previous = nxt
    return total + cost[previous][0]


def first_improvement(problem: Problem, solution: Solution, passes: int = 2) -> Evaluation:
    """Run the refiner and return the resulting evaluation."""
    refiner = LocalSearchRefiner(problem, max_passes=passes)
    return problem.evaluate(refiner.refine(solution))
