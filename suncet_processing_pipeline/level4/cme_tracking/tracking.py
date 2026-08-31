"""Cadence-independent temporal tracking of a polar CME leading front.

Tracking operates in pixels per nominal frame. Observation timestamps affect
the later kinematic fit, not which feature is selected. The automatic two-stage
method first links simultaneous angular ridge fragments over the full circle,
uses that coherent motion to infer an event sector, and then recovers the
detailed ``r(time, position_angle)`` front with sparse per-angle paths inside
that data-derived sector. The individual stages remain selectable for tests and
historical reproduction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import IntEnum
import math

import numpy as np
from scipy.signal import find_peaks

from .likelihood import LikelihoodResult
from .polar import PolarGrid


class TrackingError(ValueError):
    """Raised when a likelihood cube or tracking configuration is invalid."""


class FrontState(IntEnum):
    """State of one sample in the retained ``r(time, position_angle)`` front."""

    MISSING = 0
    OBSERVED = 1
    REJECTED = 2


@dataclass(frozen=True)
class TrackingConfig:
    """Configuration for deterministic discovery and sparse-path refinement.

    Motion constraints are in pixels per nominal frame interval. Reviewed
    frame-number gaps therefore change the allowed displacement between two
    retained images, while a corrected synthetic cadence changes only the
    later speed and acceleration products.
    """

    score_threshold: float = 2.5
    maximum_candidates_per_frame: int = 5
    minimum_peak_separation_bins: int = 2
    maximum_outward_step_px_per_frame: float = 20.0
    inward_localization_tolerance_px: float = 2.0
    maximum_gap_frames: int = 2
    observation_reward: float = 0.5
    gap_penalty: float = 0.75
    radial_jump_penalty: float = 0.15
    minimum_track_points: int = 4
    minimum_outward_displacement_px: float = 2.0
    maximum_angular_gap_bins: int = 1
    minimum_angular_support_deg: float = 8.0
    angular_consistency_half_width_deg: float = 6.0
    angular_outlier_tolerance_px: float = 8.0
    minimum_angular_neighbors: int = 3
    minimum_event_frames: int = 4
    minimum_observed_angles_per_frame: int = 3
    position_angle_window_deg: tuple[float, float] | None = None
    association_method: str = "coherent_fragments"
    minimum_fragment_overlap_fraction: float = 0.25
    minimum_fragment_angular_occupancy: float = 0.60
    automatic_sector_padding_deg: float = 16.0
    refinement_maximum_outward_step_px_per_frame: float = 20.0
    refinement_maximum_gap_frames: int = 2
    refinement_minimum_track_points: int = 4
    refinement_minimum_outward_displacement_px: float = 2.0
    refinement_minimum_event_frames: int = 4
    refinement_minimum_observed_angles_per_frame: int = 3

    def __post_init__(self) -> None:
        for name, value in (
            ("score_threshold", self.score_threshold),
            (
                "maximum_outward_step_px_per_frame",
                self.maximum_outward_step_px_per_frame,
            ),
            (
                "inward_localization_tolerance_px",
                self.inward_localization_tolerance_px,
            ),
            ("observation_reward", self.observation_reward),
            ("gap_penalty", self.gap_penalty),
            ("radial_jump_penalty", self.radial_jump_penalty),
            (
                "minimum_outward_displacement_px",
                self.minimum_outward_displacement_px,
            ),
            ("minimum_angular_support_deg", self.minimum_angular_support_deg),
            (
                "angular_consistency_half_width_deg",
                self.angular_consistency_half_width_deg,
            ),
            ("angular_outlier_tolerance_px", self.angular_outlier_tolerance_px),
        ):
            if not math.isfinite(value) or value < 0:
                raise TrackingError(f"{name} must be finite and nonnegative.")
        if self.maximum_outward_step_px_per_frame <= 0:
            raise TrackingError(
                "maximum_outward_step_px_per_frame must be positive."
            )
        if (
            not isinstance(self.association_method, str)
            or self.association_method not in {
                "coherent_fragments",
                "coherent_sector_refined_paths",
                "independent_angle_paths",
            }
        ):
            raise TrackingError(
                "association_method must be 'coherent_fragments', "
                "'coherent_sector_refined_paths', or "
                "'independent_angle_paths'."
            )
        for name, value in (
            ("automatic_sector_padding_deg", self.automatic_sector_padding_deg),
            (
                "refinement_minimum_outward_displacement_px",
                self.refinement_minimum_outward_displacement_px,
            ),
        ):
            if not math.isfinite(value) or value < 0:
                raise TrackingError(f"{name} must be finite and nonnegative.")
        if self.automatic_sector_padding_deg >= 180.0:
            raise TrackingError("automatic_sector_padding_deg must be less than 180.")
        if (
            not math.isfinite(self.refinement_maximum_outward_step_px_per_frame)
            or self.refinement_maximum_outward_step_px_per_frame <= 0
        ):
            raise TrackingError(
                "refinement_maximum_outward_step_px_per_frame must be finite "
                "and positive."
            )
        if (
            not math.isfinite(self.minimum_fragment_overlap_fraction)
            or not 0.0 < self.minimum_fragment_overlap_fraction <= 1.0
        ):
            raise TrackingError(
                "minimum_fragment_overlap_fraction must lie in (0, 1]."
            )
        if (
            not math.isfinite(self.minimum_fragment_angular_occupancy)
            or not 0.0 < self.minimum_fragment_angular_occupancy <= 1.0
        ):
            raise TrackingError(
                "minimum_fragment_angular_occupancy must lie in (0, 1]."
            )
        for name, value, minimum in (
            ("maximum_candidates_per_frame", self.maximum_candidates_per_frame, 1),
            ("minimum_peak_separation_bins", self.minimum_peak_separation_bins, 1),
            ("maximum_gap_frames", self.maximum_gap_frames, 0),
            ("minimum_track_points", self.minimum_track_points, 2),
            ("maximum_angular_gap_bins", self.maximum_angular_gap_bins, 0),
            ("minimum_angular_neighbors", self.minimum_angular_neighbors, 1),
            ("minimum_event_frames", self.minimum_event_frames, 1),
            (
                "minimum_observed_angles_per_frame",
                self.minimum_observed_angles_per_frame,
                1,
            ),
            (
                "refinement_maximum_gap_frames",
                self.refinement_maximum_gap_frames,
                0,
            ),
            (
                "refinement_minimum_track_points",
                self.refinement_minimum_track_points,
                2,
            ),
            (
                "refinement_minimum_event_frames",
                self.refinement_minimum_event_frames,
                1,
            ),
            (
                "refinement_minimum_observed_angles_per_frame",
                self.refinement_minimum_observed_angles_per_frame,
                1,
            ),
        ):
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise TrackingError(f"{name} must be an integer >= {minimum}.")
        if self.position_angle_window_deg is not None:
            try:
                window = tuple(float(value) for value in self.position_angle_window_deg)
            except (TypeError, ValueError) as exc:
                raise TrackingError(
                    "position_angle_window_deg must contain two numeric values."
                ) from exc
            if len(window) != 2 or any(
                not math.isfinite(value) or not 0.0 <= value < 360.0
                for value in window
            ):
                raise TrackingError(
                    "position_angle_window_deg must contain two finite values in "
                    "[0, 360)."
                )
            if window[0] == window[1]:
                raise TrackingError(
                    "position_angle_window_deg endpoints must differ; use null "
                    "for the full circle."
                )
            object.__setattr__(self, "position_angle_window_deg", window)


@dataclass(frozen=True)
class FrontTrack:
    """Angularly resolved CME front returned by the deterministic tracker."""

    radius_px: np.ndarray
    radius_rsun: np.ndarray
    radial_sigma_px: np.ndarray
    score: np.ndarray
    state: np.ndarray
    position_angle_deg: np.ndarray
    event_detected: bool
    quality_flags: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = self.radius_px.shape
        if len(expected) != 2:
            raise TrackingError(
                "Front samples must have shape (time, position_angle)."
            )
        for name, values in (
            ("radius_rsun", self.radius_rsun),
            ("radial_sigma_px", self.radial_sigma_px),
            ("score", self.score),
            ("state", self.state),
        ):
            if values.shape != expected:
                raise TrackingError(f"{name} shape does not match radius_px.")
        if self.position_angle_deg.shape != (expected[1],):
            raise TrackingError(
                "position_angle_deg must contain one value per angular column."
            )
        observed = self.state == int(FrontState.OBSERVED)
        if np.any(~np.isfinite(self.radius_px[observed])):
            raise TrackingError("Observed front samples must have finite radius.")
        if np.any(np.isfinite(self.radius_px[~observed])):
            raise TrackingError(
                "Missing or rejected front radii must remain represented by NaN."
            )

    @property
    def observed_mask(self) -> np.ndarray:
        """Boolean mask for measured, noninterpolated front samples."""

        return self.state == int(FrontState.OBSERVED)


@dataclass(frozen=True)
class _Candidate:
    radius_index: int
    score: float
    sigma_px: float


@dataclass(frozen=True)
class _Path:
    nodes: tuple[tuple[int, _Candidate], ...]
    objective: float

    @property
    def point_count(self) -> int:
        return len(self.nodes)


@dataclass(frozen=True)
class _FrontFragment:
    """One simultaneous, radially continuous angular ridge in a frame."""

    frame_index: int
    nodes: tuple[tuple[int, _Candidate], ...]
    objective: float
    median_radius_px: float

    @property
    def angle_indices(self) -> tuple[int, ...]:
        return tuple(angle_index for angle_index, _ in self.nodes)

    @property
    def candidate_by_angle(self) -> dict[int, _Candidate]:
        return dict(self.nodes)


@dataclass(frozen=True)
class _FragmentPath:
    """Temporally associated sequence of coherent front fragments."""

    fragments: tuple[_FrontFragment, ...]
    objective: float


def _robust_fragment_displacement_px(
    fragments: Sequence[_FrontFragment],
    *,
    endpoint_fraction: float = 0.10,
) -> float:
    """Return outward displacement without trusting either single endpoint.

    Particle-contaminated likelihood cubes can produce one anomalous fragment
    at the start or end of an otherwise weak path.  Comparing short endpoint
    medians keeps such a fragment from defining event-scale motion while
    preserving the displacement of a consistently propagating front.
    """

    # A fractional window scales with event duration and remains disjoint for
    # every nonempty path. Short analytic paths retain their exact endpoints;
    # long mission sequences cannot pass because of a brief late excursion.
    radii_px = np.asarray(
        [fragment.median_radius_px for fragment in fragments],
        dtype=np.float64,
    )
    if len(fragments) < 10:
        sample_count = 1
    else:
        sample_count = min(
            len(fragments) // 2,
            max(3, int(math.ceil(endpoint_fraction * len(fragments)))),
        )
    start_radius_px = float(
        np.median(
            [
                fragment.median_radius_px for fragment in fragments[:sample_count]
            ]
        )
    )
    end_radius_px = float(
        np.median(
            [
                fragment.median_radius_px for fragment in fragments[-sample_count:]
            ]
        )
    )
    endpoint_displacement_px = end_radius_px - start_radius_px
    if len(fragments) < 10:
        return endpoint_displacement_px

    # Also require the motion to occupy a meaningful fraction of the history.
    # A long stationary arc followed by a few allowed jumps can move its endpoint
    # median, but it cannot create a large central 80-percent radial span.
    distributed_span_px = float(
        np.quantile(radii_px, 0.90) - np.quantile(radii_px, 0.10)
    )
    return min(endpoint_displacement_px, distributed_span_px)


def _peak_sigma_px(
    score: np.ndarray,
    peak_index: int,
    threshold: float,
    radial_step_px: float,
) -> float:
    """Estimate radial localization width from the threshold-relative peak."""

    peak_value = float(score[peak_index])
    half_level = threshold + 0.5 * max(peak_value - threshold, 0.0)
    left = peak_index
    while left > 0 and np.isfinite(score[left - 1]) and score[left - 1] >= half_level:
        left -= 1
    right = peak_index
    while (
        right + 1 < score.size
        and np.isfinite(score[right + 1])
        and score[right + 1] >= half_level
    ):
        right += 1
    full_width_px = max((right - left + 1) * radial_step_px, radial_step_px)
    return full_width_px / 2.355


def _frame_candidates(
    score: np.ndarray,
    grid: PolarGrid,
    config: TrackingConfig,
) -> tuple[_Candidate, ...]:
    finite = np.isfinite(score)
    if not np.any(finite):
        return ()
    working = np.where(finite, score, -np.inf)

    # No local or endpoint peak can qualify when the row-wide maximum is below
    # threshold.  Avoiding SciPy peak construction for these empty rows is a
    # result-preserving fast path and matters for frames x position angles.
    global_index = int(np.nanargmax(working))
    if working[global_index] < config.score_threshold:
        return ()

    peak_indices, _ = find_peaks(
        working,
        height=config.score_threshold,
        distance=config.minimum_peak_separation_bins,
    )

    # scipy intentionally excludes array endpoints.  Include a qualifying
    # global maximum when that would otherwise erase a real FOV-edge sample.
    if global_index not in peak_indices:
        peak_indices = np.append(peak_indices, global_index)

    if peak_indices.size == 0:
        return ()
    ranked = sorted(
        (int(index) for index in peak_indices),
        key=lambda index: (-float(working[index]), -index),
    )[: config.maximum_candidates_per_frame]
    radial_step_px = grid.radial_step_px
    return tuple(
        _Candidate(
            radius_index=index,
            score=float(working[index]),
            sigma_px=_peak_sigma_px(
                working,
                index,
                config.score_threshold,
                radial_step_px,
            ),
        )
        for index in ranked
    )


def _track_one_position_angle(
    angle_score: np.ndarray,
    grid: PolarGrid,
    config: TrackingConfig,
    frame_numbers: np.ndarray,
) -> _Path | None:
    """Find the best outward sparse path for one position angle."""

    frame_candidates = tuple(
        _frame_candidates(frame_score, grid, config) for frame_score in angle_score
    )
    frame_count = len(frame_candidates)
    best_objective: list[np.ndarray] = [
        np.full(len(candidates), -np.inf, dtype=np.float64)
        for candidates in frame_candidates
    ]
    best_length: list[np.ndarray] = [
        np.zeros(len(candidates), dtype=np.int32) for candidates in frame_candidates
    ]
    predecessor: list[list[tuple[int, int] | None]] = [
        [None] * len(candidates) for candidates in frame_candidates
    ]

    for frame_index, candidates in enumerate(frame_candidates):
        for candidate_index, candidate in enumerate(candidates):
            node_reward = (
                candidate.score - config.score_threshold + config.observation_reward
            )
            best_objective[frame_index][candidate_index] = node_reward
            best_length[frame_index][candidate_index] = 1

            current_radius_px = grid.radius_px[candidate.radius_index]
            earliest_frame_number = (
                frame_numbers[frame_index] - config.maximum_gap_frames - 1
            )
            first_previous_frame = int(
                np.searchsorted(
                    frame_numbers,
                    earliest_frame_number,
                    side="left",
                )
            )
            for previous_frame in range(first_previous_frame, frame_index):
                frame_delta = int(
                    frame_numbers[frame_index] - frame_numbers[previous_frame]
                )
                missing_frame_count = frame_delta - 1
                for previous_index, previous_candidate in enumerate(
                    frame_candidates[previous_frame]
                ):
                    previous_objective = best_objective[previous_frame][previous_index]
                    if not np.isfinite(previous_objective):
                        continue
                    previous_radius_px = grid.radius_px[
                        previous_candidate.radius_index
                    ]
                    radial_delta_px = current_radius_px - previous_radius_px
                    if radial_delta_px < -config.inward_localization_tolerance_px:
                        continue
                    maximum_delta_px = (
                        config.maximum_outward_step_px_per_frame * frame_delta
                    )
                    if radial_delta_px > maximum_delta_px:
                        continue

                    positive_delta_fraction = max(radial_delta_px, 0.0) / maximum_delta_px
                    transition_penalty = (
                        config.gap_penalty * missing_frame_count
                        + config.radial_jump_penalty * positive_delta_fraction**2
                    )
                    objective = previous_objective + node_reward - transition_penalty
                    length = int(best_length[previous_frame][previous_index]) + 1
                    current_key = (
                        best_objective[frame_index][candidate_index],
                        int(best_length[frame_index][candidate_index]),
                    )
                    proposed_key = (objective, length)
                    if proposed_key > current_key:
                        best_objective[frame_index][candidate_index] = objective
                        best_length[frame_index][candidate_index] = length
                        predecessor[frame_index][candidate_index] = (
                            previous_frame,
                            previous_index,
                        )

    possible_ends: list[tuple[float, int, int, int]] = []
    for frame_index in range(frame_count):
        for candidate_index in range(len(frame_candidates[frame_index])):
            possible_ends.append(
                (
                    float(best_objective[frame_index][candidate_index]),
                    int(best_length[frame_index][candidate_index]),
                    frame_index,
                    candidate_index,
                )
            )
    if not possible_ends:
        return None
    objective, length, frame_index, candidate_index = max(possible_ends)
    if length < config.minimum_track_points:
        return None

    reversed_nodes: list[tuple[int, _Candidate]] = []
    while True:
        reversed_nodes.append(
            (frame_index, frame_candidates[frame_index][candidate_index])
        )
        previous = predecessor[frame_index][candidate_index]
        if previous is None:
            break
        frame_index, candidate_index = previous
    nodes = tuple(reversed(reversed_nodes))
    displacement_px = (
        grid.radius_px[nodes[-1][1].radius_index]
        - grid.radius_px[nodes[0][1].radius_index]
    )
    if displacement_px < config.minimum_outward_displacement_px:
        return None
    return _Path(nodes=nodes, objective=float(objective))


def _validated_frame_numbers(
    frame_numbers: Sequence[int] | np.ndarray | None,
    frame_count: int,
) -> np.ndarray:
    """Return a read-only, strictly increasing nominal frame coordinate."""

    if frame_numbers is None:
        values = np.arange(frame_count, dtype=np.int64)
    else:
        supplied = tuple(frame_numbers)
        if len(supplied) != frame_count:
            raise TrackingError(
                "frame_numbers must contain one integer per likelihood frame."
            )
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in supplied
        ):
            raise TrackingError("frame_numbers must contain integers.")
        values = np.asarray(supplied, dtype=np.int64)
        if np.any(np.diff(values) <= 0):
            raise TrackingError("frame_numbers must be strictly increasing.")
    values.setflags(write=False)
    return values


def _position_angle_window_mask(
    position_angle_deg: np.ndarray,
    window_deg: tuple[float, float] | None,
) -> np.ndarray:
    """Return a wrap-aware inclusive mask for a reviewed angular window."""

    if window_deg is None:
        return np.ones(position_angle_deg.shape, dtype=bool)
    start_deg, end_deg = window_deg
    angles = np.mod(np.asarray(position_angle_deg, dtype=np.float64), 360.0)
    if start_deg < end_deg:
        return (angles >= start_deg) & (angles <= end_deg)
    return (angles >= start_deg) | (angles <= end_deg)


def _close_short_circular_gaps(mask: np.ndarray, maximum_gap: int) -> np.ndarray:
    """Close short circular false runs for component grouping only."""

    closed = np.asarray(mask, dtype=bool).copy()
    if maximum_gap == 0 or not np.any(closed) or np.all(closed):
        return closed
    count = closed.size
    true_indices = np.flatnonzero(closed)
    for index, start in enumerate(true_indices):
        end = true_indices[(index + 1) % true_indices.size]
        gap = (end - start - 1) % count
        if 0 < gap <= maximum_gap:
            closed[(start + np.arange(1, gap + 1)) % count] = True
    return closed


def _circular_components(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return contiguous true components on a circular one-dimensional grid."""

    values = np.asarray(mask, dtype=bool)
    count = values.size
    if not np.any(values):
        return ()
    if np.all(values):
        return (np.arange(count, dtype=np.int64),)

    false_anchor = int(np.flatnonzero(~values)[0])
    components: list[np.ndarray] = []
    current: list[int] = []
    for offset in range(1, count + 1):
        index = (false_anchor + offset) % count
        if values[index]:
            current.append(index)
        elif current:
            components.append(np.asarray(current, dtype=np.int64))
            current = []
    if current:
        components.append(np.asarray(current, dtype=np.int64))
    return tuple(components)


def _smallest_circular_span_bins(
    angle_indices: Sequence[int],
    angle_count: int,
) -> int:
    """Number of grid bins in the shortest inclusive arc covering samples."""

    ordered = np.unique(np.asarray(angle_indices, dtype=np.int64))
    if ordered.size == 0:
        return 0
    if ordered.size == 1:
        return 1
    circular_steps = np.diff(
        np.concatenate([ordered, ordered[:1] + angle_count])
    )
    return int(angle_count - np.max(circular_steps) + 1)


def _coherent_fragments_in_frame(
    frame_score: np.ndarray,
    frame_index: int,
    grid: PolarGrid,
    config: TrackingConfig,
    angle_window_mask: np.ndarray,
) -> tuple[_FrontFragment, ...]:
    """Build simultaneous angular ridges before performing temporal linking.

    Nodes are radial score peaks.  Two nodes belong to the same provisional
    ridge only when their position-angle bins are adjacent (apart from the
    explicitly allowed short angular gap) and their radii agree within the
    existing angular-localization tolerance.  This ordering is intentional:
    compact transient hits cannot borrow support from unrelated angles at
    unrelated times.
    """

    angle_count = frame_score.shape[0]
    candidates_by_angle: list[tuple[_Candidate, ...]] = []
    for angle_index in range(angle_count):
        if angle_window_mask[angle_index]:
            candidates_by_angle.append(
                _frame_candidates(frame_score[angle_index], grid, config)
            )
        else:
            candidates_by_angle.append(())

    nodes: list[tuple[int, _Candidate]] = []
    node_ids_by_angle: list[list[int]] = [[] for _ in range(angle_count)]
    for angle_index, candidates in enumerate(candidates_by_angle):
        for candidate in candidates:
            node_ids_by_angle[angle_index].append(len(nodes))
            nodes.append((angle_index, candidate))
    if not nodes:
        return ()

    parent = np.arange(len(nodes), dtype=np.int64)

    def find(node_id: int) -> int:
        root = node_id
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[node_id]) != node_id:
            next_id = int(parent[node_id])
            parent[node_id] = root
            node_id = next_id
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    maximum_angle_step = min(
        config.maximum_angular_gap_bins + 1,
        max(angle_count - 1, 0),
    )
    radial_tolerance_px = config.angular_outlier_tolerance_px
    for angle_index, node_ids in enumerate(node_ids_by_angle):
        for angle_step in range(1, maximum_angle_step + 1):
            neighbor_index = (angle_index + angle_step) % angle_count
            for node_id in node_ids:
                first_candidate = nodes[node_id][1]
                first_radius_px = grid.radius_px[first_candidate.radius_index]
                for neighbor_id in node_ids_by_angle[neighbor_index]:
                    second_candidate = nodes[neighbor_id][1]
                    second_radius_px = grid.radius_px[second_candidate.radius_index]
                    if (
                        abs(second_radius_px - first_radius_px)
                        <= radial_tolerance_px
                    ):
                        union(node_id, neighbor_id)

    grouped_node_ids: dict[int, list[int]] = {}
    for node_id in range(len(nodes)):
        grouped_node_ids.setdefault(find(node_id), []).append(node_id)

    minimum_fragment_bins = max(
        config.minimum_observed_angles_per_frame,
        int(
            math.ceil(
                config.minimum_angular_support_deg
                / grid.position_angle_step_deg
            )
        ),
    )
    fragments: list[_FrontFragment] = []
    for component_node_ids in grouped_node_ids.values():
        # A branching graph can contain more than one radial candidate at an
        # angle. Keep one deterministic, highest-evidence sample per angle.
        selected_by_angle: dict[int, _Candidate] = {}
        for node_id in component_node_ids:
            angle_index, candidate = nodes[node_id]
            previous = selected_by_angle.get(angle_index)
            if previous is None or (candidate.score, candidate.radius_index) > (
                previous.score,
                previous.radius_index,
            ):
                selected_by_angle[angle_index] = candidate
        if len(selected_by_angle) < minimum_fragment_bins:
            continue

        angular_span_bins = _smallest_circular_span_bins(
            tuple(selected_by_angle),
            angle_count,
        )
        angular_occupancy = len(selected_by_angle) / angular_span_bins
        if angular_occupancy < config.minimum_fragment_angular_occupancy:
            continue

        selected_nodes = tuple(sorted(selected_by_angle.items()))
        node_rewards = np.asarray(
            [
                candidate.score
                - config.score_threshold
                + config.observation_reward
                for _, candidate in selected_nodes
            ],
            dtype=np.float64,
        )
        # A coherent ridge supplies more evidence as it spans more angles, but
        # those samples are correlated by the PA smoothing. The square-root
        # support gain rewards broad fronts without the grid-resolution and
        # percolation bias of a raw sum; a mean alone would throw away the
        # defining angular-coherence information.
        objective = float(np.mean(node_rewards) * math.sqrt(node_rewards.size))
        median_radius_px = float(
            np.median(
                [
                    grid.radius_px[candidate.radius_index]
                    for _, candidate in selected_nodes
                ]
            )
        )
        fragments.append(
            _FrontFragment(
                frame_index=frame_index,
                nodes=selected_nodes,
                objective=objective,
                median_radius_px=median_radius_px,
            )
        )

    return tuple(
        sorted(
            fragments,
            key=lambda fragment: (
                -fragment.objective,
                fragment.median_radius_px,
                fragment.angle_indices,
            ),
        )
    )


def _fragment_transition_penalty(
    previous: _FrontFragment,
    current: _FrontFragment,
    grid: PolarGrid,
    config: TrackingConfig,
    frame_delta: int,
) -> float | None:
    """Return a temporal-link penalty, or ``None`` for incoherent fragments."""

    previous_by_angle = previous.candidate_by_angle
    current_by_angle = current.candidate_by_angle
    common_angles = sorted(previous_by_angle.keys() & current_by_angle.keys())
    smaller_support = min(len(previous_by_angle), len(current_by_angle))
    required_overlap = max(
        min(config.minimum_angular_neighbors, smaller_support),
        int(math.ceil(config.minimum_fragment_overlap_fraction * smaller_support)),
    )
    if len(common_angles) < required_overlap:
        return None

    radial_deltas_px = np.asarray(
        [
            grid.radius_px[current_by_angle[index].radius_index]
            - grid.radius_px[previous_by_angle[index].radius_index]
            for index in common_angles
        ],
        dtype=np.float64,
    )
    radial_delta_px = float(np.median(radial_deltas_px))
    maximum_delta_px = config.maximum_outward_step_px_per_frame * frame_delta
    if radial_delta_px < -config.inward_localization_tolerance_px:
        return None
    if radial_delta_px > maximum_delta_px:
        return None

    shape_residual_px = float(
        np.median(np.abs(radial_deltas_px - radial_delta_px))
    )
    shape_tolerance_px = config.angular_outlier_tolerance_px
    if shape_residual_px > shape_tolerance_px:
        return None

    positive_delta_fraction = max(radial_delta_px, 0.0) / maximum_delta_px
    shape_fraction = (
        shape_residual_px / shape_tolerance_px
        if shape_tolerance_px > 0
        else 0.0
    )
    missing_frame_count = frame_delta - 1
    # Fragment evidence receives a square-root angular-support gain, so keep
    # transition costs in the same units.
    support_scale = math.sqrt(float(smaller_support))
    return support_scale * float(
        config.gap_penalty * missing_frame_count
        + config.radial_jump_penalty
        * (positive_delta_fraction**2 + shape_fraction**2)
    )


def _link_coherent_fragments(
    fragments_by_frame: tuple[tuple[_FrontFragment, ...], ...],
    grid: PolarGrid,
    config: TrackingConfig,
    frame_numbers: np.ndarray,
) -> _FragmentPath | None:
    """Select one persistent, outward-moving path of simultaneous ridges."""

    best_objective: list[np.ndarray] = [
        np.full(len(fragments), -np.inf, dtype=np.float64)
        for fragments in fragments_by_frame
    ]
    best_length: list[np.ndarray] = [
        np.zeros(len(fragments), dtype=np.int32)
        for fragments in fragments_by_frame
    ]
    best_running_max_radius_px: list[np.ndarray] = [
        np.full(len(fragments), np.nan, dtype=np.float64)
        for fragments in fragments_by_frame
    ]
    predecessor: list[list[tuple[int, int] | None]] = [
        [None] * len(fragments) for fragments in fragments_by_frame
    ]

    for frame_index, fragments in enumerate(fragments_by_frame):
        earliest_frame_number = (
            frame_numbers[frame_index] - config.maximum_gap_frames - 1
        )
        first_previous_frame = int(
            np.searchsorted(
                frame_numbers,
                earliest_frame_number,
                side="left",
            )
        )
        for fragment_index, fragment in enumerate(fragments):
            best_objective[frame_index][fragment_index] = fragment.objective
            best_length[frame_index][fragment_index] = 1
            best_running_max_radius_px[frame_index][fragment_index] = (
                fragment.median_radius_px
            )
            for previous_frame in range(first_previous_frame, frame_index):
                frame_delta = int(
                    frame_numbers[frame_index] - frame_numbers[previous_frame]
                )
                for previous_index, previous_fragment in enumerate(
                    fragments_by_frame[previous_frame]
                ):
                    previous_objective = best_objective[previous_frame][
                        previous_index
                    ]
                    if not np.isfinite(previous_objective):
                        continue
                    previous_running_max_radius_px = best_running_max_radius_px[
                        previous_frame
                    ][previous_index]
                    if (
                        fragment.median_radius_px
                        < previous_running_max_radius_px
                        - config.inward_localization_tolerance_px
                    ):
                        # A one-frame inward localization fluctuation is
                        # allowed, but it may not reset the reference radius.
                        # Comparing against the path's running maximum prevents
                        # a sequence of individually small negative steps from
                        # walking inward through unrelated structures.
                        continue
                    transition_penalty = _fragment_transition_penalty(
                        previous_fragment,
                        fragment,
                        grid,
                        config,
                        frame_delta,
                    )
                    if transition_penalty is None:
                        continue
                    objective = (
                        previous_objective + fragment.objective - transition_penalty
                    )
                    length = int(best_length[previous_frame][previous_index]) + 1
                    running_max_radius_px = max(
                        previous_running_max_radius_px,
                        fragment.median_radius_px,
                    )
                    current_key = (
                        best_objective[frame_index][fragment_index],
                        int(best_length[frame_index][fragment_index]),
                        -best_running_max_radius_px[frame_index][fragment_index],
                    )
                    proposed_key = (
                        objective,
                        length,
                        -running_max_radius_px,
                    )
                    if proposed_key > current_key:
                        best_objective[frame_index][fragment_index] = objective
                        best_length[frame_index][fragment_index] = length
                        best_running_max_radius_px[frame_index][fragment_index] = (
                            running_max_radius_px
                        )
                        predecessor[frame_index][fragment_index] = (
                            previous_frame,
                            previous_index,
                        )

    possible_ends: list[tuple[float, int, int, int]] = []
    for frame_index, fragments in enumerate(fragments_by_frame):
        for fragment_index in range(len(fragments)):
            possible_ends.append(
                (
                    float(best_objective[frame_index][fragment_index]),
                    int(best_length[frame_index][fragment_index]),
                    frame_index,
                    fragment_index,
                )
            )
    minimum_path_length = max(
        config.minimum_track_points,
        config.minimum_event_frames,
    )
    completed_paths: list[tuple[tuple[float, float, int, float], _FragmentPath]] = []
    for additive_objective, length, frame_index, fragment_index in possible_ends:
        if length < minimum_path_length:
            continue
        reversed_fragments: list[_FrontFragment] = []
        while True:
            fragment = fragments_by_frame[frame_index][fragment_index]
            reversed_fragments.append(fragment)
            previous = predecessor[frame_index][fragment_index]
            if previous is None:
                break
            frame_index, fragment_index = previous
        fragments = tuple(reversed(reversed_fragments))
        displacement_px = _robust_fragment_displacement_px(fragments)
        if displacement_px < config.minimum_outward_displacement_px:
            continue

        node_rewards = np.asarray(
            [
                candidate.score
                - config.score_threshold
                + config.observation_reward
                for fragment in fragments
                for _, candidate in fragment.nodes
            ],
            dtype=np.float64,
        )
        # Final component comparison should gain only sqrt(N) significance
        # from more correlated samples. Winsorizing the upper tail keeps a few
        # extreme particle-contaminated nodes from deciding the event.
        upper_cap = float(np.quantile(node_rewards, 0.90))
        robust_rewards = np.minimum(node_rewards, upper_cap)
        coherent_significance = float(
            np.mean(robust_rewards) * math.sqrt(robust_rewards.size)
        )

        transition_penalty = 0.0
        for previous_fragment, current_fragment in zip(
            fragments,
            fragments[1:],
        ):
            frame_delta = int(
                frame_numbers[current_fragment.frame_index]
                - frame_numbers[previous_fragment.frame_index]
            )
            penalty = _fragment_transition_penalty(
                previous_fragment,
                current_fragment,
                grid,
                config,
                frame_delta,
            )
            if penalty is None:  # Defensive: every stored link was validated.
                transition_penalty = math.inf
                break
            transition_penalty += penalty
        component_objective = coherent_significance - transition_penalty
        if component_objective <= 0:
            continue

        path = _FragmentPath(
            fragments=fragments,
            objective=component_objective,
        )
        # Coherent evidence remains primary after the robust displacement gate.
        # This avoids automatically preferring a fast contaminant over a slower
        # CME while preventing a long-lived, effectively stationary structure
        # from passing because of one anomalous endpoint.
        key = (
            component_objective,
            displacement_px,
            length,
            additive_objective,
        )
        completed_paths.append((key, path))

    if not completed_paths:
        return None
    return max(completed_paths, key=lambda item: item[0])[1]


def _selected_fragments_have_angular_gaps(
    fragments: Sequence[_FrontFragment],
    angle_count: int,
) -> bool:
    """Whether any selected fragment bridges an unmeasured internal PA bin."""

    for fragment in fragments:
        indices = np.asarray(fragment.angle_indices, dtype=np.int64)
        if indices.size < 2:
            continue
        circular_gaps = np.diff(np.concatenate([indices, indices[:1] + angle_count]))
        # The largest gap is the exterior of a non-halo front. Remaining gaps
        # greater than one are internal missing PA samples.
        if circular_gaps.size > 1:
            exterior = int(np.argmax(circular_gaps))
            if np.any(np.delete(circular_gaps, exterior) > 1):
                return True
    return False


def _apply_angular_consistency(
    radius_px: np.ndarray,
    score: np.ndarray,
    sigma_px: np.ndarray,
    state: np.ndarray,
    grid: PolarGrid,
    config: TrackingConfig,
) -> int:
    """Reject isolated angular radius jumps without filling missing samples."""

    half_width_bins = max(
        1,
        int(
            math.ceil(
                config.angular_consistency_half_width_deg
                / grid.position_angle_step_deg
            )
        ),
    )
    observed = state == int(FrontState.OBSERVED)
    reject = np.zeros_like(observed)
    angle_count = radius_px.shape[1]
    for frame_index in range(radius_px.shape[0]):
        observed_angles = np.flatnonzero(observed[frame_index])
        for angle_index in observed_angles:
            neighbors = (
                angle_index
                + np.arange(-half_width_bins, half_width_bins + 1, dtype=np.int64)
            ) % angle_count
            neighbor_radii = radius_px[frame_index, neighbors]
            neighbor_radii = neighbor_radii[np.isfinite(neighbor_radii)]
            if neighbor_radii.size < config.minimum_angular_neighbors:
                continue
            local_median = float(np.median(neighbor_radii))
            if (
                abs(radius_px[frame_index, angle_index] - local_median)
                > config.angular_outlier_tolerance_px
            ):
                reject[frame_index, angle_index] = True

    radius_px[reject] = np.nan
    score[reject] = np.nan
    sigma_px[reject] = np.nan
    state[reject] = int(FrontState.REJECTED)
    return int(np.count_nonzero(reject))


def _automatic_sector_window_deg(
    path: _FragmentPath,
    grid: PolarGrid,
    padding_deg: float,
) -> tuple[float, float] | None:
    """Infer the smallest padded circular PA window occupied by a path.

    ``None`` represents a halo-like component whose padded support covers the
    full circle; in that case an angular-sector refinement would add no useful
    constraint.
    """

    angle_count = grid.position_angle_deg.size
    indices = np.unique(
        np.asarray(
            [
                angle_index
                for fragment in path.fragments
                for angle_index in fragment.angle_indices
            ],
            dtype=np.int64,
        )
    )
    if indices.size == 0 or indices.size == angle_count:
        return None

    circular_steps = np.diff(
        np.concatenate([indices, indices[:1] + angle_count])
    )
    largest_gap_after = int(np.argmax(circular_steps))
    start_index = int(indices[(largest_gap_after + 1) % indices.size])
    end_index = int(indices[largest_gap_after])
    occupied_span_bins = angle_count - int(circular_steps[largest_gap_after]) + 1
    padding_bins = int(math.ceil(padding_deg / grid.position_angle_step_deg))
    if occupied_span_bins + 2 * padding_bins >= angle_count:
        return None

    start_index = (start_index - padding_bins) % angle_count
    end_index = (end_index + padding_bins) % angle_count
    return (
        float(grid.position_angle_deg[start_index] % 360.0),
        float(grid.position_angle_deg[end_index] % 360.0),
    )


def _extract_front_coherent_fragments(
    likelihood: LikelihoodResult | np.ndarray,
    grid: PolarGrid,
    config: TrackingConfig,
    *,
    frame_numbers: Sequence[int] | np.ndarray | None = None,
    refine_selected_sector: bool = False,
) -> FrontTrack:
    """Associate simultaneous angular ridges into one outward event path."""

    score_cube = (
        likelihood.score if isinstance(likelihood, LikelihoodResult) else likelihood
    )
    score_cube = np.asarray(score_cube, dtype=np.float64)
    if score_cube.ndim != 3 or score_cube.shape[1:] != grid.shape:
        raise TrackingError(
            "Likelihood must have shape (time, position_angle, radius) "
            "matching the supplied grid."
        )
    if score_cube.shape[0] < config.minimum_track_points:
        raise TrackingError(
            "Likelihood has fewer frames than minimum_track_points."
        )

    frame_count, angle_count, _ = score_cube.shape
    nominal_frame_numbers = _validated_frame_numbers(frame_numbers, frame_count)
    angle_window_mask = _position_angle_window_mask(
        grid.position_angle_deg,
        config.position_angle_window_deg,
    )
    fragments_by_frame = tuple(
        _coherent_fragments_in_frame(
            score_cube[frame_index],
            frame_index,
            grid,
            config,
            angle_window_mask,
        )
        for frame_index in range(frame_count)
    )
    selected_path = _link_coherent_fragments(
        fragments_by_frame,
        grid,
        config,
        nominal_frame_numbers,
    )

    if refine_selected_sector and selected_path is not None:
        automatic_window = _automatic_sector_window_deg(
            selected_path,
            grid,
            config.automatic_sector_padding_deg,
        )
        if automatic_window is not None:
            refinement_config = replace(
                config,
                association_method="independent_angle_paths",
                position_angle_window_deg=automatic_window,
                maximum_outward_step_px_per_frame=(
                    config.refinement_maximum_outward_step_px_per_frame
                ),
                maximum_gap_frames=config.refinement_maximum_gap_frames,
                minimum_track_points=config.refinement_minimum_track_points,
                minimum_outward_displacement_px=(
                    config.refinement_minimum_outward_displacement_px
                ),
                minimum_event_frames=config.refinement_minimum_event_frames,
                minimum_observed_angles_per_frame=(
                    config.refinement_minimum_observed_angles_per_frame
                ),
            )
            refined = _extract_front_independent_angle_paths(
                likelihood,
                grid,
                refinement_config,
                frame_numbers=nominal_frame_numbers,
            )
            if refined.event_detected:
                refined_flags = (
                    "COHERENT_FRAGMENT_ASSOCIATION",
                    "AUTOMATIC_POSITION_ANGLE_SECTOR",
                    *refined.quality_flags,
                )
                return FrontTrack(
                    radius_px=refined.radius_px,
                    radius_rsun=refined.radius_rsun,
                    radial_sigma_px=refined.radial_sigma_px,
                    score=refined.score,
                    state=refined.state,
                    position_angle_deg=refined.position_angle_deg,
                    event_detected=True,
                    quality_flags=refined_flags,
                )

    radius_px = np.full((frame_count, angle_count), np.nan, dtype=np.float64)
    radial_sigma_px = np.full_like(radius_px, np.nan)
    selected_score = np.full_like(radius_px, np.nan)
    state = np.full(
        (frame_count, angle_count),
        int(FrontState.MISSING),
        dtype=np.uint8,
    )
    flags: list[str] = ["COHERENT_FRAGMENT_ASSOCIATION"]
    if config.position_angle_window_deg is not None:
        flags.append("POSITION_ANGLE_WINDOW_APPLIED")

    if selected_path is not None:
        for fragment in selected_path.fragments:
            for angle_index, candidate in fragment.nodes:
                radius_px[fragment.frame_index, angle_index] = grid.radius_px[
                    candidate.radius_index
                ]
                radial_sigma_px[fragment.frame_index, angle_index] = candidate.sigma_px
                selected_score[fragment.frame_index, angle_index] = candidate.score
                state[fragment.frame_index, angle_index] = int(FrontState.OBSERVED)

    rejected_count = _apply_angular_consistency(
        radius_px,
        selected_score,
        radial_sigma_px,
        state,
        grid,
        config,
    )
    if rejected_count:
        flags.append("ANGULAR_OUTLIERS_REJECTED")

    observed_count_by_frame = np.count_nonzero(
        state == int(FrontState.OBSERVED),
        axis=1,
    )
    active_frames = observed_count_by_frame >= config.minimum_observed_angles_per_frame
    event_detected = selected_path is not None and (
        np.count_nonzero(active_frames) >= config.minimum_event_frames
    )
    if selected_path is None:
        flags.extend(
            [
                "INSUFFICIENT_SPATIOTEMPORAL_COHERENCE",
                "INSUFFICIENT_ANGULAR_COVERAGE",
            ]
        )
    elif not event_detected:
        flags.append("INSUFFICIENT_TEMPORAL_COVERAGE")
    if not event_detected:
        flags.append("NO_EVENT")

    if selected_path is not None and _selected_fragments_have_angular_gaps(
        selected_path.fragments,
        angle_count,
    ):
        flags.append("ANGULAR_GAPS_PRESENT")

    nonzero_radius = grid.radius_rsun > 0
    if not np.any(nonzero_radius):
        raise TrackingError("The polar grid cannot infer pixels per solar radius.")
    pixels_per_solar_radius = float(
        np.median(grid.radius_px[nonzero_radius] / grid.radius_rsun[nonzero_radius])
    )
    radius_rsun = radius_px / pixels_per_solar_radius
    return FrontTrack(
        radius_px=radius_px,
        radius_rsun=radius_rsun,
        radial_sigma_px=radial_sigma_px,
        score=selected_score,
        state=state,
        position_angle_deg=grid.position_angle_deg.copy(),
        event_detected=bool(event_detected),
        quality_flags=tuple(flags),
    )


def _extract_front_independent_angle_paths(
    likelihood: LikelihoodResult | np.ndarray,
    grid: PolarGrid,
    config: TrackingConfig | None = None,
    *,
    frame_numbers: Sequence[int] | np.ndarray | None = None,
) -> FrontTrack:
    """Retain the original independent-position-angle baseline tracker.

    Missing temporal frames and angular samples remain ``NaN`` in the returned
    front.  This baseline does not interpolate them for display or kinematics.
    ``frame_numbers`` identifies omitted nominal frames: its positive integer
    deltas scale motion limits and gap penalties. It defaults to contiguous
    row numbers for callers without an explicit acquisition coordinate.
    """

    if config is None:
        config = TrackingConfig()
    score_cube = (
        likelihood.score if isinstance(likelihood, LikelihoodResult) else likelihood
    )
    score_cube = np.asarray(score_cube, dtype=np.float64)
    if score_cube.ndim != 3 or score_cube.shape[1:] != grid.shape:
        raise TrackingError(
            "Likelihood must have shape (time, position_angle, radius) "
            "matching the supplied grid."
        )
    if score_cube.shape[0] < config.minimum_track_points:
        raise TrackingError(
            "Likelihood has fewer frames than minimum_track_points."
        )

    frame_count, angle_count, _ = score_cube.shape
    nominal_frame_numbers = _validated_frame_numbers(frame_numbers, frame_count)
    angle_window_mask = _position_angle_window_mask(
        grid.position_angle_deg,
        config.position_angle_window_deg,
    )
    paths: list[_Path | None] = []
    for angle_index in range(angle_count):
        if not angle_window_mask[angle_index]:
            paths.append(None)
            continue
        paths.append(
            _track_one_position_angle(
                score_cube[:, angle_index, :],
                grid,
                config,
                nominal_frame_numbers,
            )
        )

    accepted_angles = np.asarray([path is not None for path in paths], dtype=bool)
    grouped_angles = _close_short_circular_gaps(
        accepted_angles,
        config.maximum_angular_gap_bins,
    )
    components = _circular_components(grouped_angles)
    minimum_component_bins = max(
        1,
        int(math.ceil(config.minimum_angular_support_deg / grid.position_angle_step_deg)),
    )
    components = tuple(
        component
        for component in components
        if component.size >= minimum_component_bins
    )

    radius_px = np.full((frame_count, angle_count), np.nan, dtype=np.float64)
    radial_sigma_px = np.full_like(radius_px, np.nan)
    selected_score = np.full_like(radius_px, np.nan)
    state = np.full(
        (frame_count, angle_count),
        int(FrontState.MISSING),
        dtype=np.uint8,
    )
    flags: list[str] = []
    if config.position_angle_window_deg is not None:
        flags.append("POSITION_ANGLE_WINDOW_APPLIED")

    selected_component: np.ndarray | None = None
    if components:
        selected_component = max(
            components,
            key=lambda component: sum(
                paths[index].objective
                for index in component
                if paths[index] is not None
            ),
        )
        selected_angle_mask = np.zeros(angle_count, dtype=bool)
        selected_angle_mask[selected_component] = True
    else:
        selected_angle_mask = np.zeros(angle_count, dtype=bool)

    for angle_index, path in enumerate(paths):
        if path is None:
            continue
        selected = selected_angle_mask[angle_index]
        for frame_index, candidate in path.nodes:
            if selected:
                radius_px[frame_index, angle_index] = grid.radius_px[
                    candidate.radius_index
                ]
                radial_sigma_px[frame_index, angle_index] = candidate.sigma_px
                selected_score[frame_index, angle_index] = candidate.score
                state[frame_index, angle_index] = int(FrontState.OBSERVED)
            else:
                state[frame_index, angle_index] = int(FrontState.REJECTED)

    rejected_count = _apply_angular_consistency(
        radius_px,
        selected_score,
        radial_sigma_px,
        state,
        grid,
        config,
    )
    if rejected_count:
        flags.append("ANGULAR_OUTLIERS_REJECTED")

    observed_count_by_frame = np.count_nonzero(
        state == int(FrontState.OBSERVED),
        axis=1,
    )
    active_frames = observed_count_by_frame >= config.minimum_observed_angles_per_frame
    event_detected = selected_component is not None and (
        np.count_nonzero(active_frames) >= config.minimum_event_frames
    )
    if selected_component is None:
        flags.append("INSUFFICIENT_ANGULAR_COVERAGE")
    elif not event_detected:
        flags.append("INSUFFICIENT_TEMPORAL_COVERAGE")
    if not event_detected:
        flags.append("NO_EVENT")

    if selected_component is not None and np.any(
        ~accepted_angles[selected_component]
    ):
        flags.append("ANGULAR_GAPS_PRESENT")

    nonzero_radius = grid.radius_rsun > 0
    if not np.any(nonzero_radius):
        raise TrackingError("The polar grid cannot infer pixels per solar radius.")
    pixels_per_solar_radius = float(
        np.median(grid.radius_px[nonzero_radius] / grid.radius_rsun[nonzero_radius])
    )
    radius_rsun = radius_px / pixels_per_solar_radius
    return FrontTrack(
        radius_px=radius_px,
        radius_rsun=radius_rsun,
        radial_sigma_px=radial_sigma_px,
        score=selected_score,
        state=state,
        position_angle_deg=grid.position_angle_deg.copy(),
        event_detected=bool(event_detected),
        quality_flags=tuple(flags),
    )


def extract_front(
    likelihood: LikelihoodResult | np.ndarray,
    grid: PolarGrid,
    config: TrackingConfig | None = None,
    *,
    frame_numbers: Sequence[int] | np.ndarray | None = None,
) -> FrontTrack:
    """Extract one outward front without requiring an event-specific PA prior.

    The default method constructs simultaneous radially continuous fragments
    and links those physical candidates through time. The two-stage method can
    then infer and pad their circular angular sector before invoking the
    independent-position-angle sparse paths for detailed recovery. The
    historical independent implementation remains directly selectable.
    """

    if config is None:
        config = TrackingConfig()
    if config.association_method == "independent_angle_paths":
        return _extract_front_independent_angle_paths(
            likelihood,
            grid,
            config,
            frame_numbers=frame_numbers,
        )
    return _extract_front_coherent_fragments(
        likelihood,
        grid,
        config,
        frame_numbers=frame_numbers,
        refine_selected_sector=(
            config.association_method == "coherent_sector_refined_paths"
        ),
    )
