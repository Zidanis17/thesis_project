from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..core.models import ParserResult, Scenario
from ..core.normalization import (
    VRU_TYPES as SHARED_VRU_TYPES,
    VRU_VULNERABILITY_CLASSES as SHARED_VRU_VULNERABILITY_CLASSES,
)
from .models import AgenticAssessment, AgenticValidationResult, RetrievalIntent

if TYPE_CHECKING:
    from ..mathematical_layer import MathematicalLayerResult
    from ..reasoning_llm import EthicalReasoningResult


class AgenticEthicalController:
    """
    Deterministic orchestration and validation layer.

    This controller does not call the reasoning model, the RAG retriever, or
    Chroma. It only inspects structured inputs and emits bounded routing data.
    """

    ALLOWED_DOMINANT_FRAMEWORKS = {"EF-01", "EF-02", "EF-03", "EF-05", "EF-06"}
    VRU_TYPES = SHARED_VRU_TYPES
    VRU_VULNERABILITY_CLASSES = SHARED_VRU_VULNERABILITY_CLASSES
    PASSENGER_RISK_KEYS = ("ego:passenger", "passenger", "occupant")

    def assess(
        self,
        parser_result: ParserResult,
        mathematical_layer_result: MathematicalLayerResult | None = None,
    ) -> AgenticAssessment:
        scenario = parser_result.scenario
        scene_interpretable = self._scene_interpretable_for_scenario(
            scenario,
            mathematical_layer_result,
        )
        unknown_object_present = self._has_unknown_object(scenario)
        vru_present = self._has_vru(scenario)
        tradeoff_possible = self._has_passenger_vru_tradeoff_signal(
            parser_result,
            mathematical_layer_result,
        )
        scenario_class, candidate_frameworks = self._classify(
            scenario=scenario,
            scene_interpretable=scene_interpretable,
            unknown_object_present=unknown_object_present,
            vru_present=vru_present,
            tradeoff_possible=tradeoff_possible,
        )

        notes: list[str] = [
            f"scenario_class={scenario_class}",
            f"scene_interpretable={scene_interpretable}",
            f"collision_unavoidable={scenario.collision_unavoidable}",
        ]
        if unknown_object_present:
            notes.append("unknown_object_present=True")
        if vru_present:
            notes.append("vulnerable_road_users_present=True")
        if tradeoff_possible:
            notes.append("passenger_vru_tradeoff_possible=True")

        return AgenticAssessment(
            scene_interpretable=scene_interpretable,
            collision_unavoidable=scenario.collision_unavoidable,
            vulnerable_road_users_present=vru_present,
            passenger_vru_tradeoff_possible=tradeoff_possible,
            unknown_object_present=unknown_object_present,
            candidate_frameworks=candidate_frameworks,
            retrieval_intent=self.build_retrieval_intent(
                scenario,
                mathematical_layer_result,
            ),
            notes=notes,
        )

    def build_retrieval_intent(
        self,
        scenario: Scenario,
        mathematical_layer_result: MathematicalLayerResult | None = None,
    ) -> RetrievalIntent:
        scenario_class, _candidate_frameworks = self._classify(
            scenario=scenario,
            scene_interpretable=self._scene_interpretable_for_scenario(
                scenario,
                mathematical_layer_result,
            ),
            unknown_object_present=self._has_unknown_object(scenario),
            vru_present=self._has_vru(scenario),
            tradeoff_possible=self._scenario_has_passenger_vru_tradeoff_signal(
                scenario,
                mathematical_layer_result,
            ),
        )
        return self._intent_for_class(scenario_class)

    def validate_reasoning_result(
        self,
        parser_result: ParserResult,
        mathematical_layer_result: MathematicalLayerResult | None,
        reasoning_result: EthicalReasoningResult,
    ) -> AgenticValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        scenario = parser_result.scenario
        dominant_framework = reasoning_result.dominant_framework

        if dominant_framework is None:
            if getattr(reasoning_result, "runtime_available", False):
                errors.append("dominant_framework is missing.")
            else:
                warnings.append(
                    "dominant_framework is unavailable because reasoning runtime did not complete."
                )
        elif dominant_framework == "EF-04":
            errors.append("EF-04 must not be used as dominant_framework.")
        elif dominant_framework not in self.ALLOWED_DOMINANT_FRAMEWORKS:
            errors.append(
                "dominant_framework must be one of EF-01, EF-02, EF-03, EF-05, or EF-06."
            )

        if scenario.collision_unavoidable is True and dominant_framework == "EF-02":
            errors.append("EF-02 must not dominate when collision_unavoidable is true.")

        if not self._scene_interpretable_for_scenario(scenario, mathematical_layer_result):
            if dominant_framework is not None and dominant_framework != "EF-06":
                warnings.append("EF-06 should dominate when scene_interpretable is false.")

        if self._has_unknown_object(scenario):
            if dominant_framework is not None and dominant_framework != "EF-06":
                warnings.append("EF-06 should dominate when an unknown object is present.")

        if dominant_framework == "EF-05" and not self._has_passenger_vru_tradeoff_signal(
            parser_result,
            mathematical_layer_result,
        ):
            errors.append(
                "EF-05 should only dominate with passenger/occupant versus VRU tradeoff evidence."
            )

        if mathematical_layer_result is not None:
            math_actions = set(mathematical_layer_result.risk_score_matrix)
            reasoning_actions = set(reasoning_result.risk_scores_per_action)
            invented_actions = sorted(reasoning_actions - math_actions)
            if invented_actions:
                errors.append(
                    "risk_scores_per_action contains actions not present in the mathematical layer: "
                    + ", ".join(invented_actions)
                )

        return AgenticValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=warnings,
        )

    def _has_vru(self, scenario: Scenario) -> bool:
        return any(
            obstacle.vulnerability_class.strip().lower() in self.VRU_VULNERABILITY_CLASSES
            or obstacle.type.strip().lower() in self.VRU_TYPES
            for obstacle in scenario.obstacles
        )

    def _has_unknown_object(self, scenario: Scenario) -> bool:
        return any("unknown" in obstacle.type.strip().lower() for obstacle in scenario.obstacles)

    def _scene_interpretable(
        self,
        mathematical_layer_result: MathematicalLayerResult | None,
    ) -> bool:
        if mathematical_layer_result is None:
            return True
        return bool(mathematical_layer_result.global_metrics.get("scene_interpretable", True))

    def _has_passenger_vru_tradeoff_signal(
        self,
        parser_result: ParserResult,
        mathematical_layer_result: MathematicalLayerResult | None,
    ) -> bool:
        return self._scenario_has_passenger_vru_tradeoff_signal(
            parser_result.scenario,
            mathematical_layer_result,
        )

    def _classify(
        self,
        *,
        scenario: Scenario,
        scene_interpretable: bool,
        unknown_object_present: bool,
        vru_present: bool,
        tradeoff_possible: bool,
    ) -> tuple[str, list[str]]:
        if not scene_interpretable or unknown_object_present:
            return "epistemic_uncertainty_case", ["EF-06"]

        if tradeoff_possible:
            return "passenger_vru_tradeoff_case", ["EF-05", "EF-03", "EF-01"]

        if scenario.collision_unavoidable is None:
            return "epistemic_uncertainty_case", ["EF-06", "EF-02"]

        if scenario.collision_unavoidable is False:
            return "avoidable_rule_based_case", ["EF-02"]

        if scenario.collision_unavoidable is True and vru_present:
            return "unavoidable_vru_dilemma", ["EF-03", "EF-01"]

        if scenario.collision_unavoidable is True and not vru_present:
            return "unavoidable_aggregate_risk_case", ["EF-01"]

        return "epistemic_uncertainty_case", ["EF-06"]

    def _intent_for_class(self, scenario_class: str) -> RetrievalIntent:
        if scenario_class == "avoidable_rule_based_case":
            return RetrievalIntent(
                scenario_class=scenario_class,
                required_frameworks=["EF-02"],
                retrieval_focus_terms=[
                    "RSS safety rules",
                    "deontological safety",
                    "responsibility sensitive safety",
                    "rule violation",
                ],
                reason="Interpretable avoidable scenario; rule-based safety context should guide retrieval.",
            )

        if scenario_class == "unavoidable_vru_dilemma":
            return RetrievalIntent(
                scenario_class=scenario_class,
                required_frameworks=["EF-03", "EF-04", "EF-01"],
                retrieval_focus_terms=[
                    "vulnerable road user",
                    "worst-off protection",
                    "risk distribution",
                    "aggregate harm",
                    "maximin",
                ],
                reason="Unavoidable collision with vulnerable road users; retrieve maximin and risk-distribution context.",
            )

        if scenario_class == "unavoidable_aggregate_risk_case":
            return RetrievalIntent(
                scenario_class=scenario_class,
                required_frameworks=["EF-01", "EF-04"],
                retrieval_focus_terms=[
                    "aggregate harm",
                    "total risk minimization",
                    "expected harm",
                    "utilitarian risk",
                ],
                reason="Unavoidable collision without VRU evidence; aggregate risk context is most relevant.",
            )

        if scenario_class == "passenger_vru_tradeoff_case":
            return RetrievalIntent(
                scenario_class=scenario_class,
                required_frameworks=["EF-05", "EF-03", "EF-04"],
                retrieval_focus_terms=[
                    "ethical valence theory",
                    "passenger pedestrian tradeoff",
                    "social valence",
                    "vulnerable road user",
                ],
                reason="Passenger or occupant protection appears to trade off against VRU protection.",
            )

        return RetrievalIntent(
            scenario_class="epistemic_uncertainty_case",
            required_frameworks=["EF-06", "EF-02"],
            retrieval_focus_terms=[
                "uncertainty",
                "reasonable skilled driver",
                "fallback reasoning",
                "scene interpretability",
                "safe behavior under uncertainty",
            ],
            reason="Scene interpretation is uncertain or includes unknown objects; retrieve fallback and safety context.",
        )

    def _scene_interpretable_for_scenario(
        self,
        scenario: Scenario,
        mathematical_layer_result: MathematicalLayerResult | None,
    ) -> bool:
        if mathematical_layer_result is not None:
            return self._scene_interpretable(mathematical_layer_result)
        confidence_values = (
            scenario.sensor_confidence.overall_scene_confidence,
            scenario.sensor_confidence.lidar,
            scenario.sensor_confidence.camera,
            scenario.sensor_confidence.radar,
        )
        if any(value is None for value in confidence_values):
            return False
        confidence = (
            0.40 * scenario.sensor_confidence.overall_scene_confidence
            + 0.20 * scenario.sensor_confidence.lidar
            + 0.20 * scenario.sensor_confidence.camera
            + 0.20 * scenario.sensor_confidence.radar
        )
        return confidence >= 0.85

    def _scenario_has_passenger_vru_tradeoff_signal(
        self,
        scenario: Scenario,
        mathematical_layer_result: MathematicalLayerResult | None,
    ) -> bool:
        if scenario.collision_unavoidable is not True or not self._has_vru(scenario):
            return False

        if scenario.ego_vehicle.passenger_at_risk is True:
            return True

        if mathematical_layer_result is None:
            return False

        risk_score_matrix = mathematical_layer_result.risk_score_matrix
        if not risk_score_matrix:
            return False

        passenger_scores_by_action: dict[str, float] = {}
        vru_scores_by_action: dict[str, float] = {}
        vru_ids = set(self._vru_stakeholder_ids(scenario))
        if not vru_ids:
            return False

        for action, scores in risk_score_matrix.items():
            passenger_key = self._passenger_risk_key(scores)
            if passenger_key is None:
                return False
            passenger_scores_by_action[action] = float(scores.get(passenger_key, 0.0))
            vru_scores_by_action[action] = sum(
                float(scores.get(stakeholder_id, 0.0)) for stakeholder_id in vru_ids
            )

        return self._best_action(passenger_scores_by_action) != self._best_action(vru_scores_by_action)

    def _passenger_risk_key(self, scores: dict[str, Any]) -> str | None:
        for key in self.PASSENGER_RISK_KEYS:
            if key in scores:
                return key
        return None

    def _vru_stakeholder_ids(self, scenario: Scenario) -> list[str]:
        stakeholder_ids: list[str] = []
        for obstacle in scenario.obstacles:
            type_name = obstacle.type.lower()
            vulnerability = obstacle.vulnerability_class.lower()
            if (
                obstacle.type.strip().lower() in self.VRU_TYPES
                or obstacle.vulnerability_class.strip().lower() in self.VRU_VULNERABILITY_CLASSES
                or any(
                    token in type_name or token in vulnerability
                    for token in (
                        "pedestrian",
                        "child",
                        "elderly",
                        "cyclist",
                        "motorcyclist",
                    )
                )
            ):
                stakeholder_ids.append(obstacle.id)
        return stakeholder_ids

    @staticmethod
    def _best_action(scores_by_action: dict[str, float]) -> str | None:
        if not scores_by_action:
            return None
        return min(scores_by_action.items(), key=lambda item: (item[1], item[0]))[0]
