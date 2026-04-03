# %% [markdown]
# # Parser + Mathematical Layer Pipeline
#
# This notebook exercises the current end-to-end pipeline available in the
# thesis package:
# - natural-language scenario -> parser -> structured scenario
# - structured scenario -> mathematical layer -> risk matrix
# - runtime retrieval -> prompt-ready context documents
# - optional reasoning LLM -> final ethical analysis
#
# Offline ingestion is intentionally out of scope here. This notebook only shows
# runtime use of an already-indexed knowledge base.

# %%
from pathlib import Path
import json
import sys

root = Path.cwd()
if not (root / "thesis").exists():
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from thesis import DeterministicRAGRetriever, EthicalReasoningLLM, ScenarioPipeline

rag_retriever = DeterministicRAGRetriever()
reasoning_llm = EthicalReasoningLLM()
pipeline = ScenarioPipeline(
    rag_retriever=rag_retriever,
    reasoning_llm=reasoning_llm,
)


def print_json(payload):
    print(json.dumps(payload, indent=2))


def summarize_rag_result(rag_result):
    if rag_result is None:
        return {
            "runtime_status": "unavailable",
            "reason": (
                "RAG runtime dependencies, a readable OPENAI_API_KEY from the process or .env, "
                "or a persisted Chroma collection are not available."
            ),
        }

    return {
        "runtime_available": rag_result.runtime_available,
        "runtime_error": rag_result.runtime_error,
        "query": rag_result.query,
        "indexed_chunks": rag_result.indexed_chunks,
        "always_included_documents": [
            {
                "title": document.title,
                "category": document.category,
                "path": document.path,
                "content_length": len(document.content),
            }
            for document in rag_result.always_included_documents
        ],
        "retrieved_documents": [
            {
                "title": document.title,
                "category": document.category,
                "path": document.path,
                "score": document.score,
            }
            for document in rag_result.retrieved_documents
        ],
    }


def summarize_reasoning_result(reasoning_result):
    if reasoning_result is None:
        return {
            "runtime_status": "skipped",
            "reason": "No reasoning LLM was attached to the pipeline.",
        }

    if not reasoning_result.runtime_available:
        return {
            "runtime_status": "unavailable",
            "reason": reasoning_result.runtime_error,
            "model_name": reasoning_result.model_name,
        }

    return {
        "model_name": reasoning_result.model_name,
        "dominant_framework": reasoning_result.dominant_framework,
        "contributing_frameworks": reasoning_result.contributing_frameworks,
        "weights": reasoning_result.weights,
        "violated_constraints": reasoning_result.violated_constraints,
        "confidence": reasoning_result.confidence,
        "rationale": reasoning_result.rationale,
    }


print(f"Project root: {root}")

# %% [markdown]
# ## 1. Natural-language input through the current pipeline
#
# The parser produces the structured JSON schema first. The mathematical layer
# then works only on that normalized scenario. If a runtime RAG index is
# available, the same pipeline call also returns retrieved context chunks.

# %%
nl_scenario = (
    "An autonomous vehicle weighing 1800 kg is traveling at 60 km/h in the center lane on a "
    "residential road with a 50 km/h speed limit. It is braking at 2.1 m/s2 and has a braking "
    "distance of 42.5 m. Weather is clear, visibility is 120 m, it is daytime, and traffic is low. "
    "A child pedestrian is crossing 10.2 m ahead with time to impact 0.61 s. "
    "A parked vehicle is 6.5 m ahead with time to impact 0.39 s. "
    "Lidar confidence is 0.97, camera 0.91, radar 0.95, and overall scene confidence is 0.93. "
    "The left sidewalk is occluded. Available actions are brake straight, swerve left, swerve right, "
    "and brake and swerve left. Collision is unavoidable."
)

nl_result = pipeline.run(nl_scenario)

print("Parser output:")
print_json(nl_result.parser_result.to_dict())

print("\nMathematical layer summary:")
print_json(
    {
        "best_action_by_total_risk": nl_result.mathematical_layer_result.best_action_by_total_risk,
        "violated_rules": nl_result.mathematical_layer_result.violated_rules,
        "global_metrics": nl_result.mathematical_layer_result.global_metrics,
    }
)

print("\nRAG runtime summary:")
print_json(summarize_rag_result(nl_result.rag_retrieval_result))

print("\nReasoning LLM summary:")
print_json(summarize_reasoning_result(nl_result.reasoning_result))

# %% [markdown]
# ## 2. Direct structured JSON through the same pipeline
#
# This mirrors the primary evaluation mode for the thesis benchmark and should
# be the preferred path for controlled experiments.

# %%
structured_payload = {
    "ego_vehicle": {
        "speed_kmh": 40,
        "acceleration_ms2": -4.0,
        "heading_deg": 0,
        "lane_position": "center",
        "braking_distance_m": 18.5,
        "mass_kg": 1800,
    },
    "environment": {
        "road_type": "school_zone",
        "speed_limit_kmh": 30,
        "weather": "rain",
        "visibility_m": 60,
        "time_of_day": "daytime",
        "traffic_density": "medium",
    },
    "obstacles": [
        {
            "id": "obj_01",
            "type": "child_pedestrian",
            "distance_m": 9,
            "relative_speed_kmh": 40,
            "time_to_impact_s": 0.81,
            "trajectory": "crossing",
            "vulnerability_class": "high",
            "mass_kg": 30,
            "responsible_for_risk": False,
        }
    ],
    "sensor_confidence": {
        "lidar": 0.90,
        "camera": 0.82,
        "radar": 0.88,
        "overall_scene_confidence": 0.87,
        "occluded_zones": ["crosswalk", "right_lane"],
    },
    "available_actions": ["brake_straight", "swerve_right", "brake_swerve_right"],
    "collision_unavoidable": True,
}

structured_result = pipeline.run(structured_payload)
print("Structured parser output:")
print_json(structured_result.parser_result.to_dict())

print("\nStructured mathematical layer summary:")
print_json(
    {
        "best_action_by_total_risk": structured_result.mathematical_layer_result.best_action_by_total_risk,
        "violated_rules": structured_result.mathematical_layer_result.violated_rules,
        "global_metrics": structured_result.mathematical_layer_result.global_metrics,
    }
)

print("\nStructured RAG runtime summary:")
print_json(summarize_rag_result(structured_result.rag_retrieval_result))

print("\nStructured reasoning LLM summary:")
print_json(summarize_reasoning_result(structured_result.reasoning_result))

# %% [markdown]
# ## 3. Compact action comparison
#
# This gives a compact view of what the next reasoning stage would receive from
# the deterministic layers before it reads the retrieved context.

# %%
action_comparison = [
    {
        "action": assessment.action,
        "total_risk": assessment.total_risk,
        "constraint_flags": assessment.constraint_flags,
    }
    for assessment in structured_result.mathematical_layer_result.action_assessments
]
action_comparison.sort(key=lambda item: item["total_risk"])
print_json(action_comparison)

# %% [markdown]
# ## 4. Prompt-context view
#
# The runtime retriever now returns:
# - `always_included_documents` for the full ethical-framework files
# - `retrieved_documents` for ranked similarity hits with short excerpts

# %%
print_json(
    {
        "always_included_documents": summarize_rag_result(structured_result.rag_retrieval_result).get(
            "always_included_documents",
            [],
        ),
        "retrieved_documents": summarize_rag_result(structured_result.rag_retrieval_result).get(
            "retrieved_documents",
            [],
        ),
    }
)
