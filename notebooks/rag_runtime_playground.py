# %% [markdown]
# # RAG Runtime Playground
#
# This notebook isolates runtime retrieval over the thesis knowledge base.
#
# It does not run ingestion. The notebook assumes:
# - a persisted Chroma collection already exists under `knowledge_base/.chroma`
# - runtime embedding dependencies are installed
# - `OPENAI_API_KEY` is present in the process environment or the project `.env`
#
# The goal is to inspect the exact retrieval artifact that should be passed into
# the later LLM prompt rather than rebuilding the index here.

# %%
from pathlib import Path
import json
import sys

root = Path.cwd()
if not (root / "thesis").exists():
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from thesis import (
    DeterministicMathematicalLayer,
    DeterministicRAGRetriever,
    DeterministicScenarioParser,
    ScenarioPipeline,
)

parser = DeterministicScenarioParser()
math_layer = DeterministicMathematicalLayer()
retriever = DeterministicRAGRetriever()
pipeline = ScenarioPipeline(rag_retriever=retriever)


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
                "excerpt": document.excerpt,
            }
            for document in rag_result.retrieved_documents
        ],
    }


print(f"Project root: {root}")

# %% [markdown]
# ## 1. Runtime setup
#
# This shows the runtime retriever configuration only. If retrieval later comes
# back as unavailable, the issue is with runtime dependencies or the persisted
# index, not with ingestion logic inside this notebook.

# %%
print_json(
    {
        "knowledge_base_path": str(retriever.knowledge_base_path),
        "persist_directory": str(retriever.persist_directory),
        "top_k": retriever.top_k,
        "framework_top_k": retriever.framework_top_k,
    }
)

# %% [markdown]
# ## 2. Structured scenario -> mathematical layer -> runtime retrieval
#
# The retriever query is stronger when it includes deterministic mathematical
# layer output, especially violated rules and the current best action by risk.

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

scenario = parser.parse(structured_payload).scenario
analysis = math_layer.analyze(scenario)

try:
    retrieval_result = retriever.retrieve(scenario, analysis)
except RuntimeError as exc:
    retrieval_result = None
    print(f"RAG runtime unavailable: {exc}")

print("Normalized scenario:")
print_json(scenario.to_dict())

print("\nMathematical layer summary:")
print_json(
    {
        "best_action_by_total_risk": analysis.best_action_by_total_risk,
        "violated_rules": analysis.violated_rules,
        "global_metrics": analysis.global_metrics,
    }
)

print("\nRuntime retrieval summary:")
print_json(summarize_rag_result(retrieval_result))

# %% [markdown]
# ## 3. Retrieved prompt context
#
# This is the prompt-ready runtime artifact:
# - `always_included_documents` contains the full ethical-framework files
# - `retrieved_documents` contains ranked similarity hits with excerpts

# %%
print_json(
    {
        "always_included_documents": summarize_rag_result(retrieval_result).get(
            "always_included_documents",
            [],
        ),
        "retrieved_documents": summarize_rag_result(retrieval_result).get("retrieved_documents", []),
    }
)

# %% [markdown]
# ## 4. Full ethical-framework context
#
# These files are always included in full so the later prompt builder has the
# complete ethical framework reference set available on every run.

# %%
print_json(
    [
        {
            "title": document.title,
            "path": document.path,
            "content": document.content,
        }
        for document in (retrieval_result.always_included_documents if retrieval_result else [])
    ]
)

# %% [markdown]
# ## 5. Same runtime stage through the full pipeline
#
# This confirms the pipeline can carry parser output, mathematical analysis, and
# retrieval results together in one runtime call without invoking ingestion.

# %%
nl_scenario = (
    "An autonomous vehicle is traveling at 35 km/h in a school zone during light rain. "
    "Visibility is 60 m and traffic density is medium. A child pedestrian is crossing 7 m ahead. "
    "The right lane and crosswalk are partially occluded. Available actions are brake straight, "
    "swerve right, and brake and swerve right. Collision is unavoidable."
)

pipeline_result = pipeline.run(nl_scenario)

print_json(
    {
        "parser_input_mode": pipeline_result.parser_result.input_mode,
        "best_action_by_total_risk": pipeline_result.mathematical_layer_result.best_action_by_total_risk,
        "violated_rules": pipeline_result.mathematical_layer_result.violated_rules,
    }
)

print("\nPipeline RAG runtime summary:")
print_json(summarize_rag_result(pipeline_result.rag_retrieval_result))

# %% [markdown]
# ## 6. Notes
#
# If `indexed_chunks` is zero or the runtime is unavailable, fix the offline
# ingestion step separately and rerun this notebook. Keep retrieval and
# ingestion debugging separated so runtime behavior stays easy to inspect.
