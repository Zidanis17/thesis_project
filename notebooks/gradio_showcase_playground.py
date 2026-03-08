from pathlib import Path
import json
import sys
import traceback

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "Gradio is not installed. Install project dependencies first, then rerun this notebook."
    ) from exc

root = Path.cwd()
if not (root / "thesis").exists():
    root = root.parent

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from thesis import DeterministicRAGRetriever, EthicalReasoningLLM, ScenarioPipeline

rag_retriever = DeterministicRAGRetriever()
reasoning_llm = EthicalReasoningLLM(model_name="gpt-4o-mini", temperature=0.0)
pipeline = ScenarioPipeline(
    rag_retriever=rag_retriever,
    reasoning_llm=reasoning_llm,
)

# ── Example inputs ─────────────────────────────────────────────────────────────

EXAMPLE_NL = (
    "An autonomous vehicle weighing 1800 kg is traveling at 60 km/h in the center lane on a "
    "residential road with a 50 km/h speed limit. It is braking at 2.1 m/s2 and has a braking "
    "distance of 42.5 m. Weather is clear, visibility is 120 m, it is daytime, and traffic is low. "
    "A child pedestrian is crossing 10.2 m ahead with time to impact 0.61 s. "
    "A parked vehicle is 6.5 m ahead with time to impact 0.39 s. "
    "Lidar confidence is 0.97, camera 0.91, radar 0.95, and overall scene confidence is 0.93. "
    "The left sidewalk is occluded. Available actions are brake straight, swerve left, swerve right, "
    "and brake and swerve left. Collision is unavoidable."
)

EXAMPLE_JSON = json.dumps(
    {
        "ego_vehicle": {
            "speed_kmh": 55,
            "acceleration_ms2": -3.1,
            "heading_deg": 0,
            "lane_position": "center",
            "braking_distance_m": 48.2,
            "mass_kg": 1900,
        },
        "environment": {
            "road_type": "urban_arterial",
            "speed_limit_kmh": 60,
            "weather": "clear",
            "visibility_m": 110,
            "time_of_day": "daytime",
            "traffic_density": "medium",
        },
        "obstacles": [
            {
                "id": "obj_01",
                "type": "pedestrian_adult",
                "distance_m": 11.0,
                "relative_speed_kmh": 55,
                "time_to_impact_s": 0.72,
                "trajectory": "crossing_jaywalking",
                "vulnerability_class": "high",
                "mass_kg": 80,
                "responsible_for_risk": True,
            },
            {
                "id": "obj_02",
                "type": "pedestrian_adult",
                "distance_m": 11.3,
                "relative_speed_kmh": 55,
                "time_to_impact_s": 0.74,
                "trajectory": "crossing_jaywalking",
                "vulnerability_class": "high",
                "mass_kg": 75,
                "responsible_for_risk": True,
            },
            {
                "id": "obj_03",
                "type": "pedestrian_adult",
                "distance_m": 11.1,
                "relative_speed_kmh": 55,
                "time_to_impact_s": 0.73,
                "trajectory": "crossing_jaywalking",
                "vulnerability_class": "high",
                "mass_kg": 70,
                "responsible_for_risk": True,
            },
            {
                "id": "obj_04",
                "type": "parked_vehicle",
                "distance_m": 8.7,
                "relative_speed_kmh": 55,
                "time_to_impact_s": 0.57,
                "trajectory": "stationary_right_shoulder",
                "vulnerability_class": "low",
                "mass_kg": 1400,
                "responsible_for_risk": False,
            },
        ],
        "sensor_confidence": {
            "lidar": 0.96,
            "camera": 0.94,
            "radar": 0.95,
            "overall_scene_confidence": 0.95,
            "occluded_zones": [],
        },
        "available_actions": ["brake_straight", "swerve_left", "swerve_right"],
        "collision_unavoidable": True,
        "_meta": {
            "input_mode": "sensor_fusion",
            "warnings": [
                "obj_01, obj_02, obj_03 confirmed mid-block crossing against signal — all responsible_for_risk",
                "swerve_right impacts unoccupied parked vehicle only — no persons in or around obj_04",
                "swerve_left enters oncoming lane confirmed clear by radar",
                "brake_straight projects full-speed impact into pedestrian group",
            ],
        },
    },
    indent=2,
)

EXAMPLE_JSON_2 = json.dumps(
    {
        "ego_vehicle": {
            "speed_kmh": 60,
            "acceleration_ms2": -3.8,
            "heading_deg": 0,
            "lane_position": "center",
            "braking_distance_m": 48.0,
            "mass_kg": 1900,
        },
        "environment": {
            "road_type": "urban_arterial",
            "speed_limit_kmh": 50,
            "weather": "clear",
            "visibility_m": 95,
            "time_of_day": "daytime",
            "traffic_density": "medium",
        },
        "obstacles": [
            {
                "id": "obj_01",
                "type": "impaired_driver_vehicle",
                "distance_m": 18.0,
                "relative_speed_kmh": 60,
                "time_to_impact_s": 1.08,
                "trajectory": "head_on_wrong_lane",
                "vulnerability_class": "medium",
                "mass_kg": 1600,
                "responsible_for_risk": True,
            },
            {
                "id": "obj_02",
                "type": "pedestrian_adult",
                "distance_m": 11.0,
                "relative_speed_kmh": 60,
                "time_to_impact_s": 0.55,
                "trajectory": "crossing",
                "vulnerability_class": "child",
                "mass_kg": 72,
                "responsible_for_risk": False,
            },
            {
                "id": "obj_03",
                "type": "pedestrian_adult",
                "distance_m": 12.0,
                "relative_speed_kmh": 60,
                "time_to_impact_s": 0.60,
                "trajectory": "crossing",
                "vulnerability_class": "pedestrian",
                "mass_kg": 68,
                "responsible_for_risk": False,
            },
        ],
        "sensor_confidence": {
            "lidar": 0.97,
            "camera": 0.95,
            "radar": 0.96,
            "overall_scene_confidence": 0.96,
            "occluded_zones": ["right sidewalk"],
        },
        "available_actions": ["brake_straight", "swerve_left", "swerve_right"],
        "collision_unavoidable": False,
        "_meta": {
            "input_mode": "sensor_fusion",
            "warnings": [
                "Speeding above posted limit; braking required by RSS.",
                "swerve_left crosses into oncoming obj_01.",
                "swerve_right mounts right sidewalk with occluded zone and pedestrians crossing.",
            ],
        },
    },
    indent=2,
)

# ── Panel summarisers ──────────────────────────────────────────────────────────

def summarize_rag_result(rag_result) -> dict:
    """
    Build a display-safe dict for the RAG tab.

    Frameworks now arrive through retrieved_documents (scored retrieval path).
    always_included_documents is only populated when the vector store is down
    (disk fallback). Never expose full_content — it is a raw JSON blob.
    """
    if rag_result is None:
        return {"runtime_status": "not_requested", "reason": "RAG stage not provided."}

    framework_docs = []
    supporting_docs = []

    for doc in rag_result.retrieved_documents:
        entry = {
            "title": doc.title,
            "path": doc.path,
            "score": doc.score,
            "excerpt": doc.excerpt[:400] if doc.excerpt else "",
        }
        if doc.category == "ethical_frameworks":
            framework_docs.append(entry)
        else:
            entry["category"] = doc.category
            supporting_docs.append(entry)

    # Disk-fallback frameworks (vector store unavailable)
    fallback_docs = [
        {
            "title": doc.title,
            "path": doc.path,
            "score": "fallback",
            "content_chars": len(doc.content),
        }
        for doc in rag_result.always_included_documents
        if doc.category == "ethical_frameworks"
    ]

    return {
        "runtime_available": rag_result.runtime_available,
        "runtime_error": rag_result.runtime_error,
        "indexed_chunks": rag_result.indexed_chunks,
        "query": rag_result.query,
        "frameworks_retrieved": len(framework_docs) + len(fallback_docs),
        "supporting_docs_retrieved": len(supporting_docs),
        "frameworks": framework_docs or fallback_docs,
        "supporting_documents": supporting_docs,
    }


def summarize_reasoning_result(reasoning_result) -> dict:
    """
    Build a display-safe dict for the Reasoning tab.
    Drops system_prompt — it is the full multi-line prompt and makes the tab unreadable.
    """
    if reasoning_result is None:
        return {"runtime_status": "not_requested", "reason": "Reasoning stage not provided."}
    d = reasoning_result.to_dict()
    d.pop("system_prompt", None)
    return d


# ── Runtime banner ─────────────────────────────────────────────────────────────

def build_runtime_banner() -> str:
    rag_ok = rag_retriever.vector_store is not None
    llm_ok = reasoning_llm.client is not None
    rag_status = "✅ ready" if rag_ok else "⚠️ unavailable — will use disk fallback"
    llm_status = f"✅ {reasoning_llm.model_name}" if llm_ok else "⚠️ unavailable — check OPENAI_API_KEY"
    return "\n".join([
        "### Runtime",
        f"- Knowledge base: `{rag_retriever.knowledge_base_path}`",
        f"- RAG retriever: {rag_status}",
        f"- Reasoning LLM: {llm_status}",
        "- Input accepts natural language descriptions or structured JSON.",
    ])


# ── Summary panel ──────────────────────────────────────────────────────────────

def build_summary_message(result) -> str:
    parser_result = result.parser_result
    math_result = result.mathematical_layer_result
    reasoning_result = result.reasoning_result
    rag_result = result.rag_retrieval_result

    lines = [
        f"**Input mode:** `{parser_result.input_mode}`",
        f"**Deterministic best action:** `{math_result.best_action_by_total_risk}`",
    ]

    if parser_result.warnings:
        for w in parser_result.warnings:
            lines.append(f"⚠️ {w}")

    lines.append(
        "**RSS violations:** " + (", ".join(f"`{r}`" for r in math_result.violated_rules) or "none")
    )

    # RAG summary — frameworks come through retrieved_documents in normal operation
    if rag_result is not None:
        fw_retrieved = [d for d in rag_result.retrieved_documents if d.category == "ethical_frameworks"]
        other_retrieved = [d for d in rag_result.retrieved_documents if d.category != "ethical_frameworks"]
        fallback_fw = [d for d in rag_result.always_included_documents if d.category == "ethical_frameworks"]
        total_fw = len(fw_retrieved) + len(fallback_fw)

        if rag_result.runtime_available:
            rag_line = f"**RAG:** {total_fw} framework(s) retrieved"
            if other_retrieved:
                rag_line += f" + {len(other_retrieved)} supporting doc(s)"
            if fallback_fw:
                rag_line += " *(disk fallback — vector store unavailable)*"
            lines.append(rag_line)
            if fw_retrieved:
                lines.append("Retrieved: " + ", ".join(f"`{d.title}`" for d in fw_retrieved))
        else:
            lines.append(f"⚠️ **RAG unavailable:** `{rag_result.runtime_error}`")

    lines.append("")

    # Reasoning result
    if reasoning_result is not None and reasoning_result.runtime_available:
        lines += [
            f"### 🎯 Recommended action: `{reasoning_result.recommended_action}`",
            f"**Dominant framework:** `{reasoning_result.dominant_framework}`",
            "**Contributing:** " + (
                ", ".join(f"`{f}`" for f in reasoning_result.contributing_frameworks) or "none"
            ),
            "**Weights:** " + ", ".join(
                f"{k}=`{v:.3f}`" for k, v in reasoning_result.weights.items()
            ),
            f"**Confidence:** `{reasoning_result.confidence}`",
            "**Action constraints:** " + (
                ", ".join(f"`{c}`" for c in reasoning_result.violated_constraints) or "none"
            ),
            "",
            "**Rationale:**",
            reasoning_result.rationale,
        ]
    else:
        lines.append("⚠️ Reasoning LLM unavailable — deterministic layer output only.")
        if reasoning_result is not None and reasoning_result.runtime_error:
            lines.append(f"**Error:** `{reasoning_result.runtime_error}`")

    return "\n".join(lines)


# ── Pipeline runner ────────────────────────────────────────────────────────────

def empty_panel() -> dict:
    return {"status": "waiting_for_input"}


def run_showcase(message):
    user_message = (message or "").strip()
    if not user_message:
        raise gr.Error("Enter a scenario description or paste a structured JSON payload.")

    try:
        result = pipeline.run(user_message)
        summary_msg = build_summary_message(result)
        parser_payload = result.parser_result.to_dict()
        math_payload = result.mathematical_layer_result.to_dict()
        rag_payload = summarize_rag_result(result.rag_retrieval_result)
        reasoning_payload = summarize_reasoning_result(result.reasoning_result)
    except Exception as exc:
        tb = traceback.format_exc()
        summary_msg = f"**Pipeline error:** `{exc}`\n\n```\n{tb}\n```"
        parser_payload = {"error": str(exc), "traceback": tb}
        math_payload = empty_panel()
        rag_payload = empty_panel()
        reasoning_payload = empty_panel()

    return summary_msg, parser_payload, math_payload, rag_payload, reasoning_payload


def clear_showcase():
    e = empty_panel()
    return "", "Run a scenario to see the recommendation summary.", e, e, e, e


# ── Gradio UI ──────────────────────────────────────────────────────────────────

with gr.Blocks(title="AV Ethics Pipeline Showcase", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # AV Ethics Pipeline Showcase
        Paste a scenario in natural language or JSON. The left pane shows the recommendation
        summary; the tabs on the right expose raw pipeline outputs per stage.
        """
    )
    gr.Markdown(build_runtime_banner())

    with gr.Row():
        with gr.Column(scale=3):
            user_input = gr.Textbox(
                label="Scenario Input",
                placeholder="Describe an AV dilemma or paste the structured JSON payload...",
                lines=14,
            )
            summary_output = gr.Markdown("Run a scenario to see the recommendation summary.")
            with gr.Row():
                send_button = gr.Button("Run Scenario", variant="primary")
                clear_button = gr.Button("Clear")
            gr.Examples(
                examples=[[EXAMPLE_NL], [EXAMPLE_JSON], [EXAMPLE_JSON_2]],
                inputs=user_input,
                label="Example Inputs",
                example_labels=[
                    "Natural Language — child crossing",
                    "JSON 1 — jaywalking group (utilitarian dilemma)",
                    "JSON 2 — wrong-lane + bus stop (deontological hard rejection)",
                ],
            )

        with gr.Column(scale=2):
            with gr.Tab("Parser"):
                parser_output = gr.JSON(label="Parser Result", value=empty_panel())
            with gr.Tab("Math"):
                math_tab_output = gr.JSON(label="Mathematical Layer Result", value=empty_panel())
            with gr.Tab("RAG"):
                rag_output = gr.JSON(label="RAG Retrieval Result", value=empty_panel())
            with gr.Tab("Reasoning"):
                reasoning_output = gr.JSON(label="Reasoning Result", value=empty_panel())

    # NOTE: math_tab_output (not math_output) — avoids name collision with any
    # local variable named math_output from earlier pipeline calls.
    pipeline_outputs = [summary_output, parser_output, math_tab_output, rag_output, reasoning_output]
    clear_outputs = [user_input, summary_output, parser_output, math_tab_output, rag_output, reasoning_output]

    send_button.click(run_showcase, inputs=user_input, outputs=pipeline_outputs)
    user_input.submit(run_showcase, inputs=user_input, outputs=pipeline_outputs)
    clear_button.click(clear_showcase, outputs=clear_outputs)
