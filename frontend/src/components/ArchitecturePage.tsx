export function ArchitecturePage() {
  return (
    <div className="arch-page">
      <div className="arch-header">
        <div className="arch-header-copy">
          <span className="eyebrow">System Reference</span>
          <h1 className="arch-title">Pipeline Architecture</h1>
          <p className="arch-subtitle">
            End-to-end data flow from scenario ingestion through deterministic scoring,
            RAG retrieval, and LLM ethical reasoning to the final framework analysis output.
          </p>
        </div>
        <div className="arch-header-meta">
          <div className="arch-meta-chip">6 stages</div>
          <div className="arch-meta-chip">Parallel S2 · S3</div>
          <div className="arch-meta-chip">6 ethical frameworks</div>
        </div>
      </div>

      {/* The inner div remaps all Claude artifact CSS variables to the active app theme tokens */}
      <div className="arch-diagram-wrap">
        <div
          className="arch-diagram-vars"
          dangerouslySetInnerHTML={{
            __html: `
<div style="padding:1rem 0;font-family:var(--font-sans);">

<!-- Stage 0 -->
<div style="border:1.5px dashed var(--color-border-secondary);border-radius:var(--border-radius-lg);overflow:hidden;">
  <div style="padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--color-border-tertiary);background:var(--color-background-secondary);">
    <span style="font-size:10px;font-weight:500;color:var(--color-text-secondary);border:1px solid var(--color-border-secondary);border-radius:10px;padding:1px 7px;">offline · setup</span>
    <span style="font-size:13px;font-weight:500;color:var(--color-text-primary);">Stage 0 — Ethical knowledge base ingestion</span>
    <span style="margin-left:auto;font-size:10px;color:var(--color-text-tertiary);font-family:var(--font-mono);">KnowledgeBaseIngester</span>
  </div>
  <div style="padding:10px 14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;">
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Source files</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">.md · .txt · .json · .pdf<br>knowledge_base/ directory<br>ignored: readme · .gitkeep</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Framework docs</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">ethical_frameworks/ → stored <em>whole</em><br>no chunking — preserves full<br>decision_logic + tradeoffs</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">General docs</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">chunked: 400 tok · 60 overlap<br>RecursiveCharacterTextSplitter<br>tiktoken encoder (cl100k_base)</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Embedding fields (frameworks)</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">foundation · decision_logic · pros<br>cons · tradeoffs · use_when<br>avoid_when · dominant_when</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Embedding model</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">OpenAIEmbeddings<br>text-embedding-ada-002<br>OPENAI_API_KEY required</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Vector store</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">ChromaDB PersistentClient<br>LangChain Chroma collection<br>persists to .chroma/</div>
    </div>
  </div>
</div>

<!-- connector 0→1 -->
<div style="display:flex;flex-direction:column;align-items:center;height:38px;justify-content:center;gap:0;">
  <div style="width:1.5px;flex:1;background:var(--color-border-secondary);"></div>
  <div style="font-size:10px;color:var(--color-text-tertiary);padding:1px 8px;">vector store ready</div>
  <div style="width:1.5px;flex:1;background:var(--color-border-secondary);"></div>
  <div style="width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid var(--color-border-secondary);"></div>
</div>

<!-- Stage 1 -->
<div style="border:1.5px solid #1D9E75;border-radius:var(--border-radius-lg);overflow:hidden;">
  <div style="padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--color-border-tertiary);">
    <span style="font-size:10px;font-weight:500;color:#1D9E75;font-family:var(--font-mono);">STAGE 1</span>
    <span style="font-size:13px;font-weight:500;color:var(--color-text-primary);">Scenario input</span>
    <span style="margin-left:auto;font-size:10px;color:var(--color-text-tertiary);font-family:var(--font-mono);">Scenario</span>
  </div>
  <div style="padding:10px 14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;">
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Ego vehicle</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">speed_kmh · mass_kg<br>braking_distance_m</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Obstacles</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">type · vulnerability_class<br>distance_m · trajectory<br>responsible_for_risk</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Environment</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">road_type · weather<br>visibility_m · speed_limit_kmh<br>traffic_density</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Sensor confidence</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">lidar · camera · radar<br>overall_scene_confidence</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Available actions</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">brake_straight · swerve_left<br>swerve_right · brake_swerve_left<br>brake_swerve_right …</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Flags &amp; occlusion zones</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">collision_unavoidable (bool)<br>zones: left/right sidewalk<br>crosswalk · bike lane · intersection</div>
    </div>
  </div>
</div>

<!-- branch label -->
<div style="text-align:center;font-size:10px;color:var(--color-text-tertiary);margin:4px 0 0;">scenario feeds both stages in parallel</div>
<!-- branch SVG -->
<svg width="100%" viewBox="0 0 680 32" style="display:block;overflow:visible;">
  <line x1="340" y1="0" x2="340" y2="14" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <line x1="168" y1="14" x2="512" y2="14" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <line x1="168" y1="14" x2="168" y2="32" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <line x1="512" y1="14" x2="512" y2="32" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
</svg>

<!-- Parallel: Stage 2 + Stage 3 -->
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">

  <!-- Stage 2 -->
  <div style="border:1.5px solid #7F77DD;border-radius:var(--border-radius-lg);overflow:hidden;">
    <div style="padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--color-border-tertiary);">
      <span style="font-size:10px;font-weight:500;color:#7F77DD;font-family:var(--font-mono);">STAGE 2</span>
      <span style="font-size:13px;font-weight:500;color:var(--color-text-primary);">RAG retrieval</span>
    </div>
    <div style="padding:10px 14px;display:flex;flex-direction:column;gap:7px;">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #7F77DD;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Query &amp; search</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">scenario semantics → embedded<br>ChromaDB cosine similarity (top-k)<br>DeterministicRAGRetriever</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #7F77DD;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">EF-02 guarantee</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">deontological safety always<br>force-included in rag_context<br>regardless of retrieval rank</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #7F77DD;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Retrieved fields per framework</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">framework_id · foundation<br>decision_logic · pros · cons<br>best_fit / poor_fit scenarios<br>use_when · avoid_when<br>dominant_when · tradeoffs</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #7F77DD;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Output</div>
        <div style="font-size:10px;color:var(--color-text-secondary);">rag_context — structured list of EKB entries</div>
      </div>
    </div>
  </div>

  <!-- Stage 3 -->
  <div style="border:1.5px solid #BA7517;border-radius:var(--border-radius-lg);overflow:hidden;">
    <div style="padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--color-border-tertiary);">
      <span style="font-size:10px;font-weight:500;color:#BA7517;font-family:var(--font-mono);">STAGE 3</span>
      <span style="font-size:13px;font-weight:500;color:var(--color-text-primary);">Deterministic mathematical layer</span>
    </div>
    <div style="padding:10px 14px;display:flex;flex-direction:column;gap:7px;">
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #BA7517;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Global metrics</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">sensor_fusion = 0.4×overall + 0.2×(lidar+camera+radar)<br>scene_uncertainty = 1 − sensor_fusion<br>visibility_pressure · speed_limit_delta · braking_margin_m</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #BA7517;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Collision probability (per action × obstacle)</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">0.45×braking_p + 0.35×time_p + 0.07×uncertainty<br>+ 0.05×visibility + 0.04×weather + 0.04×traffic<br>+ 0.10 bonus if collision_unavoidable</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #BA7517;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Harm estimate</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">KE × mass_factor × vuln_mult × angle_factor<br>vuln: high=2.5 · medium=1.5 · low=0.9<br>risk_score = P(collision) × harm_estimate</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #BA7517;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Occlusion zones</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">exposure × zone_factor × uncertainty weights<br>P capped at 0.40 per zone<br>swerve into high-exposure → constraint_flag</div>
      </div>
      <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;border-left:3px solid #BA7517;">
        <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Output</div>
        <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">risk_score_matrix · action_assessments<br>violated_rules (speed_limit_exceeded · cannot_stop_within_visible_distance)<br>best_action_by_total_risk</div>
      </div>
    </div>
  </div>

</div>

<!-- merge SVG -->
<svg width="100%" viewBox="0 0 680 36" style="display:block;overflow:visible;">
  <line x1="168" y1="0" x2="168" y2="18" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <line x1="512" y1="0" x2="512" y2="18" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <line x1="168" y1="18" x2="512" y2="18" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <line x1="340" y1="18" x2="340" y2="29" stroke="var(--color-border-secondary)" stroke-width="1.5"/>
  <path d="M334 25 L340 33 L346 25" fill="none" stroke="var(--color-border-secondary)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<!-- merge label -->
<div style="text-align:center;font-size:10px;color:var(--color-text-tertiary);margin:0 0 4px;">rag_context · mathematical_layer · scenario</div>

<!-- Stage 4 -->
<div style="border:1.5px solid #D85A30;border-radius:var(--border-radius-lg);overflow:hidden;">
  <div style="padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--color-border-tertiary);">
    <span style="font-size:10px;font-weight:500;color:#D85A30;font-family:var(--font-mono);">STAGE 4</span>
    <span style="font-size:13px;font-weight:500;color:var(--color-text-primary);">LLM ethical reasoning</span>
    <span style="margin-left:auto;font-size:10px;color:var(--color-text-tertiary);font-family:var(--font-mono);">ETHICAL_REASONING_SYSTEM_PROMPT</span>
  </div>
  <div style="padding:10px 14px;display:flex;flex-direction:column;gap:8px;">
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:9px 11px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:5px;">Reasoning steps</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 14px;font-size:10px;color:var(--color-text-secondary);line-height:1.55;">
        <div>① Classify: collision avoidable? vulnerable road users present? passenger vs pedestrian dilemma?</div>
        <div>② Consult EKB: use_when / avoid_when / dominant_when / decision_logic / tradeoffs per retrieved framework</div>
        <div>③ Apply math layer: risk_score_matrix, RSS constraint flags, best_action_by_total_risk (EF-01 reference only, not default)</div>
        <div>④ Select dominant_framework via dominant_when fields — framework that shaped the decision, not the one with most numerical support</div>
        <div>⑤ Set EF-04 weights: bayesian + equality + maximin = 1.0 (reflect ethical balance of this scenario)</div>
        <div>⑥ Return framework-centered JSON; discuss actions only in the rationale, not as a dedicated output field</div>
      </div>
    </div>
    <div>
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:5px;">Six ethical frameworks — Ethical Knowledge Base (EKB)</div>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:5px;">
        <div style="background:var(--color-background-secondary);border-radius:8px;padding:7px 9px;border-left:3px solid #1D9E75;">
          <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);">EF-01</div>
          <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.4;margin-top:2px;">Utilitarian risk minimization</div>
          <div style="font-size:10px;color:var(--color-text-tertiary);font-style:italic;margin-top:1px;">minimise total expected harm</div>
        </div>
        <div style="background:var(--color-background-secondary);border-radius:8px;padding:7px 9px;border-left:3px solid #7F77DD;">
          <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);">EF-02</div>
          <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.4;margin-top:2px;">Deontological safety</div>
          <div style="font-size:10px;color:var(--color-text-tertiary);font-style:italic;margin-top:1px;">RSS rules as hard constraints</div>
        </div>
        <div style="background:var(--color-background-secondary);border-radius:8px;padding:7px 9px;border-left:3px solid #378ADD;">
          <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);">EF-03</div>
          <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.4;margin-top:2px;">Rawlsian maximin</div>
          <div style="font-size:10px;color:var(--color-text-tertiary);font-style:italic;margin-top:1px;">protect the most vulnerable</div>
        </div>
        <div style="background:var(--color-background-secondary);border-radius:8px;padding:7px 9px;border-left:3px dashed var(--color-border-secondary);">
          <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);">EF-04 <span style="font-weight:400;font-size:9px;color:var(--color-text-tertiary);">never dominant</span></div>
          <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.4;margin-top:2px;">Ethics of risk — hybrid substrate</div>
          <div style="font-size:10px;color:var(--color-text-tertiary);font-style:italic;margin-top:1px;">bayesian · equality · maximin weights</div>
        </div>
        <div style="background:var(--color-background-secondary);border-radius:8px;padding:7px 9px;border-left:3px solid #D85A30;">
          <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);">EF-05</div>
          <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.4;margin-top:2px;">Ethical valence theory</div>
          <div style="font-size:10px;color:var(--color-text-tertiary);font-style:italic;margin-top:1px;">harm min + social valence hierarchy</div>
        </div>
        <div style="background:var(--color-background-secondary);border-radius:8px;padding:7px 9px;border-left:3px dashed var(--color-border-secondary);">
          <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);">EF-06 <span style="font-weight:400;font-size:9px;color:var(--color-text-tertiary);">fallback</span></div>
          <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.4;margin-top:2px;">Virtue ethics</div>
          <div style="font-size:10px;color:var(--color-text-tertiary);font-style:italic;margin-top:1px;">reasonable driver standard · explain layer</div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- connector 4→5 -->
<div style="display:flex;flex-direction:column;align-items:center;height:38px;justify-content:center;gap:0;">
  <div style="width:1.5px;flex:1;background:var(--color-border-secondary);"></div>
  <div style="font-size:10px;color:var(--color-text-tertiary);">JSON output</div>
  <div style="width:1.5px;flex:1;background:var(--color-border-secondary);"></div>
  <div style="width:0;height:0;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid var(--color-border-secondary);"></div>
</div>

<!-- Stage 5 -->
<div style="border:1.5px solid #639922;border-radius:var(--border-radius-lg);overflow:hidden;">
  <div style="padding:9px 14px;display:flex;align-items:center;gap:8px;border-bottom:1px solid var(--color-border-tertiary);">
    <span style="font-size:10px;font-weight:500;color:#639922;font-family:var(--font-mono);">STAGE 5</span>
    <span style="font-size:13px;font-weight:500;color:var(--color-text-primary);">Policy output</span>
    <span style="margin-left:auto;font-size:10px;color:var(--color-text-tertiary);">JSON schema</span>
  </div>
  <div style="padding:10px 14px;display:grid;grid-template-columns:1fr 1fr 1fr;gap:7px;">
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Decision</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">dominant_framework<br>contributing_frameworks<br>weights_reasoning</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">EF-04 weights</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">bayesian · equality · maximin<br>∑ = 1.0<br>+ weights_reasoning</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Justification</div>
      <div style="font-size:10px;color:var(--color-text-secondary);line-height:1.55;">rationale (&lt;60 words)<br>confidence (0–1)<br>violated_constraints</div>
    </div>
    <div style="background:var(--color-background-secondary);border-radius:8px;padding:8px 10px;grid-column:span 3;">
      <div style="font-size:11px;font-weight:500;color:var(--color-text-primary);margin-bottom:3px;">Risk scores — source of truth</div>
      <div style="font-size:10px;color:var(--color-text-secondary);">risk_scores_per_action — verbatim copy of risk_score_matrix from Stage 3 · never altered by LLM · maps action → stakeholder → score</div>
    </div>
  </div>
</div>

</div>`,
          }}
        />
      </div>
    </div>
  )
}
