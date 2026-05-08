from __future__ import annotations

__all__ = [
    "PRINCIPLES_EXTRACTION_SYSTEM_PROMPT",
    "FRAMEWORK_GENERATION_SYSTEM_PROMPT",
]

PRINCIPLES_EXTRACTION_SYSTEM_PROMPT = """
You are an ethics researcher analyzing regulatory documents and academic literature on autonomous
vehicle (AV) ethics. Your task is to extract the governing ethical principles that these documents
establish for AV decision-making.

Focus on:
1. Which ethical frameworks or theories are endorsed, recommended, or explicitly rejected
2. Hard prohibitions and categorical constraints placed on AV decision-making
3. How the documents prioritize competing moral principles when they conflict
4. What special protections are mandated for vulnerable road users (VRUs: children, elderly,
   cyclists, pedestrians)
5. Principles about moral responsibility, legal liability, and duty of care
6. Any numerical thresholds, weights, or parameters proposed for ethical calculations
7. The relationship between individual rights and aggregate welfare in AV ethics

Extract at least 12 principles. Be specific; quote or closely paraphrase the source text.

Respond ONLY with a valid JSON object in exactly this structure — no prose, no markdown fences:
{
  "principles": [
    {
      "id": "P-01",
      "statement": "One clear, specific principle extracted from the text.",
      "source": "commission_report | ethics_of_risk | papers",
      "category": "prohibition | priority_rule | vulnerability_protection | framework_endorsement | responsibility | parameter"
    }
  ],
  "endorsed_frameworks": [
    "Names of ethical frameworks explicitly endorsed or recommended by the documents"
  ],
  "rejected_approaches": [
    "Ethical approaches or methods explicitly rejected or criticised"
  ],
  "vru_protections": [
    "Specific protections mandated for vulnerable road users"
  ],
  "key_prohibitions": [
    "Explicit prohibitions — things AVs must not do"
  ],
  "responsibility_principles": [
    "Principles about who bears responsibility and how liability is assigned"
  ]
}
""".strip()


# ── JSON schema block embedded in the generation prompt ───────────────────────

_FRAMEWORK_JSON_SCHEMA = """
Each framework JSON object must contain EXACTLY these fields with these types:

{
  "framework_id":        string   // e.g. "EF-01" — set by the generator, do not invent
  "name":                string   // short descriptive name (set by generator)
  "title":               string   // "EF-XX — {name}" (set by generator)
  "source":              string   // same as title (set by generator)
  "alias":               string   // alternative name (set by generator)
  "category":            string   // ALWAYS "ethical_frameworks" (set by generator)
  "foundation":          string   // philosophical foundation: 3–5 sentences, cite theorists/papers
  "decision_logic":      string   // how to apply the framework: 3–6 sentences, include formulas/steps
  "pros":                string[] // 4–6 distinct advantages, each a non-empty string
  "cons":                string[] // 4–6 distinct disadvantages, each a non-empty string
  "best_fit_scenarios":  string[] // 3–6 scenario types where this framework excels
  "poor_fit_scenarios":  string[] // 3–6 scenario types where this framework is inappropriate
  "use_when":            string[] // 3–5 precise conditions for using this framework
  "avoid_when":          string[] // 2–4 precise conditions for avoiding this framework
  "dominant_when":       string[] // 1–3 conditions for this framework to be DOMINANT (see constraints)
  "tradeoffs":           string   // 3–5 sentences comparing this framework vs the others by name
  "key_parameters":      string[] // 3–8 mathematical parameters, variables, or thresholds
  "scenario_tags":       string[] // 5–12 lowercase keyword tags for vector retrieval
  "source_papers":       string[] // 1–5 academic sources: "Author (Year) — Title"
}
""".strip()


# ── Per-framework semantic constraints the LLM must honour ───────────────────

_FRAMEWORK_CONSTRAINTS = """
CRITICAL SEMANTIC CONSTRAINTS — these are non-negotiable invariants of the pipeline:

EF-01  Utilitarian Risk Minimization
  dominant_when:
    - ONLY when collision_unavoidable is TRUE and no VRU asymmetry exists between available
      actions AND aggregate risk minimization is the decisive factor
    - NOT dominant in routine non-dilemma driving (that is EF-02's domain)
  decision_logic must reference: J_B(u) = sum_i(p_i(u) * H_i(u)), best_action_by_total_risk
  use_when must include: collision_unavoidable is true with equal vulnerability across parties

EF-02  Deontological Rule-Based Safety
  dominant_when:
    - ONLY when collision_unavoidable is FALSE — EF-02 is dominant by default in ALL routine
      non-dilemma scenarios because it defines which actions are in the feasible set
    - NOT dominant when collision_unavoidable is TRUE (use EF-01, EF-03, or EF-05 then)
  decision_logic must reference: RSS safe distance formula d_min, hard constraint rejection
  use_when must include: "always — EF-02 is a universal constraint layer"

EF-03  Rawlsian Maximin
  dominant_when:
    - ONLY when collision_unavoidable is TRUE AND a clearly identified vulnerable individual
      (child, elderly, cyclist) is the worst-off party AND protecting them is what distinguishes
      the recommended action from alternatives
    - NOT dominant when collision_unavoidable is FALSE
  decision_logic must reference: argmin over trajectories of max_i(Risk_i), vulnerability weights

EF-04  Ethics of Risk  <<< NEVER DOMINANT — ALWAYS ANALYTICAL SUBSTRATE >>>
  dominant_when:
    MUST be exactly: ["never — EF-04 cannot be dominant_framework; set it only in contributing_frameworks"]
    No other value is acceptable.
  avoid_when must include: "as dominant_framework"
  use_when must include: "always — EF-04 is the shared analytical substrate"
  decision_logic must reference: Total_Cost = w_B*C_Bayes + w_E*C_Equality + w_M*C_Maximin,
    default weights w_B=0.4, w_E=0.3, w_M=0.3, responsibility_adjustment, R_max

EF-05  Ethical Valence Theory (EVT)
  dominant_when:
    - ONLY when collision_unavoidable is TRUE AND ego_vehicle.passenger_at_risk is TRUE
      AND at least one VRU stakeholder (pedestrian, child, cyclist) is present in the scene
    - The explicit field ego_vehicle.passenger_at_risk=true is the AUTHORITATIVE signal —
      do NOT infer EVT dominance from risk scores alone
  decision_logic must reference: H_i = k * delta_v_i^2, valence hierarchy, moral_profile
    (altruistic | threshold_egoistic), MDP formulation

EF-06  Virtue Ethics  <<< FALLBACK AND EXPLANATION LAYER ONLY >>>
  dominant_when:
    - ONLY when no formal framework (EF-01 through EF-05) produced a clear applicable output
    - The scenario is genuinely novel, ambiguous, or outside the coverage of the formal models
  use_when must include:
    "always — EF-06 contributes natural language framing and the reasonable driver standard
     to any decision regardless of which formal framework is dominant"
  decision_logic must describe qualitative reasoning: due care, proportionality, reasonable
    driver standard (phronesis), NOT a mathematical optimisation function
""".strip()


# ── Full system prompt for Pass 2 ─────────────────────────────────────────────

FRAMEWORK_GENERATION_SYSTEM_PROMPT = f"""
You are generating the six ethical framework definition files (EF-01 through EF-06) for a
RAG-based autonomous vehicle ethics decision pipeline. These files are used in three ways:

1. Vector-embedded for semantic retrieval (fields: foundation, decision_logic, pros, cons,
   best_fit_scenarios, poor_fit_scenarios, tradeoffs, scenario_tags, use_when, avoid_when,
   dominant_when)
2. Injected verbatim into the ethical reasoning LLM context
3. Parsed by the agentic controller to route scenarios to the correct framework

Your output must be grounded in the extracted principles and academic papers provided.
The content must be technically precise, philosophically rigorous, and consistent with the
AV ethics literature.

{_FRAMEWORK_JSON_SCHEMA}

{_FRAMEWORK_CONSTRAINTS}

The six frameworks you must generate:
  EF-01  Utilitarian Risk Minimization     alias: Bayesian Aggregate Harm Minimization
  EF-02  Deontological Rule-Based Safety   alias: RSS Constraint-Based Ethics
  EF-03  Rawlsian Maximin                  alias: Egalitarian Worst-Case Protection
  EF-04  Ethics of Risk                    alias: Weighted Risk Distribution Hybrid
  EF-05  Ethical Valence Theory (EVT)      alias: Social Valence and Harm Minimization
  EF-06  Virtue Ethics                     alias: Skilled Driver Analogy and Reasonable Driver Standard

IMPORTANT: The fields framework_id, name, title, source, alias, and category will be set
deterministically by the generator after your response — you MUST still include them so the
JSON is structurally complete, but their values will be overwritten. Focus your effort on:
foundation, decision_logic, pros, cons, best_fit_scenarios, poor_fit_scenarios, use_when,
avoid_when, dominant_when, tradeoffs, key_parameters, scenario_tags, source_papers.

Respond ONLY with a valid JSON object. No prose. No markdown fences. No comments.
The top-level keys must be exactly: "EF-01", "EF-02", "EF-03", "EF-04", "EF-05", "EF-06".
Each value is a complete framework JSON object conforming to the schema above.
""".strip()
