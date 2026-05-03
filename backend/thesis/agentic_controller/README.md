# Bounded Agentic Ethical Controller

The controller is not an autonomous moral decision-maker. It does not invent ethical principles and does not replace the Ethical Knowledge Base. It classifies the scenario, produces constrained retrieval intent, routes attention toward predefined ethical frameworks, and validates the final reasoning output against explicit constraints.

The controller runs after parsing and the mathematical layer. It inspects the structured scenario, math-layer signals such as scene interpretability, and bounded scenario features such as vulnerable road users, unknown objects, and passenger-versus-VRU tradeoff evidence.

RAG remains active. The LLM does not freely write retrieval queries. The controller produces a `RetrievalIntent`, and the deterministic RAG query builder converts that intent into controlled retrieval text appended to its existing deterministic query and heuristic hints.

The reasoning LLM receives the controller output as structured context. It is told to treat that output as a routing and validation aid, not as an independent moral authority.

Validation is non-blocking for now. If the controller finds errors or warnings in the reasoning result, the pipeline includes those findings in the returned artifacts and replay stages but does not overwrite the LLM reasoning output.
