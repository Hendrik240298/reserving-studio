# Streamlit And Snowflake Migration Summary

## Context

The current reserving-studio frontend is built with Dash. It supports the main reserving GUI, interactive Chainladder workflows, and the AI chat interface. The project goal is increasingly business-oriented: make the reserving workflow user-friendly, easy to access, shareable, and deployable in a Snowflake-centered environment.

Streamlit was discussed as a potential replacement because it is widely used by data analytics teams and has first-class support inside Snowflake.

## Snowflake Claim

The claim is broadly true: Snowflake supports native Streamlit apps through Streamlit in Snowflake.

Streamlit in Snowflake can:

- Run inside Snowflake/Snowsight.
- Use Snowflake-managed compute and storage.
- Use Snowflake RBAC for access control.
- Query Snowflake data without moving data or application code to a separate external hosting platform.
- Work with Snowpark, UDFs, stored procedures, and Snowflake Native App Framework.
- Be deployed through Snowsight, SQL, or Snowflake CLI.

However, this does not mean a Streamlit app becomes a public SaaS-style app automatically. It is primarily shareable inside the Snowflake environment and subject to Snowflake privileges, runtime constraints, dependency rules, external network access policies, and organizational governance.

## Why Move Toward Streamlit

If the main success criteria are business adoption, Snowflake-native access, and ease of sharing, Streamlit is strategically attractive.

Main reasons to move:

- Streamlit is the Snowflake-native Python app framework.
- Business users can access apps through the Snowflake environment instead of a separately hosted Dash server.
- Snowflake RBAC can govern access to the app and underlying data.
- Deployment and sharing should be easier in a Snowflake-centered organization.
- Actuaries and data analysts may already be familiar with Streamlit-style analytical apps.
- Streamlit is faster for simple analytical workflows and prototype iteration.
- Streamlit has built-in chat primitives such as `st.chat_input`, `st.chat_message`, `st.status`, and `st.write_stream`.
- A one-window AI assistant with sidebar context is likely simpler to build in Streamlit than Dash.
- Streamlit encourages workflow-oriented business screens, which may be more user-friendly than a complex technical cockpit.

The key conclusion was that Streamlit may be the better frontend direction if the project is intended to succeed as an accessible business application on Snowflake.

## Why Not Migrate Blindly

Streamlit is not superior to Dash for every interaction pattern.

Dash remains stronger for:

- Complex event-driven UIs.
- Fine-grained callback graphs.
- Multi-panel synchronization.
- Highly custom layout and styling.
- Interactive cell-level workflows.
- Existing Plotly/Dash callback-driven state flows.

The current Dash app already has substantial functionality, including reserving controls, interactive drop selection, recalculation flows, AI chat state, proposal acceptance/rejection, scenario ledgers, evidence traces, and E2E test coverage. Recreating this exactly in Streamlit would be a meaningful rewrite.

The recommendation is therefore not to copy the Dash app screen-for-screen. Instead, Streamlit should be used to redesign the frontend around clearer business workflows.

## Streamlit Interaction Model

Streamlit is script-based, but not limited to trivial scripts.

Its model is:

- The app runs as a Python script.
- On user interaction, Streamlit reruns the script from top to bottom.
- Per-user state is stored in `st.session_state`.
- Shared cached data or resources are handled through `st.cache_data` and `st.cache_resource`.
- Widget callbacks exist through `on_change` and `on_click`, but they are not equivalent to Dash callbacks.

Dash uses a global callback graph with `Input`, `Output`, and `State`. Streamlit uses a rerun-and-state model. This makes Streamlit simpler for many analytical apps, but it requires careful design for complex workflows.

OOP is still fully possible. Backend services, reserving engines, AI services, repositories, and controllers can remain class-based. The important point is that persistent state should live in `st.session_state`, a cache, a database, or another explicit persistence layer rather than relying on long-lived UI object instances.

## Interactive Drop Workflow

The interactive Chainladder drop workflow can be implemented in Streamlit, but it will not be a direct Dash callback translation.

A simple Streamlit version is feasible:

- Show the triangle or link-ratio table.
- Let the user select cells, rows, or ratios.
- Store selected drops in `st.session_state`.
- Recalculate the reserving scenario.
- Re-render updated results and diagnostics.

The full current Dash behavior may be harder to reproduce if it depends on fine-grained callback wiring, multiple synchronized stores, browser polling, and rich cell-level UI behavior. Streamlit is still viable, but the workflow should be redesigned around explicit business actions such as review, test scenario, accept proposal, and export/sign off.

## AI Chat And Sidebar

The AI chat interface is likely easier in Streamlit than Dash for a single-window application.

Streamlit provides native chat elements:

- `st.chat_input` for prompt entry.
- `st.chat_message` for user and assistant messages.
- `st.status` for long-running tool execution or review progress.
- `st.write_stream` for streamed responses.

The sidebar can also be integrated naturally using `st.sidebar`.

The sidebar could show:

- Analysis Basis.
- Current session summary.
- Scenario ledger.
- Evidence trace.
- Proposal status.
- Memory proposals.
- Preset prompts.
- Key reserving assumptions.

Streamlit sidebars are less flexible than a fully custom Dash layout, but they are well suited to structured contextual information.

## Snowflake Runtime Considerations

Streamlit in Snowflake has real constraints that need to be tested early.

Important considerations:

- Warehouse runtimes and container runtimes have different capabilities.
- Warehouse runtime dependencies are limited to the Snowflake Anaconda channel.
- Container runtime dependency installation is more flexible but can require External Access Integration for PyPI.
- Python and Streamlit versions are constrained by runtime choice.
- External resources, external scripts, and custom components are restricted by Snowflake security policies.
- File upload and frontend message sizes have runtime-specific limits.
- Caching behavior differs between warehouse and container runtimes.
- External LLM calls may require Snowflake-approved external network access, secrets, or a Snowflake-native AI service.
- Local file state such as `sessions/`, `chats/`, YAML memory files, and `.env` secrets should not be assumed to work unchanged in Snowflake.

The project should validate dependency compatibility for packages such as `chainladder`, `pandas`, `scikit-learn`, `numba`, `sparse`, `plotly`, and any AI/networking dependencies before committing to a full migration.

## Local-First Strategy

The Streamlit version can and should be developed locally first without Snowflake involvement.

The local prototype can use:

- Local CSV, parquet, or sample data.
- Local YAML/session files.
- `.env` for local API keys.
- `streamlit run` for local development.
- The existing reserving and AI backend logic where possible.

This allows the project to be prototyped, tested, and published to GitHub before adapting it for work deployment in Snowflake.

Recommended local command shape:

```bash
uv pip install streamlit
uv run streamlit run streamlit_app.py
```

The prototype should be designed so Snowflake is a replaceable deployment and data backend, not hard-coded into the UI.

## Architecture Recommendation

The frontend should be separated from the core reserving and AI logic.

Recommended structure:

```text
Streamlit UI
  -> reserving and AI services
  -> data repository interface
      -> local CSV/YAML implementation for GitHub prototype
      -> Snowflake implementation for work deployment
```

Avoid putting Snowflake SQL directly inside page rendering code. Prefer repository or service calls such as:

```python
claims = data_repository.load_claims(segment_id)
premium = data_repository.load_premium(segment_id)
```

This makes it possible to run locally with file-backed repositories and later deploy in Snowflake with Snowflake-backed repositories.

The same principle applies to AI:

- Use a local AI provider or `.env` configuration during prototyping.
- Later replace or adapt the AI provider for Snowflake external access, company-hosted APIs, or Snowflake-native AI services.
- Keep AI calls behind an `AIChatService` or equivalent interface.

## Suggested Migration Path

Do not attempt a big-bang Dash-to-Streamlit rewrite.

Recommended sequence:

1. Build a local Streamlit prototype with one segment and baseline reserving results.
2. Add triangle/results display using the existing reserving backend.
3. Add the AI chat window using Streamlit chat primitives.
4. Add sidebar context for Analysis Basis, evidence trace, scenario ledger, and proposal status.
5. Add one high-value workflow, such as drop review and scenario recalculation.
6. Add accept/reject proposal handling.
7. Move persistence behind repository interfaces.
8. Create Snowflake-backed repository implementations.
9. Deploy a small proof of concept to Streamlit in Snowflake.
10. Validate dependencies, performance, RBAC, external AI access, and user feedback.
11. Migrate additional workflows only after the proof of concept works well.

## Product Direction

The Streamlit app should probably not duplicate every Dash screen. It should be redesigned around business workflows:

- Load segment.
- Review data.
- Run baseline reserving.
- Ask AI.
- Test recommended scenario.
- Accept or reject proposal.
- Review evidence.
- Export or sign off.

This may be more user-friendly and business-aligned than a dense actuarial cockpit.

## Decision Summary

If the objective is a polished, highly interactive local analytical workbench, Dash remains strong.

If the objective is business adoption in a Snowflake-centered environment, Streamlit is the stronger strategic frontend choice.

The best path is to make Streamlit the future-facing business frontend while preserving the existing reserving engine, scenario logic, diagnostics, AI services, and control-plane concepts as reusable backend layers.

Dash can remain temporarily as a reference implementation or advanced workbench during migration, but new business-facing development should prioritize Streamlit if Snowflake deployment and user access are central to the project's success.
