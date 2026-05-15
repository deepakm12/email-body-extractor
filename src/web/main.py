"""UI for the Email Body Extraction Platform."""

import json
import os

import requests
import streamlit as st

_API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:3000").rstrip("/")
API_URL = f"{_API_BASE_URL}/api/v1/extract"
STREAM_URL = f"{_API_BASE_URL}/api/v1/extract/stream"

_ALL_PROVIDER_LABELS: dict[str | None, str] = {
    None: "Default",
    "openai": "OpenAI",
    "azure_openai": "Azure OpenAI",
    "gemini": "Google Gemini",
    "anthropic": "Anthropic",
}


def _fetch_configured_providers() -> list[str]:
    """Return provider names that are configured, or all names if API is unreachable."""
    try:
        resp = requests.get(f"{_API_BASE_URL}/api/v1/providers", timeout=3)
        if resp.status_code == 200:
            return [p["name"] for p in resp.json().get("providers", []) if p.get("configured")]
    except Exception:
        pass
    return list(p for p in _ALL_PROVIDER_LABELS if p is not None)


# Page configuration
st.set_page_config(page_title="Email Body Extractor", page_icon="📧", layout="wide")

st.markdown(
    """
    <style>
        .confidence-high { color: #28a745; font-weight: bold; }
        .confidence-medium { color: #ffc107; font-weight: bold; }
        .confidence-low { color: #dc3545; font-weight: bold; }
        .flow-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 500;
        }
        .flow-non-llm { background-color: #e8f5e9; color: #2e7d32; }
        .flow-llm { background-color: #e3f2fd; color: #1565c0; }
        .agent-card {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            background: #fafafa;
        }
        .agent-success { border-left: 4px solid #4caf50; }
        .agent-fail { border-left: 4px solid #f44336; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_confidence_badge(confidence: float) -> str:
    """Render a confidence badge with color coding."""
    if confidence >= 0.85:
        css_class = "confidence-high"
        label = "High"
    elif confidence >= 0.6:
        css_class = "confidence-medium"
        label = "Medium"
    else:
        css_class = "confidence-low"
        label = "Low"

    return f'<span class="{css_class}">{label} ({confidence:.0%})</span>'


def render_flow_badge(flow_used: str) -> str:
    """Render a flow badge."""
    if "non_llm" in flow_used.lower():
        css_class = "flow-badge flow-non-llm"
        label = "Deterministic"
    else:
        css_class = "flow-badge flow-llm"
        label = "LLM Agent"

    return f'<span class="{css_class}">{label}</span>'


def render_agent_trace(trace: list[dict[str, object]]) -> None:
    """Render agent execution trace."""
    if not trace:
        st.info("No agent trace available for non-LLM extraction.")
        return

    st.subheader("Agent Execution Trace")

    for i, step in enumerate(trace):
        agent_name = step.get("agent", "unknown")
        success = step.get("success", False)
        css_class = "agent-success" if success else "agent-fail"

        with st.container():
            st.markdown(
                f"""
                <div class="agent-card {css_class}">
                    <strong>Step {i+1}: {str(agent_name).title()} Agent</strong>
                    {"✅ Success" if success else "❌ Failed"}<br/>
                    <small>{step.get("output_preview", "")}</small>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Details"):
                st.json(step)


def main() -> None:
    """Main UI application."""

    # Header
    st.title("Email Body Extractor")
    st.markdown("""Extracts the **latest meaningful email message** using deterministic and agentic AI workflows.""")

    # Sidebar configuration
    with st.sidebar:
        mode = st.selectbox(
            "Extraction Mode",
            options=["auto", "non_llm", "llm"],
            format_func=lambda x: {
                "auto": "Auto (Recommended)",
                "non_llm": "Non-LLM",
                "llm": "LLM",
            }[x],
            accept_new_options=False,
            help="Auto: Uses non-LLM first, falls back to LLM if confidence is low",
        )
        _configured = _fetch_configured_providers()
        _provider_options: list[str | None] = [None] + _configured
        provider = st.selectbox(
            "LLM Provider",
            options=_provider_options,
            format_func=lambda x: _ALL_PROVIDER_LABELS.get(x, str(x)),
            accept_new_options=False,
            disabled=(mode == "non_llm"),
            help="Only providers with a configured API key are shown.",
        )
        use_streaming = st.toggle(
            "Stream response",
            value=False,
            disabled=(mode in ("non_llm", "auto")),
            help="Stream LLM responses",
        )

        st.divider()
        st.header("History")
        history_entries = []
        try:
            history_resp = requests.get(f"{_API_BASE_URL}/api/v1/history", timeout=3)
            if history_resp.status_code == 200:
                history_entries = history_resp.json()
        except Exception:
            pass

        if history_entries and isinstance(history_entries, list):
            for entry in history_entries[:10]:
                if not isinstance(entry, dict):
                    continue
                confidence = entry.get("confidence", 0)
                with st.expander(f"{entry.get('content_preview','')[:40]}…"):
                    st.write(f"**Flow:** {entry.get('flow_used')}")
                    st.write(f"**Confidence:** {confidence:.0%}")
                    st.code(entry.get("latest_message", ""), language=None)
            if st.button("Clear history"):
                try:
                    requests.delete(f"{_API_BASE_URL}/api/v1/history", timeout=3)
                    st.rerun()
                except Exception:
                    st.error("Could not clear history")
        else:
            st.caption("No history yet.")

    # Input section
    st.header("Input")

    input_tab1, input_tab2 = st.tabs(["Paste Content", "Upload .eml File"])
    content = ""
    is_eml = False

    with input_tab1:
        content = st.text_area(
            "Email Content (HTML or Plain Text)",
            height=250,
            placeholder="Paste your email content here...",
        )

    with input_tab2:
        uploaded_file = st.file_uploader("Choose an .eml file", type=["eml"])
        if uploaded_file is not None:
            raw_bytes = uploaded_file.read()
            content = raw_bytes.decode("utf-8", errors="replace")
            is_eml = True
            st.info(f"Uploaded: {uploaded_file.name}")

    # Extract button
    col1, col2 = st.columns([1, 5])
    with col1:
        extract_clicked = st.button("Extract", type="primary", use_container_width=True)

    if extract_clicked and content:
        payload: dict[str, object] = {"content": content, "mode": mode, "is_eml": is_eml}
        if provider:
            payload["provider"] = provider

        if use_streaming and mode == "llm":
            with st.spinner("Connecting..."):
                try:
                    stream_resp = requests.post(STREAM_URL, json=payload, stream=True, timeout=120)
                    stream_resp.raise_for_status()
                    st.header("Result")
                    output_placeholder = st.empty()
                    status_placeholder = st.empty()
                    full_text = ""
                    final_result: dict[str, object] = {}
                    for raw_line in stream_resp.iter_lines():
                        if not raw_line:
                            continue
                        line = raw_line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        event = json.loads(line[6:])
                        if event["type"] == "start":
                            status_placeholder.info(f"Running {event['agent']} agent...")
                        elif event["type"] == "token":
                            full_text += event["text"]
                            output_placeholder.code(full_text, language=None)
                        elif event["type"] == "agent_done":
                            status_placeholder.success(f"{event['agent'].title()} agent done")
                        elif event["type"] == "done":
                            final_result = event.get("result", {})
                            status_placeholder.empty()
                        elif event["type"] == "error":
                            st.error(f"Stream error: {event['message']}")
                            break
                    if final_result:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Confidence", f"{final_result.get('confidence', 0):.0%}")
                        with col2:
                            st.metric("Flow", str(final_result.get("flow_used", "")))
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server.")
                except Exception as e:
                    st.error(f"Stream error: {e}")
        else:
            with st.spinner("Processing..."):
                try:
                    response = requests.post(API_URL, json=payload, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    if result.get("success") and result.get("data"):
                        data = result["data"]
                        st.header("Result")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            confidence = data.get("confidence", 0)
                            st.metric(label="Confidence", value=f"{confidence:.0%}")
                        with col2:
                            st.markdown(
                                f"**Flow Used**<br/>{render_flow_badge(data.get('flow_used', 'unknown'))}",
                                unsafe_allow_html=True,
                            )
                        with col3:
                            latest = data.get("latest_message", "")
                            st.metric(label="Characters", value=len(latest))
                        st.progress(min(1.0, max(0.0, confidence)))
                        st.subheader("Extracted Message")
                        st.code(data.get("latest_message", ""), language=None)
                        st_copy = st.button("Copy to Clipboard")
                        if st_copy:
                            st.write("Copied!")
                        agent_trace = result.get("agent_trace")
                        if agent_trace or data.get("flow_used", "").startswith("llm"):
                            with st.expander("Agent Execution Trace", expanded=False):
                                render_agent_trace(agent_trace or [])
                        with st.expander("Metadata", expanded=False):
                            st.json(data.get("metadata", {}))
                    else:
                        st.error(f"Extraction failed: {result.get('error', 'Unknown error')}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server.")
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The LLM may be taking too long.")
                except Exception as e:
                    st.error(f"Error: {e}")
    elif extract_clicked and not content:
        st.warning("Please provide email content to extract.")
    st.divider()
    st.caption("Email Body Extraction Platform")


if __name__ == "__main__":
    main()
