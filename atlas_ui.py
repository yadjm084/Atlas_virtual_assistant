"""Gradio UI builder for Atlas."""

from __future__ import annotations

import json

import gradio as gr

from atlas_state import (
    build_dorm_state_html,
    build_pipeline_timeline_html,
    build_stage_hint_html,
    build_status_overview_html,
)


ATLAS_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.sky,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
)


ATLAS_CSS = """
:root {
  --atlas-bg: linear-gradient(180deg, #f5f8fc 0%, #eef4fb 100%);
  --atlas-panel: #ffffff;
  --atlas-subpanel: #f7f9fc;
  --atlas-soft: #f4f8fc;
  --atlas-soft-2: #eef4fb;
  --atlas-border: #d8e2ee;
  --atlas-shadow: 0 18px 38px rgba(16, 34, 53, 0.08);
  --atlas-text: #16273a;
  --atlas-muted: #66788c;
  --atlas-accent: #0f6fff;
  --atlas-accent-soft: rgba(15, 111, 255, 0.08);
  --atlas-hero: linear-gradient(135deg, #10243a 0%, #173a5a 54%, #0b6a84 100%);
  --atlas-hero-text: #f5f9ff;
}

body, .gradio-container {
  background: var(--atlas-bg);
}

.gradio-container {
  color: var(--atlas-text);
}

.gradio-container ::selection {
  background: #0f6fff !important;
  color: #ffffff !important;
  -webkit-text-fill-color: #ffffff !important;
}

.gradio-container ::-moz-selection {
  background: #0f6fff !important;
  color: #ffffff !important;
}

#atlas-shell {
  max-width: 1500px;
  margin: 0 auto;
  padding-bottom: 24px;
}

.atlas-hero {
  max-width: 1500px;
  margin: 0 auto 24px auto;
  padding: 30px 34px;
  border-radius: 28px;
  background: var(--atlas-hero);
  box-shadow: var(--atlas-shadow);
}

.atlas-kicker {
  display: inline-flex;
  align-items: center;
  padding: 7px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #b8d8ff;
  background: rgba(255, 255, 255, 0.10);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.atlas-hero h1 {
  margin: 14px 0 12px 0;
  font-size: 52px;
  line-height: 0.96;
  font-weight: 800;
  letter-spacing: -0.04em;
  color: var(--atlas-hero-text);
}

.atlas-hero p {
  margin: 0 0 16px 0;
  max-width: 920px;
  font-size: 16px;
  line-height: 1.7;
  color: rgba(245, 249, 255, 0.82);
}

.atlas-pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}

.atlas-pill {
  display: inline-flex;
  align-items: center;
  padding: 9px 14px;
  border-radius: 999px;
  font-size: 13px;
  font-weight: 600;
  color: var(--atlas-hero-text);
  background: rgba(255, 255, 255, 0.12);
  border: 1px solid rgba(255, 255, 255, 0.16);
}

.atlas-layout {
  display: grid;
  grid-template-columns: minmax(270px, 310px) minmax(0, 1fr);
  gap: 18px;
}

.atlas-sidebar,
.atlas-main {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.atlas-card {
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
  border: 1px solid var(--atlas-border);
  border-radius: 22px;
  box-shadow: var(--atlas-shadow);
  padding: 12px;
  --block-background-fill: transparent;
  --block-border-color: transparent;
  --panel-background-fill: transparent;
  --input-background-fill: var(--atlas-subpanel);
  --background-fill-primary: #ffffff;
  --background-fill-secondary: var(--atlas-subpanel);
  --body-background-fill: transparent;
  --body-text-color: var(--atlas-text);
}

.atlas-card > .gr-group,
.atlas-card > .gr-block,
.atlas-card > .gradio-group,
.atlas-card > .gradio-column,
.atlas-card .gr-form,
.atlas-card .form,
.atlas-card .wrap,
.atlas-card > div,
.atlas-card section,
.atlas-card article,
.atlas-card [class*="gradio"],
.atlas-card [class*="container"],
.atlas-card [class*="wrap"],
.atlas-card [class*="panel"],
.atlas-card .gr-row,
.atlas-card .gr-column,
.atlas-card .gradio-row,
.atlas-card .gradio-column,
.atlas-card fieldset,
.atlas-card [data-testid="block"],
.atlas-card [data-testid="textbox"],
.atlas-card [data-testid="accordion"],
.atlas-card [data-testid="audio"],
.atlas-stage-panel > .gr-group,
.atlas-stage-panel > .gr-block,
.atlas-stage-panel > .gradio-group,
.atlas-stage-panel > .gradio-column,
.atlas-stage-panel > div,
.atlas-stage-panel section,
.atlas-stage-panel article,
.atlas-stage-panel [class*="gradio"],
.atlas-stage-panel [class*="container"],
.atlas-stage-panel [class*="wrap"],
.atlas-stage-panel [class*="panel"],
.atlas-stage-panel .gr-row,
.atlas-stage-panel .gr-column,
.atlas-stage-panel .gradio-row,
.atlas-stage-panel .gradio-column,
.atlas-stage-panel fieldset,
.atlas-stage-panel [data-testid="block"],
.atlas-stage-panel [data-testid="column"],
.atlas-stage-panel [data-testid="row"],
.atlas-stage-panel [data-testid="textbox"],
.atlas-stage-panel [data-testid="accordion"],
.atlas-stage-panel [data-testid="audio"] {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

.atlas-card h2 {
  margin: 0 0 8px 0;
  font-size: 24px;
  letter-spacing: -0.03em;
  color: var(--atlas-text);
}

.atlas-card h3 {
  margin: 0 0 8px 0;
  font-size: 20px;
  letter-spacing: -0.02em;
  color: var(--atlas-text);
}

.atlas-section-copy {
  margin-top: -2px;
  margin-bottom: 10px;
  color: var(--atlas-muted);
  font-size: 13px;
  line-height: 1.55;
}

.atlas-card label,
.atlas-card .prose,
.atlas-card .gr-markdown,
.atlas-card .gr-markdown p,
.atlas-card .gr-markdown span,
.atlas-card .gr-markdown strong {
  color: var(--atlas-text) !important;
}

.atlas-card .gr-markdown h1,
.atlas-card .gr-markdown h2,
.atlas-card .gr-markdown h3,
.atlas-card .gr-markdown h4 {
  color: var(--atlas-text) !important;
}

.atlas-card code,
.atlas-card kbd,
.atlas-card pre code,
.atlas-stage-panel code,
.atlas-stage-panel kbd,
.atlas-stage-panel pre code {
  background: #eaf2ff !important;
  color: #173a66 !important;
  border: 1px solid #c6daf7;
  border-radius: 8px;
  padding: 0.08rem 0.35rem;
  font-weight: 700;
}

.atlas-card .gr-markdown,
.atlas-card .gr-markdown *,
.atlas-stage-panel .gr-markdown,
.atlas-stage-panel .gr-markdown * {
  background: transparent !important;
}

.atlas-stage-panel,
.atlas-stage-panel *:not(button):not(svg):not(path):not(audio) {
  color: var(--atlas-text) !important;
}

.atlas-code textarea,
.atlas-status textarea,
.atlas-response textarea,
.atlas-field textarea,
.atlas-field input {
  background: var(--atlas-subpanel) !important;
  color: var(--atlas-text) !important;
  border: 1px solid var(--atlas-border) !important;
  box-shadow: none !important;
}

.atlas-field,
.atlas-field > div,
.atlas-field [data-testid="textbox"],
.atlas-field [data-testid="textbox"] > div,
.atlas-field [data-testid="textbox"] > div > div {
  background: var(--atlas-subpanel) !important;
  border: none !important;
  box-shadow: none !important;
}

.atlas-code textarea,
.atlas-status textarea {
  font-family: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace !important;
  font-size: 13px !important;
}

.atlas-response textarea {
  font-size: 15px !important;
  line-height: 1.55 !important;
}

.atlas-status textarea {
  min-height: 158px !important;
}

.atlas-compact textarea {
  min-height: 52px !important;
}

.atlas-card input {
  background: var(--atlas-subpanel) !important;
  color: var(--atlas-text) !important;
  border: 1px solid var(--atlas-border) !important;
}

.atlas-audio,
.atlas-audio > div,
.atlas-audio .gradio-audio,
.atlas-audio .gradio-audio .wrap,
.atlas-audio .gradio-audio .container,
.atlas-audio [class*="Audio"],
.atlas-audio [class*="audio"],
.atlas-audio [data-testid="audio"],
.atlas-audio [data-testid="audio"] > div,
.atlas-audio [data-testid="block"] {
  background: #fbfdff !important;
  border-radius: 18px !important;
  border: 1px dashed #bfd2e8 !important;
  box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.8) !important;
}

.atlas-audio [class*="audio"] {
  min-height: 220px !important;
}

.atlas-audio button,
.atlas-audio [role="button"],
.atlas-audio select,
.atlas-audio option,
.atlas-audio svg,
.atlas-audio label,
.atlas-audio span,
.atlas-audio p,
.atlas-audio div {
  color: var(--atlas-text) !important;
  fill: var(--atlas-text) !important;
}

.atlas-audio button,
.atlas-audio [role="button"],
.atlas-audio select {
  background: white !important;
  border: 1px solid var(--atlas-border) !important;
  box-shadow: none !important;
}

.atlas-card button {
  border-radius: 14px !important;
  font-weight: 700 !important;
}

.atlas-card button.primary,
.atlas-card .gr-button-primary {
  background: linear-gradient(135deg, #0f6fff 0%, #0c9ec6 100%) !important;
  color: white !important;
  border: none !important;
  box-shadow: 0 12px 24px rgba(15, 111, 255, 0.22);
}

.atlas-card button.secondary {
  background: #edf3f9 !important;
  color: var(--atlas-text) !important;
  border: 1px solid var(--atlas-border) !important;
}

.atlas-card button:not(.primary):not(.gr-button-primary) {
  background: #edf3f9 !important;
  color: var(--atlas-text) !important;
  border: 1px solid var(--atlas-border) !important;
  box-shadow: none !important;
}

.atlas-card button span,
.atlas-card button div,
.atlas-card button p {
  background: transparent !important;
  color: inherit !important;
}

.atlas-card .gr-accordion,
.atlas-card .gradio-accordion {
  border-radius: 16px !important;
  background: linear-gradient(180deg, var(--atlas-soft) 0%, var(--atlas-soft-2) 100%) !important;
  border: 1px solid var(--atlas-border) !important;
}

.atlas-card .gr-accordion summary,
.atlas-card .gradio-accordion summary {
  color: var(--atlas-text) !important;
}

.atlas-card .gr-accordion summary,
.atlas-card .gradio-accordion summary,
.atlas-card .gr-accordion details,
.atlas-card .gradio-accordion details,
.atlas-card .gr-accordion [data-testid="accordion"],
.atlas-card .gradio-accordion [data-testid="accordion"] {
  background: transparent !important;
}

.atlas-divider {
  height: 1px;
  margin: 12px 0 4px 0;
  background: linear-gradient(90deg, transparent, rgba(108, 134, 163, 0.24), transparent);
}

.atlas-card .gr-block-label,
.atlas-card .block-label,
.atlas-card .label-wrap > span,
.atlas-card .label-wrap > label {
  background: transparent !important;
  color: var(--atlas-muted) !important;
  border: none !important;
  padding: 0 !important;
  box-shadow: none !important;
  font-weight: 700 !important;
  letter-spacing: 0 !important;
}

.atlas-status-summary,
.atlas-hint-card,
.atlas-dorm-grid,
.atlas-flow-card,
.atlas-progress-card {
  background: linear-gradient(180deg, var(--atlas-soft) 0%, var(--atlas-soft-2) 100%);
  border: 1px solid var(--atlas-border);
  border-radius: 18px;
  padding: 16px;
}

.atlas-status-badges {
  display: flex;
  gap: 8px;
  margin-bottom: 14px;
}

.atlas-status-badge {
  padding: 7px 11px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  color: var(--atlas-accent);
  background: var(--atlas-accent-soft);
}

.atlas-status-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.atlas-status-card {
  background: linear-gradient(180deg, #f9fbff 0%, #f4f8fc 100%) !important;
}

.atlas-metric {
  background: white;
  border: 1px solid var(--atlas-border);
  border-radius: 14px;
  padding: 12px;
}

.atlas-metric-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--atlas-muted);
  margin-bottom: 6px;
}

.atlas-metric-value {
  font-size: 16px;
  font-weight: 700;
  color: var(--atlas-text);
}

.atlas-progress-card .atlas-timeline {
  display: grid;
  grid-template-columns: 1fr;
  gap: 10px;
  margin: 0;
}

.atlas-stage {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 12px 14px;
  border-radius: 18px;
  border: 1px solid var(--atlas-border);
  background: white;
  min-height: 76px;
}

.atlas-stage.completed {
  border-color: rgba(15, 111, 255, 0.22);
  background: linear-gradient(180deg, #f3f9ff 0%, #eef7ff 100%);
}

.atlas-stage.active {
  border-color: rgba(15, 111, 255, 0.42);
  background: linear-gradient(180deg, #f5fbff 0%, #eaf5ff 100%);
  box-shadow: 0 12px 24px rgba(15, 111, 255, 0.12);
}

.atlas-stage.locked {
  opacity: 0.7;
}

.atlas-stage-index {
  flex: 0 0 34px;
  width: 34px;
  height: 34px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 800;
  color: var(--atlas-text);
  background: #eef3f8;
}

.atlas-stage.active .atlas-stage-index,
.atlas-stage.completed .atlas-stage-index {
  color: white;
  background: linear-gradient(135deg, #0f6fff 0%, #0c9ec6 100%);
}

.atlas-stage-title {
  font-size: 13px;
  font-weight: 700;
  line-height: 1.25;
}

.atlas-stage-meta {
  margin-top: 2px;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--atlas-muted);
}

.atlas-hint-kicker {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--atlas-accent);
  font-weight: 700;
  margin-bottom: 8px;
}

.atlas-hint-stepno {
  font-size: 12px;
  font-weight: 700;
  color: var(--atlas-accent);
  margin-bottom: 10px;
}

.atlas-hint-card h3 {
  font-size: 28px;
  line-height: 1.05;
  letter-spacing: -0.03em;
  margin: 0 0 10px 0;
}

.atlas-hint-card p {
  margin: 0;
  color: var(--atlas-muted);
  font-size: 15px;
  line-height: 1.65;
}

.atlas-hint-prompt {
  margin-top: 14px;
  padding: 12px 14px;
  border-radius: 14px;
  background: white;
  border: 1px solid var(--atlas-border);
  font-size: 14px;
  color: var(--atlas-text);
  font-weight: 600;
}

.atlas-hint-next {
  margin-top: 12px;
  font-size: 12px;
  color: var(--atlas-muted);
  font-weight: 700;
}

.atlas-stage-shell {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.atlas-stage-panel {
  background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
  border: 1px solid var(--atlas-border);
  border-radius: 22px;
  box-shadow: var(--atlas-shadow);
  padding: 18px;
}

.atlas-stage-shell .atlas-stage-panel h2 {
  margin-bottom: 8px;
}

.atlas-stage-panel-primary {
  border-color: #b9d4f4;
  box-shadow: 0 24px 44px rgba(16, 34, 53, 0.10);
}

.atlas-stage-panel-primary::before {
  content: "";
  display: block;
  height: 4px;
  border-radius: 999px;
  background: linear-gradient(135deg, #0f6fff 0%, #0c9ec6 100%);
  margin: -4px 0 14px 0;
}

.atlas-section-title {
  margin: 10px 0 6px 0;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--atlas-muted);
  font-weight: 700;
}

.atlas-examples-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 12px;
}

.atlas-example-note {
  margin-top: 10px;
  font-size: 12px;
  color: var(--atlas-muted);
}

.atlas-step-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin: 12px 0 16px 0;
}

.atlas-meta-card {
  background: linear-gradient(180deg, #f4f8fc 0%, #eef4fb 100%);
  border: 1px solid var(--atlas-border);
  border-radius: 14px;
  padding: 12px;
}

.atlas-meta-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--atlas-muted);
  margin-bottom: 6px;
}

.atlas-meta-value {
  font-size: 13px;
  font-weight: 700;
  color: var(--atlas-text);
  line-height: 1.45;
}

.atlas-dorm-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.atlas-dorm-card {
  background: white;
  border: 1px solid var(--atlas-border);
  border-radius: 14px;
  padding: 14px;
}

.atlas-dorm-hero {
  background: linear-gradient(135deg, #f3f8ff 0%, #eef7ff 100%);
}

.atlas-dorm-wide {
  grid-column: span 2;
}

.atlas-dorm-label {
  font-size: 11px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--atlas-muted);
  margin-bottom: 8px;
}

.atlas-dorm-value {
  font-size: 22px;
  font-weight: 800;
  color: var(--atlas-text);
  letter-spacing: -0.02em;
}

.atlas-dorm-subvalue {
  margin-top: 8px;
  font-size: 12px;
  color: var(--atlas-muted);
  font-weight: 600;
}

.atlas-progress-track {
  height: 10px;
  border-radius: 999px;
  background: #e8eef5;
  overflow: hidden;
}

.atlas-progress-bar {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(135deg, #0f6fff 0%, #0c9ec6 100%);
}

.atlas-progress-label {
  margin-top: 8px;
  font-size: 13px;
  font-weight: 700;
  color: var(--atlas-text);
}

@media (max-width: 1100px) {
  .atlas-hero h1 {
    font-size: 40px;
  }

  .atlas-layout {
    grid-template-columns: 1fr;
  }

  .atlas-status-grid,
  .atlas-dorm-grid,
  .atlas-examples-grid,
  .atlas-step-meta {
    grid-template-columns: 1fr;
  }

  .atlas-dorm-wide {
    grid-column: span 1;
  }
}
"""


def build_demo(actions: AtlasActions):
    initial_state = actions.init_state()
    initial_status = actions.get_status_text(initial_state)
    initial_ready = actions.get_ready_time_left(initial_state)

    with gr.Blocks(title="Atlas - Virtual Assistant") as demo:
        gr.HTML(
            """
            <section class="atlas-hero">
              
              <h1>Atlas Virtual Assistant</h1>
              <p>
                A voice assistant for movie information and smart dorm control. The demo pipeline covers
                user verification, wake word detection, Whisper ASR, joint intent and slot prediction,
                fulfillment, answer generation, and text to speech.
              </p>
              <div class="atlas-pill-row">
                <span class="atlas-pill">Voice Verification</span>
                <span class="atlas-pill">Wake Word</span>
                <span class="atlas-pill">Whisper ASR</span>
                <span class="atlas-pill">Joint Intent + Slots</span>
                <span class="atlas-pill">Weather + Movies + Dorm Control</span>
                <span class="atlas-pill">TTS Audio Output</span>
              </div>
            </section>
            """
        )

        state = gr.State(initial_state)

        with gr.Row(elem_id="atlas-shell", elem_classes=["atlas-layout"]):
            with gr.Column(elem_classes=["atlas-sidebar"]):
                with gr.Group(elem_classes=["atlas-card", "atlas-progress-card"]):
                    gr.Markdown("## Demo Progress")
                    gr.Markdown(
                        "The live architecture follows the same sequence as the course pipeline.",
                        elem_classes=["atlas-section-copy"],
                    )
                    pipeline_timeline = gr.HTML(build_pipeline_timeline_html(initial_state))

                with gr.Group(elem_classes=["atlas-card", "atlas-status-card"]):
                    gr.Markdown("## System State")
                    gr.Markdown(
                        "A live summary of Atlas while you move through the guided demo.",
                        elem_classes=["atlas-section-copy"],
                    )
                    assistant_status = gr.HTML(build_status_overview_html(initial_state, initial_ready))
                    gr.Markdown("Ready countdown", elem_classes=["atlas-section-title"])
                    ready_countdown_box = gr.Textbox(
                        label="Ready Countdown",
                        value="Sleeping",
                        lines=1,
                        elem_classes=["atlas-field", "atlas-compact"],
                        show_label=False,
                    )
                    with gr.Accordion("Technical Status Details", open=False):
                        assistant_status_raw = gr.Textbox(
                            label="Raw Status",
                            value=initial_status,
                            lines=7,
                            elem_classes=["atlas-status"],
                        )

                with gr.Column(visible=False) as dorm_panel:
                    with gr.Group(elem_classes=["atlas-card"]):
                        gr.Markdown("## Dorm State")
                        gr.Markdown(
                            "The smart dorm status updates after Atlas fulfills a control intent.",
                            elem_classes=["atlas-section-copy"],
                        )
                        dorm_visual = gr.HTML(build_dorm_state_html(initial_state))
                        with gr.Accordion("Raw Control State", open=False):
                            control_box = gr.Textbox(
                                label="Control System State",
                                value=json.dumps(initial_state["control_state"], indent=2),
                                lines=12,
                                elem_classes=["atlas-code"],
                            )

            with gr.Column(elem_classes=["atlas-main"]):
                stage_hint_html = gr.HTML(build_stage_hint_html(initial_state))

                with gr.Column(elem_classes=["atlas-stage-shell"]):
                    with gr.Column(visible=True) as verify_panel:
                        with gr.Group(elem_classes=["atlas-stage-panel", "atlas-stage-panel-primary"]):
                            gr.Markdown("## Step 1. User Verification")
                            gr.Markdown(
                                "Confirm the speaker identity first. Atlas stays locked until verification succeeds.",
                                elem_classes=["atlas-section-copy"],
                            )
                            gr.HTML(
                                f"""
                                <section class="atlas-step-meta">
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Enrollment</div>
                                    <div class="atlas-meta-value">{actions.runtime.profile_load_status}</div>
                                  </div>
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Threshold</div>
                                    <div class="atlas-meta-value">{actions.runtime.chosen_threshold}</div>
                                  </div>
                                </section>
                                """
                            )
                            gr.Markdown("Voice sample", elem_classes=["atlas-section-title"])
                            verification_audio_input = gr.Audio(type="filepath", label="Verification Audio Input", show_label=False, elem_classes=["atlas-audio"])
                            gr.Markdown("Verification result", elem_classes=["atlas-section-title"])
                            verify_output = gr.Textbox(label="Verification Output", lines=3, show_label=False, elem_classes=["atlas-field", "atlas-response"])
                            with gr.Row():
                                btn_verify = gr.Button("Verify Voice", variant="primary")
                                btn_reset_verification = gr.Button("Reset Verification")
                            with gr.Accordion("Verification Details", open=False):
                                verification_scores_output = gr.Textbox(
                                    label="Verification Scores",
                                    lines=8,
                                    value="{}",
                                    elem_classes=["atlas-code"],
                                )
                            gr.HTML('<div class="atlas-divider"></div>')
                            gr.Markdown("### Quick Verification Bypass")
                            gr.Markdown("Allowed codes: `Adjmal`, `Nair`, `Sharma`", elem_classes=["atlas-section-copy"])
                            with gr.Row():
                                verification_code_input = gr.Textbox(
                                    label="Verification Code",
                                    placeholder="Enter Adjmal, Nair, or Sharma",
                                    show_label=False,
                                    elem_classes=["atlas-field"],
                                )
                                btn_skip_verify = gr.Button("Skip Verification with Code")

                    with gr.Column(visible=False) as wake_panel:
                        with gr.Group(elem_classes=["atlas-stage-panel", "atlas-stage-panel-primary"]):
                            gr.Markdown("## Step 2. Wake Word")
                            gr.Markdown(
                                "Wake Atlas so it enters command mode. This stage only appears after user verification succeeds.",
                                elem_classes=["atlas-section-copy"],
                            )
                            gr.HTML(
                                f"""
                                <section class="atlas-step-meta">
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Wake Model</div>
                                    <div class="atlas-meta-value">{actions.runtime.wake_model_status}</div>
                                  </div>
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Threshold</div>
                                    <div class="atlas-meta-value">{actions.runtime.wake_threshold}</div>
                                  </div>
                                </section>
                                """
                            )
                            gr.Markdown("Wake word sample", elem_classes=["atlas-section-title"])
                            wake_audio_input = gr.Audio(type="filepath", label="Wake Word Audio Input", show_label=False, elem_classes=["atlas-audio"])
                            gr.Markdown("Wake result", elem_classes=["atlas-section-title"])
                            wake_output = gr.Textbox(label="Wake Word Output", lines=2, show_label=False, elem_classes=["atlas-field", "atlas-response"])
                            with gr.Row():
                                btn_wake = gr.Button("Detect Wake Word", variant="primary")
                                btn_reset_wake = gr.Button("Reset Wake Word")
                            gr.HTML('<div class="atlas-divider"></div>')
                            gr.Markdown("### Quick Wake Bypass")
                            gr.Markdown("Allowed code: `Hey Atlas`", elem_classes=["atlas-section-copy"])
                            with gr.Row():
                                wake_code_input = gr.Textbox(
                                    label="Wake Word Code",
                                    placeholder="Enter Hey Atlas",
                                    show_label=False,
                                    elem_classes=["atlas-field"],
                                )
                                btn_skip_wake = gr.Button("Skip Wake Word with Code")

                    with gr.Column(visible=False) as asr_panel:
                        with gr.Group(elem_classes=["atlas-stage-panel", "atlas-stage-panel-primary"]):
                            gr.Markdown("## Step 3. Speech Recognition")
                            gr.Markdown(
                                "Capture the command that Atlas should interpret. Use recorded audio or type one of the examples below.",
                                elem_classes=["atlas-section-copy"],
                            )
                            gr.HTML(
                                f"""
                                <section class="atlas-step-meta">
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">ASR Model</div>
                                    <div class="atlas-meta-value">{actions.runtime.asr_model_status}</div>
                                  </div>
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Tip</div>
                                    <div class="atlas-meta-value">Typed input is the fastest route for the live demo.</div>
                                  </div>
                                </section>
                                """
                            )
                            gr.Markdown("Speak a command", elem_classes=["atlas-section-title"])
                            command_audio_input = gr.Audio(type="filepath", label="Command Audio Input", show_label=False, elem_classes=["atlas-audio"])
                            gr.Markdown("Transcript", elem_classes=["atlas-section-title"])
                            transcript_box = gr.Textbox(label="Transcript", lines=3, elem_classes=["atlas-field", "atlas-response"], show_label=False)
                            gr.Markdown("Or type a command", elem_classes=["atlas-section-title"])
                            typed_transcript_input = gr.Textbox(
                                label="Typed Sentence",
                                placeholder="Type a sentence here to bypass speech recognition",
                                lines=2,
                                show_label=False,
                                elem_classes=["atlas-field"],
                            )
                            with gr.Row():
                                btn_asr = gr.Button("Run ASR", variant="primary")
                                btn_use_typed_transcript = gr.Button("Use Typed Sentence")
                            gr.Markdown("### Quick Demo Commands", elem_classes=["atlas-section-copy"])
                            with gr.Row(elem_classes=["atlas-examples-grid"]):
                                example_weather = gr.Button("Weather Example")
                                example_movie = gr.Button("Movie Example")
                                example_dorm = gr.Button("Dorm Example")
                                example_oos = gr.Button("Out-of-Scope Example")
                            gr.Markdown(
                                "Use the example buttons to fill the typed command box quickly during the demo.",
                                elem_classes=["atlas-example-note"],
                            )

                    with gr.Column(visible=False) as intent_panel:
                        with gr.Group(elem_classes=["atlas-stage-panel", "atlas-stage-panel-primary"]):
                            gr.Markdown("## Step 4. Intent Detection")
                            gr.Markdown(
                                "Atlas now maps the transcript to one supported intent and extracts the slots needed for fulfillment.",
                                elem_classes=["atlas-section-copy"],
                            )
                            gr.HTML(
                                f"""
                                <section class="atlas-step-meta">
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Intent Model</div>
                                    <div class="atlas-meta-value">{actions.runtime.intent_model_status}</div>
                                  </div>
                                  <div class="atlas-meta-card">
                                    <div class="atlas-meta-label">Check</div>
                                    <div class="atlas-meta-value">Confirm the intent and slots before running the action.</div>
                                  </div>
                                </section>
                                """
                            )
                            gr.Markdown("Predicted intent", elem_classes=["atlas-section-title"])
                            intent_box = gr.Textbox(label="Detected Intent", show_label=False, elem_classes=["atlas-field"])
                            gr.Markdown("Extracted slots", elem_classes=["atlas-section-title"])
                            slots_box = gr.Textbox(label="Detected Slots", lines=6, elem_classes=["atlas-field", "atlas-code"], show_label=False)
                            with gr.Row():
                                btn_intent = gr.Button("Detect Intent", variant="primary")
                            with gr.Accordion("Manual / Bypass Options for Intent", open=False):
                                manual_intent = gr.Textbox(label="Manual Intent", value="LightOn")
                                manual_slots = gr.Textbox(
                                    label="Manual Slots (JSON)",
                                    lines=6,
                                    value='''{
  "ROOM": "bedroom"
}''',
                                    elem_classes=["atlas-code"],
                                )
                                btn_manual_intent = gr.Button("Use Manual Intent / Slots")

                    with gr.Column(visible=False) as fulfillment_panel:
                        with gr.Group(elem_classes=["atlas-stage-panel", "atlas-stage-panel-primary"]):
                            gr.Markdown("## Step 5. Action / Fulfillment")
                            gr.Markdown(
                                "Fulfillment turns the intent into an API response, local timer action, or simulated dorm-state update.",
                                elem_classes=["atlas-section-copy"],
                            )
                            gr.Markdown("Action result", elem_classes=["atlas-section-title"])
                            api_box = gr.Textbox(label="Fulfillment / API Output", lines=10, elem_classes=["atlas-field", "atlas-code"], show_label=False)
                            with gr.Row():
                                btn_fulfill = gr.Button("Run Action", variant="primary")
                            with gr.Accordion("Manual / Bypass Options for Fulfillment", open=False):
                                manual_api_result = gr.Textbox(
                                    label="Manual API Result (JSON)",
                                    lines=6,
                                    value='''{
  "status": "success",
  "message": "bedroom light turned on"
}''',
                                    elem_classes=["atlas-code"],
                                )
                                btn_manual_api = gr.Button("Use Manual API Result")

                    with gr.Column(visible=False) as response_panel:
                        with gr.Group(elem_classes=["atlas-stage-panel", "atlas-stage-panel-primary"]):
                            gr.Markdown("## Step 6. Assistant Response")
                            gr.Markdown(
                                "Generate a user-facing response from the fulfillment result, then synthesize speech for the same answer.",
                                elem_classes=["atlas-section-copy"],
                            )
                            gr.Markdown("Atlas response", elem_classes=["atlas-section-title"])
                            answer_box = gr.Textbox(label="Final Answer", lines=4, elem_classes=["atlas-field", "atlas-response"], show_label=False)
                            gr.Markdown("Speech status", elem_classes=["atlas-section-title"])
                            tts_status_box = gr.Textbox(label="TTS Output", lines=2, show_label=False, elem_classes=["atlas-field"])
                            gr.Markdown(f"TTS backend: {actions.runtime.tts_status}", elem_classes=["atlas-section-copy"])
                            gr.Markdown("Audio playback", elem_classes=["atlas-section-title"])
                            tts_audio_output = gr.Audio(label="TTS Audio", show_label=False, elem_classes=["atlas-audio"])
                            with gr.Row():
                                btn_answer = gr.Button("Generate Answer", variant="primary")
                                btn_tts = gr.Button("Run TTS")
                                btn_reset = gr.Button("Reset All")
                            with gr.Accordion("Manual / Bypass Options for Answer Generation", open=False):
                                manual_answer = gr.Textbox(
                                    label="Manual Answer",
                                    lines=3,
                                    value="The bedroom light is now on.",
                                    elem_classes=["atlas-field", "atlas-response"],
                                )
                                btn_manual_answer = gr.Button("Use Manual Answer")

        ready_timer = gr.Timer(1.0)

        example_weather.click(lambda: "does it rain in Ottawa today", outputs=typed_transcript_input)
        example_movie.click(lambda: "who directed Dune Part Two", outputs=typed_transcript_input)
        example_dorm.click(lambda: "turn on the bedroom light", outputs=typed_transcript_input)
        example_oos.click(lambda: "book a flight to Toronto", outputs=typed_transcript_input)

        btn_verify.click(
            fn=actions.do_verify_ui,
            inputs=[verification_audio_input, state],
            outputs=[verify_output, verification_scores_output, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_skip_verify.click(
            fn=actions.verify_with_code_ui,
            inputs=[verification_code_input, state],
            outputs=[verify_output, verification_scores_output, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_reset_verification.click(
            fn=actions.reset_verification_ui,
            inputs=[state],
            outputs=[verify_output, verification_code_input, verification_scores_output, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_wake.click(
            fn=actions.do_wake_ui,
            inputs=[wake_audio_input, state],
            outputs=[wake_output, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_skip_wake.click(
            fn=actions.skip_wake_with_code_ui,
            inputs=[wake_code_input, state],
            outputs=[wake_output, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_reset_wake.click(
            fn=actions.reset_wake_word_ui,
            inputs=[state],
            outputs=[wake_output, wake_code_input, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_asr.click(
            fn=actions.do_asr_ui,
            inputs=[command_audio_input, state],
            outputs=[transcript_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_use_typed_transcript.click(
            fn=actions.use_typed_transcript_ui,
            inputs=[typed_transcript_input, state],
            outputs=[transcript_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_intent.click(
            fn=actions.do_intent_ui,
            inputs=[transcript_box, state],
            outputs=[intent_box, slots_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_manual_intent.click(
            fn=actions.use_manual_intent_ui,
            inputs=[manual_intent, manual_slots, state],
            outputs=[intent_box, slots_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_fulfill.click(
            fn=actions.do_fulfillment_ui,
            inputs=[state],
            outputs=[api_box, control_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_manual_api.click(
            fn=actions.use_manual_api_result_ui,
            inputs=[manual_api_result, state],
            outputs=[api_box, control_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_answer.click(
            fn=actions.do_answer_ui,
            inputs=[state],
            outputs=[answer_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_manual_answer.click(
            fn=actions.use_manual_answer_ui,
            inputs=[manual_answer, state],
            outputs=[answer_box, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_tts.click(
            fn=actions.do_tts_ui,
            inputs=[state],
            outputs=[tts_status_box, tts_audio_output, state, assistant_status, assistant_status_raw, pipeline_timeline, dorm_visual, stage_hint_html, verify_panel, wake_panel, asr_panel, intent_panel, fulfillment_panel, response_panel, dorm_panel],
        )
        btn_reset.click(
            fn=actions.reset_all_ui,
            inputs=[],
            outputs=[
                state,
                verification_audio_input,
                wake_audio_input,
                command_audio_input,
                assistant_status,
                assistant_status_raw,
                ready_countdown_box,
                verify_output,
                verification_scores_output,
                verification_code_input,
                wake_output,
                wake_code_input,
                transcript_box,
                typed_transcript_input,
                intent_box,
                slots_box,
                api_box,
                control_box,
                answer_box,
                tts_status_box,
                tts_audio_output,
                manual_answer,
                manual_intent,
                manual_slots,
                manual_api_result,
                pipeline_timeline,
                dorm_visual,
                stage_hint_html,
                verify_panel,
                wake_panel,
                asr_panel,
                intent_panel,
                fulfillment_panel,
                response_panel,
                dorm_panel,
            ],
        )
        ready_timer.tick(
            fn=actions.tick_ready_timer_ui,
            inputs=[state],
            outputs=[
                state,
                assistant_status,
                assistant_status_raw,
                ready_countdown_box,
                pipeline_timeline,
                dorm_visual,
                stage_hint_html,
                verify_panel,
                wake_panel,
                asr_panel,
                intent_panel,
                fulfillment_panel,
                response_panel,
                dorm_panel,
            ],
        )

    return demo
