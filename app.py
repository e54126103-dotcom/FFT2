"""
Active Noise Cancellation FFT Web App

How to run:

    pip install -r requirements.txt
    streamlit run app.py

Upload or record a WAV sample, choose a reconstruction model, and run the
FFT-based active noise cancellation simulation.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy.io import wavfile

from analyze_noise import (
    DEFAULT_BURST_ATTENUATION,
    DEFAULT_BURST_PADDING_MS,
    DEFAULT_BURST_THRESHOLD_MULTIPLIER,
    DEFAULT_BURST_WINDOW_MS,
    DEFAULT_DELAYS_MS,
    DEFAULT_TOP_N_VALUES,
    MODE_ENERGY,
    MODE_HARMONIC,
    MODE_PEAK_BAND,
    MODE_TOP_N,
    RECONSTRUCTION_MODES,
    load_audio_mono,
    normalize_audio_array,
    run_analysis,
)


OUTPUT_DIR = Path("output")


st.set_page_config(page_title="FFT 主動降噪分析", layout="wide")


def parse_top_n_values(raw_text: str) -> list[int]:
    values = []
    for item in raw_text.replace("，", ",").split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("N must be positive.")
        values.append(value)
    if not values:
        raise ValueError("Please enter at least one N value.")
    return sorted(set(values))


def read_uploaded_wav(uploaded_file) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(BytesIO(uploaded_file.getvalue()))
    return sample_rate, normalize_audio_array(audio)


def save_recorded_audio(audio_recording, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    recorded_path = output_dir / "recorded_noise.wav"
    audio_bytes = audio_recording.getvalue()
    with open(recorded_path, "wb") as file:
        file.write(audio_bytes)
    return recorded_path


def show_download_button(path: Path, label: str, mime: str) -> None:
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            use_container_width=True,
        )


def show_image(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def display_numeric_table(df: pd.DataFrame) -> pd.DataFrame:
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: f"{value:.6g}")
    return display


st.title("FFT 主動降噪模擬分析")
st.caption("用 Fourier Transform 保留 complex FFT coefficients，重建主要噪音結構並產生反相信號。")

with st.expander("執行方式", expanded=False):
    st.code("pip install -r requirements.txt\nstreamlit run app.py", language="bash")

st.info("建議錄製穩定低頻噪音，例如冷氣、風扇、抽風機。第一次不要錄人聲或音樂。")
st.markdown(
    """
真實冷氣、風扇、馬達噪音通常不是由任意的 top 10 個頻率組成，而是常具有主要基頻與倍頻結構。若只保留少數離散 FFT bins，重建效果通常會很差。因此本工具預設採用「基頻與倍頻模式」，先偵測輸入音訊的主要基頻 f0，再保留 f0 及其倍頻附近的頻帶，以更符合穩定機械噪音的物理特性。
"""
)

st.subheader("音訊來源")
audio_source = st.radio("音訊來源", options=["上傳 WAV 檔", "直接錄音"], horizontal=True)
uploaded_file = None
audio_recording = None
selected_audio = None
selected_audio_path = None

if audio_source == "上傳 WAV 檔":
    uploaded_file = st.file_uploader("上傳 WAV 噪音音檔", type=["wav"])
    selected_audio = uploaded_file
else:
    st.caption("請錄製 5-10 秒冷氣聲、風扇聲或環境噪音。")
    audio_recording = st.audio_input("直接錄製噪音樣本", sample_rate=44100)
    if audio_recording is not None:
        recorded_path = save_recorded_audio(audio_recording, OUTPUT_DIR)
        selected_audio = audio_recording
        selected_audio_path = recorded_path
        st.success("錄音完成，請按下開始分析。")
        st.audio(audio_recording.getvalue(), format="audio/wav")
        st.caption(f"錄音已儲存為 {recorded_path.as_posix()}")

with st.sidebar:
    st.header("分析參數")
    reconstruction_mode = st.selectbox("重建模式", RECONSTRUCTION_MODES, index=0)
    if reconstruction_mode == MODE_TOP_N:
        st.caption("這個模式只保留少數離散頻率點，對真實噪音通常重建效果較差，主要作為傅立葉主頻分析的簡化示範。")

    min_freq_hz = st.number_input("最低分析頻率 (Hz)", min_value=0.0, value=20.0, step=5.0)
    max_freq_hz = st.number_input("最高分析頻率 (Hz)", min_value=1.0, value=3000.0, step=100.0)
    fundamental_max_freq_hz = st.number_input("基頻搜尋上限 (Hz)", min_value=1.0, value=1000.0, step=50.0)
    harmonic_bandwidth_hz = st.selectbox("harmonic_bandwidth_hz (Hz)", [1.0, 3.0, 5.0, 10.0, 20.0], index=2)
    auto_harmonics = st.checkbox("最大倍頻數自動到最高分析頻率", value=True)
    max_harmonics = None
    if not auto_harmonics:
        max_harmonics = st.number_input("最大倍頻數", min_value=1, value=20, step=1)

    energy_ratio_percent = st.select_slider("能量保留比例", options=[70, 80, 90, 95, 99], value=90)
    peak_band_count = st.number_input("Peak 頻帶保留數", min_value=1, value=10, step=1)
    top_n_text = st.text_input(
        "Top N 選項（比較用）",
        value=", ".join(str(value) for value in DEFAULT_TOP_N_VALUES),
        help="用逗號分隔，例如：1, 3, 5, 10, 20, 50",
    )
    main_n = st.number_input("Top N 模式使用的 N", min_value=1, value=10, step=1)
    st.divider()
    st.markdown("**延遲模擬**")
    st.caption("使用目前選擇的 reconstruction mode。預設延遲：0, 0.1, 0.5, 1, 2, 5, 10 ms")
    st.divider()
    st.markdown("**突發降噪**")
    enable_burst_reduction = st.checkbox("啟用突發噪音偵測與衰減", value=True)
    burst_window_ms = st.number_input(
        "突發偵測 RMS 視窗 (ms)", min_value=1.0, value=DEFAULT_BURST_WINDOW_MS, step=1.0
    )
    burst_threshold_multiplier = st.number_input(
        "突發偵測門檻倍數",
        min_value=1.0,
        value=DEFAULT_BURST_THRESHOLD_MULTIPLIER,
        step=0.5,
        help="數值越大越不容易判定為突發噪音。",
    )
    burst_padding_ms = st.number_input(
        "突發區段前後保留 (ms)", min_value=0.0, value=DEFAULT_BURST_PADDING_MS, step=5.0
    )
    burst_attenuation = st.slider(
        "突發區段衰減比例", min_value=0.0, max_value=1.0, value=DEFAULT_BURST_ATTENUATION, step=0.05
    )

run_clicked = st.button("Run FFT Noise Cancellation Analysis", type="primary", use_container_width=True)

if selected_audio is None:
    st.info("請先上傳音檔或完成錄音。")

if run_clicked:
    if selected_audio is None:
        st.error("請先上傳音檔或完成錄音。")
        st.stop()

    try:
        top_n_values = parse_top_n_values(top_n_text)
    except ValueError as exc:
        st.error(f"Top N 參數格式錯誤：{exc}")
        st.stop()

    if min_freq_hz >= max_freq_hz:
        st.error("最低分析頻率必須小於最高分析頻率。")
        st.stop()

    with st.spinner("正在偵測基頻、建立重建模型、計算指標並輸出檔案..."):
        try:
            if selected_audio_path is not None:
                sample_rate, original = load_audio_mono(selected_audio_path)
            else:
                sample_rate, original = read_uploaded_wav(selected_audio)

            result = run_analysis(
                sample_rate=sample_rate,
                original=original,
                output_dir=OUTPUT_DIR,
                min_freq_hz=min_freq_hz,
                max_freq_hz=max_freq_hz,
                fundamental_max_freq_hz=fundamental_max_freq_hz,
                reconstruction_mode=reconstruction_mode,
                harmonic_bandwidth_hz=harmonic_bandwidth_hz,
                max_harmonics=int(max_harmonics) if max_harmonics is not None else None,
                energy_ratio=energy_ratio_percent / 100.0,
                peak_bandwidth_hz=harmonic_bandwidth_hz,
                peak_band_count=int(peak_band_count),
                top_n_values=top_n_values,
                main_n=int(main_n),
                delays_ms=DEFAULT_DELAYS_MS,
                enable_burst_reduction=enable_burst_reduction,
                burst_window_ms=burst_window_ms,
                burst_threshold_multiplier=burst_threshold_multiplier,
                burst_padding_ms=burst_padding_ms,
                burst_attenuation=burst_attenuation,
            )
        except Exception as exc:
            st.error(f"分析失敗：{exc}")
            st.stop()

    st.session_state["analysis_result"] = result
    st.success("分析完成，所有輸出已儲存到 output/ 資料夾。")

result = st.session_state.get("analysis_result")

if result:
    sample_rate = result["sample_rate"]
    audio_outputs = result["audio_outputs"]
    plot_paths = result["plot_paths"]
    csv_paths = result["csv_paths"]
    selected_metrics = result["selected_metrics"]
    f0 = float(result["fundamental_frequency"].iloc[0]["detected_f0_Hz"])

    st.subheader("音檔與模型資訊")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Sample rate", f"{sample_rate} Hz")
    col_b.metric("Duration", f"{result['duration']:.2f} s")
    col_c.metric("偵測到主要基頻", f"{f0:.2f} Hz")
    col_d.metric("保留 bins", str(selected_metrics["retained_bins"]))
    st.caption(f"目前重建模式：{result['reconstruction_mode']}")

    st.subheader("基頻與倍頻表")
    f0_col, harmonic_col = st.columns([1, 2])
    with f0_col:
        st.markdown("**fundamental_frequency.csv**")
        st.dataframe(display_numeric_table(result["fundamental_frequency"]), use_container_width=True, hide_index=True)
    with harmonic_col:
        st.markdown("**harmonic_components.csv**")
        st.dataframe(display_numeric_table(result["harmonic_components"]), use_container_width=True, hide_index=True)

    st.subheader("原始頻譜與倍頻頻帶")
    col_spec, col_harmonic = st.columns(2)
    with col_spec:
        show_image(plot_paths["spectrum_before"], "原始頻譜，標出 f0")
    with col_harmonic:
        show_image(plot_paths["harmonic_spectrum"], "f0 與倍頻位置")
    show_image(plot_paths["selected_harmonic_bands"], "selected_harmonic_bands.png")

    st.subheader("重建與殘餘結果")
    col_wave, col_recon_spec = st.columns(2)
    with col_wave:
        show_image(plot_paths["original_vs_reconstructed_waveform"], "原始噪音 vs. 重建噪音波形")
    with col_recon_spec:
        show_image(plot_paths["original_vs_reconstructed_spectrum"], "原始頻譜 vs. 重建頻譜")
    show_image(plot_paths["residual_spectrum"], "residual_spectrum.png")

    st.subheader("模式比較與 metrics.csv")
    st.dataframe(display_numeric_table(result["metrics"]), use_container_width=True, hide_index=True)
    show_image(plot_paths["mode_comparison"], "mode_comparison.png")

    st.subheader("音訊播放")
    audio_cols = st.columns(4)
    audio_items = [
        ("原始音訊", audio_outputs["original"]),
        ("重建主噪音", audio_outputs["reconstructed_main_noise"]),
        ("反相信號", audio_outputs["anti_noise"]),
        ("殘餘噪音", audio_outputs["residual_noise"]),
    ]
    for column, (label, path) in zip(audio_cols, audio_items):
        with column:
            st.markdown(f"**{label}**")
            st.audio(path.read_bytes(), format="audio/wav")

    st.divider()
    st.header("第二階段：真實時間延遲模擬")
    st.caption("延遲模擬使用目前選擇的 reconstruction mode 產生的 reconstructed_main_noise；反波延遲採 zero padding，不使用 np.roll。")
    st.dataframe(display_numeric_table(result["delay_metrics"]), use_container_width=True, hide_index=True)
    col_delay_mse, col_delay_nrr = st.columns(2)
    with col_delay_mse:
        show_image(plot_paths["delay_vs_mse"], "delay_vs_mse.png")
    with col_delay_nrr:
        show_image(plot_paths["delay_vs_nrr"], "delay_vs_nrr.png")
    show_image(plot_paths["delayed_signal_comparison"], "delayed_signal_comparison.png")

    delayed_audio_keys = [
        ("0 ms 延遲反波", "delayed_anti_noise_0p0ms"),
        ("1 ms 延遲反波", "delayed_anti_noise_1p0ms"),
        ("5 ms 延遲反波", "delayed_anti_noise_5p0ms"),
        ("10 ms 延遲反波", "delayed_anti_noise_10p0ms"),
    ]
    st.markdown("**延遲後反相信號播放**")
    delay_audio_cols = st.columns(4)
    for column, (label, key) in zip(delay_audio_cols, delayed_audio_keys):
        path = audio_outputs.get(key)
        if path is not None and path.exists():
            with column:
                st.markdown(f"**{label}**")
                st.audio(path.read_bytes(), format="audio/wav")

    if not result["burst_metrics"].empty:
        st.divider()
        st.header("第三階段：突發噪音偵測與降噪")
        st.caption("突發降噪使用短時間 RMS 包絡偵測瞬間尖峰，並只衰減突發區段；適合敲擊、爆音、瞬間風噪等非穩態噪音。")
        st.dataframe(display_numeric_table(result["burst_metrics"]), use_container_width=True, hide_index=True)
        st.markdown("**偵測到的突發區段**")
        st.dataframe(display_numeric_table(result["burst_segments"]), use_container_width=True, hide_index=True)
        col_burst_env, col_burst_wave = st.columns(2)
        with col_burst_env:
            show_image(plot_paths["burst_detection_envelope"], "burst_detection_envelope.png")
        with col_burst_wave:
            show_image(plot_paths["burst_waveform_comparison"], "burst_waveform_comparison.png")
        show_image(plot_paths["burst_spectrum_comparison"], "burst_spectrum_comparison.png")

        st.markdown("**突發降噪音訊播放**")
        burst_audio_cols = st.columns(3)
        burst_audio_items = [
            ("原始音訊", audio_outputs["original"]),
            ("突發降噪後音訊", audio_outputs["burst_reduced_noise"]),
            ("被移除/衰減的突發成分", audio_outputs["burst_removed_noise"]),
        ]
        for column, (label, path) in zip(burst_audio_cols, burst_audio_items):
            with column:
                st.markdown(f"**{label}**")
                st.audio(path.read_bytes(), format="audio/wav")

    st.divider()
    st.header("下載輸出檔案")

    st.markdown("**CSV 表格**")
    csv_paths_list = list(csv_paths.items())
    for row_start in range(0, len(csv_paths_list), 3):
        cols = st.columns(3)
        for column, (name, path) in zip(cols, csv_paths_list[row_start : row_start + 3]):
            with column:
                show_download_button(path, f"下載 {path.name}", "text/csv")

    st.markdown("**WAV 音檔**")
    wav_paths = list(audio_outputs.items())
    for row_start in range(0, len(wav_paths), 3):
        cols = st.columns(3)
        for column, (name, path) in zip(cols, wav_paths[row_start : row_start + 3]):
            with column:
                show_download_button(path, f"下載 {path.name}", "audio/wav")

    st.markdown("**PNG 圖表**")
    png_paths = list(plot_paths.items())
    for row_start in range(0, len(png_paths), 3):
        cols = st.columns(3)
        for column, (name, path) in zip(cols, png_paths[row_start : row_start + 3]):
            with column:
                show_download_button(path, f"下載 {path.name}", "image/png")
