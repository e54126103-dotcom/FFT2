# FFT 主動降噪分析工具

這個專案用 FFT 分析真實噪音錄音，保留選定頻率範圍中的 complex FFT coefficients 來重建主要噪音，再產生反相信號並計算殘餘噪音的 MSE、RMS 與 NRR dB。

## 執行方式

```bash
pip install -r requirements.txt
streamlit run app.py
```

也可以把 `input_noise.wav` 放在專案資料夾後用命令列執行：

```bash
python analyze_noise.py
```

## 建議錄音

建議先錄製 5-10 秒穩定低頻噪音，例如冷氣、風扇、抽風機或馬達聲。第一次不要錄人聲或音樂，因為它們通常不是穩定的基頻與倍頻結構。

## 四種重建模式

### 1. 基頻與倍頻模式

這是預設與建議模式。程式會先在指定頻率範圍內偵測最主要的基頻 `f0`，再建立 `f0, 2f0, 3f0, ...` 的倍頻結構，並保留每個倍頻附近的頻帶，例如 `f0 = 60 Hz` 且 bandwidth 為 `5 Hz` 時，會保留 `55-65 Hz`、`115-125 Hz`、`175-185 Hz` 等頻帶。

這種方法比較符合冷氣、風扇、馬達等穩定機械噪音的物理特性。

### 2. 能量比例保留模式

程式會依照 FFT power 由大到小排序頻率 bins，從最大能量開始保留，直到累積能量達到指定比例，例如 90%。這個模式不固定保留 top 10，而是根據輸入音訊本身的能量分布決定要保留多少個 bins。

適合用來觀察「保留多少頻譜能量才足以近似原噪音」。

### 3. Peak 頻帶保留模式

程式會找出頻譜中的主要 peaks，但不是只保留單一 FFT bin，而是保留每個 peak 附近的一段頻帶。這比只抓離散頻率點更接近真實噪音，因為真實錄音中的峰值常會因錄音長度、轉速微變或環境反射而分散在附近頻帶。

### 4. Top N 頻率點模式（簡化比較用）

這是原本的簡化示範模式，只保留少數振幅最大的離散 FFT bins。它可以用來說明傅立葉主頻分析，但對真實噪音通常重建效果較差，因為真實噪音不一定由任意 top N 個單點頻率組成。

## 重要實作原則

- 主實驗不直接使用 `anti_noise = -original_signal`。
- 所有重建模式都保留 complex FFT coefficients，因此振幅與相位都會被保留。
- 使用 `numpy.fft.rfft` 與 `numpy.fft.irfft`。
- 延遲模擬使用 zero padding，不使用 circular shift 或 `np.roll()`。
- ideal reference 只作為理想對照。

## 延遲訊號模擬

延遲模擬會使用目前選擇的 reconstruction mode 產生的 `reconstructed_main_noise`，再建立：

```text
anti_noise_delayed(t) = -reconstructed_main_noise(t - tau)
```

程式會輸出不同延遲下的反相信號與殘餘訊號，例如 `delayed_anti_noise_1p0ms.wav`、`delayed_residual_1p0ms.wav`，並產生 `delay_vs_mse.png`、`delay_vs_nrr.png`、`delayed_signal_comparison.png`。

## 突發噪音降噪

穩定機械噪音適合用基頻與倍頻模型；但敲擊聲、爆音、瞬間風噪等突發噪音不是穩定週期訊號。網站新增「突發噪音偵測與衰減」功能，使用短時間 RMS 包絡偵測瞬間能量尖峰，將高於門檻的區段加上前後 padding 後做衰減。

可調參數包含：

- RMS 視窗長度
- 偵測門檻倍數
- 突發區段前後 padding
- 衰減比例

輸出包含 `burst_metrics.csv`、`burst_segments.csv`、`burst_reduced_noise.wav`、`burst_removed_noise.wav`、`burst_detection_envelope.png`、`burst_waveform_comparison.png`、`burst_spectrum_comparison.png`。

## 輸出檔案

所有輸出會存到 `output/`。

- `fundamental_frequency.csv`
- `harmonic_components.csv`
- `dominant_frequencies.csv`
- `metrics.csv`
- `delay_metrics.csv`
- `original_normalized.wav`
- `reconstructed_*.wav`
- `anti_noise_*.wav`
- `residual_*.wav`
- `spectrum_before.png`
- `harmonic_spectrum.png`
- `selected_harmonic_bands.png`
- `original_vs_reconstructed_waveform.png`
- `original_vs_reconstructed_spectrum.png`
- `residual_spectrum.png`
- `mode_comparison.png`
- `delay_vs_mse.png`
- `delay_vs_nrr.png`
- `delayed_signal_comparison.png`
- `burst_metrics.csv`
- `burst_segments.csv`
- `burst_reduced_noise.wav`
- `burst_removed_noise.wav`
- `burst_detection_envelope.png`
- `burst_waveform_comparison.png`
- `burst_spectrum_comparison.png`
