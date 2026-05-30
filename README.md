
# Speech Separation & ASR Feasibility Test
### Kazakh Audiobook with Background Music → Whisper Transcription + WER Evaluation

This notebook is a feasibility study for transcribing Kazakh-language audio that contains intentionally added background instrumental music (bed music), a common format in audiobooks and storytelling recordings.

---

## Problem Statement

The target dataset consists of Kazakh speech recordings with accompanying background music — similar to narrated audiobooks or fairy tales. The goal is to evaluate whether separating the speech from the music before transcription improves ASR accuracy, measured by Word Error Rate (WER).

---

## Test Data

- **Audio source:** Ілияс Есенберлін — *Ғашықтар*, 1-бөлім (YouTube, ~30 min)
- **Reference text:** Book transcript (`book_transcript_gashyktar001.txt`)
- **Language:** Kazakh (`kk`)
- **Audio starts at:** 6 seconds (skips intro title/author announcement)

---

## Pipeline

```
YouTube audio
      │
      ▼
 yt-dlp (download as WAV)
      │
      ▼
 ffmpeg (trim from 6s onwards)
      │
      ├──────────────────────────┐
      │                          │
      ▼                          ▼
 Raw audio               Demucs htdemucs
 (with music)         (vocals stem only)
      │                          │
      ▼                          ▼
 Whisper large-v3        Whisper large-v3
 (language=kk)           (language=kk)
      │                          │
      ▼                          ▼
 transcript_raw.txt      transcript_sep.txt
      │                          │
      └──────────┬───────────────┘
                 ▼
          WER comparison
         (jiwer, normalized)
```

---

## Requirements

### System
- Python 3.11+
- ffmpeg (must be on PATH)
- CUDA-capable GPU (recommended; CPU fallback supported but slow)

### Python packages
```bash
pip install yt-dlp faster-whisper demucs jiwer
```

---

## Notebook Structure

| Cell | Description |
|------|-------------|
| 1 | Verify ffmpeg installation + run Demucs separation on full audio, measure processing time |
| 2 | Trim both raw and separated audio starting from 6 seconds |
| 3 | Transcribe both files using Whisper `large-v3` |
| 4 | Install jiwer |
| 5 | Compute and compare WER with text normalization |

---

## Results

| Condition | WER |
|-----------|-----|
| With background music (raw) | 79.3% |
| After Demucs separation | 78.5% |
| Relative improvement | 1.0% |

Separation provides a modest but real improvement of 1.0% relative WER reduction.

---

## Notes

- **Reference text:** `book_transcript_gashyktar001.txt` contains Chapter 1 of the book only (105 lines), which matches the audio content exactly — no alignment issue
- **Remaining WER (~79%)** is due to Whisper's imperfect Kazakh transcription: mishearing Kazakh-specific vocabulary, proper nouns, and dialect-specific words (e.g. `Репин → Рейпейнатында`, `Ұлбосын → ұл болсын`)
- Text normalization before WER computation removes punctuation, quotes, and dashes (`—`, `-`) and lowercases all text so formatting differences do not affect the score
- Whisper `large-v3` was used for best available Kazakh language support
- Demucs model: `htdemucs` (default, Hybrid Transformer)
- Environment: Python 3.11, CUDA GPU, Windows 10/11
