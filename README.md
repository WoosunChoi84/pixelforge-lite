# PixelForge Lite

[![CI](https://github.com/WoosunChoi84/pixelforge-lite/actions/workflows/ci.yml/badge.svg)](https://github.com/WoosunChoi84/pixelforge-lite/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/pixelforge-lite.svg)](https://pypi.org/project/pixelforge-lite/)
[![Python versions](https://img.shields.io/pypi/pyversions/pixelforge-lite.svg)](https://pypi.org/project/pixelforge-lite/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

픽셀 아트 입력 전용 경량 SDK. 이미지를 픽셀 아트로 변환하는 파이썬 파이프라인.

## 설치

```bash
pip install pixelforge-lite
```

또는 로컬 개발 설치:

```bash
pip install -e .
```

**의존성**: numpy, Pillow, opencv-python (자동 설치).

## 기본 사용법

```python
import pixelforge_litemodel as pf

# 프리셋으로 변환
pipeline = pf.PixelArtPipeline(preset="icon_48")
result = pipeline.convert("input.jpg")
result.save("output.png")

# 커스텀 그리드 크기로 변환
pipeline = pf.PixelArtPipeline(grid_size=(64, 64), max_colors=16)
result = pipeline.convert("photo.png")
result.save("pixel_art.png")

# 프리셋 + 그리드 오버라이드
pipeline = pf.PixelArtPipeline(preset="gameboy", grid_size=(32, 32))
```

## 입력 형식

파일 경로, NumPy 배열, 바이트 데이터, PIL RGBA 이미지를 입력으로 받는다.

```python
# 파일 경로
result = pipeline.convert("image.png")

# NumPy 배열 (H, W, 3) uint8 RGB 또는 (H, W, 4) uint8 RGBA
import numpy as np
img = np.array(...)
result = pipeline.convert(img)

# 바이트 데이터
with open("image.png", "rb") as f:
    result = pipeline.convert(f.read())
```

**투명 대응**: RGBA PNG 입력 시 alpha 채널을 binary mask로 이진화(임계값 128)하여 파이프라인 전체에 병렬 전달. 투명 영역은 출력에서도 투명으로 보존된다.

## 출력 사용

```python
result = pipeline.convert("input.jpg")

# 파일로 저장 (RGBA 입력은 자동으로 RGBA PNG 출력)
result.save("output.png")

# PIL Image 객체로 변환
pil_image = result.to_pil()

# 바이트로 직렬화
png_bytes = result.to_bytes(format="png")

# 팔레트 확인
print(result.palette)  # [(r, g, b), ...]

# alpha 마스크 확인 (RGBA 입력인 경우)
print(result.alpha)  # (H, W) uint8 {0, 255} 또는 None
```

## 프리셋

```python
# 프리셋 목록
print(pf.list_presets())
# ['cryptopunks', 'gameboy', 'gba', 'hd_384', 'hd_512', 'icon_32',
#  'icon_48', 'icon_64', 'mid_128', 'mid_256', 'nes_sprite',
#  'snes_large', 'snes_small', 'stardew']

# 프리셋 정보
info = pf.get_preset_info("gameboy")
print(info)  # {'name': 'gameboy', 'grid_size': (8, 8), 'description': '...'}

# 사용
pipeline = pf.PixelArtPipeline(preset="stardew")
result = pipeline.convert("character.png")
```

프리셋 파라미터 오버라이드:

```python
pipeline = pf.PixelArtPipeline(preset="gameboy", max_colors=4, dithering=True)
```

## 팔레트

```python
# 내장 팔레트 목록
print(pf.list_palettes())
# ['cga_16', 'endesga_32', 'gameboy_4', 'nes_54', 'pico8_16', 'snes_256', 'sweetie_16']

# 내장 팔레트 사용
pipeline = pf.PixelArtPipeline(grid_size=(32, 32), palette="pico8_16")

# 커스텀 팔레트 사용
my_palette = [(0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 0, 255)]
pipeline = pf.PixelArtPipeline(grid_size=(32, 32), palette=my_palette)
```

## 설정 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `grid_size` | `(int, int)` | 필수 | 출력 해상도 (너비, 높이) |
| `aspect_ratio` | `str` | `"fit"` | 비율 처리: `"fit"`, `"fill"`, `"stretch"` |
| `max_colors` | `int \| None` | `None` (자동) | 최대 색상 수 (1-256) |
| `palette` | `str \| list \| None` | `None` (자동) | 팔레트 이름 또는 RGB 리스트 |
| `dithering` | `bool` | `False` | 디더링 활성화 |
| `dither_method` | `str` | `"floyd"` | 디더링 방식: `"floyd"`, `"bayer"`, `"atkinson"` |

`preset`을 사용하는 경우 추가 파라미터로 프리셋 값을 오버라이드한다.

## 바이트 입출력

```python
pipeline = pf.PixelArtPipeline(preset="icon_32")
output_bytes = pipeline.convert_bytes(input_bytes, format="png")
```

## 파이프라인 내부 동작

```
Input
  │
  ├─ Stage 0: input_analyzer
  │    unique color count, scaling_direction (down/up/identity),
  │    palette_sufficient, dominant_color
  │
  ├─ Stage 1: preprocessor
  │    RGBA → RGB (white 배경 합성), aspect_ratio 처리
  │
  ├─ Stage 2: resampler (direction-aware)
  │    up       → nearest (cv2.INTER_NEAREST)
  │    down     → CIELAB + k=2 k-means mode resampling with C6 bg-aware gate
  │    identity → copy / minor INTER_NEAREST
  │
  └─ Stage 3: quantizer
       palette_sufficient → skip
       else → CIELAB ΔE² nearest + optional dithering (floyd/bayer/atkinson)
       → Output
```

**핵심 알고리즘 특징**:
- **CIELAB + k=2 k-means**: 블록 내 AA 전이 픽셀과 코어 픽셀을 분리
- **C6 bg-aware gate**: `(dominant_ratio ≥ 0.25) AND (edge_dom_ratio ≥ 0.7)` 조건에서만 오버라이드 작동
- **결정론적 초기화**: L축 분위 기반 k-means 초기화로 재현성 확보
- **Fast path**: 이미 팔레트 충분한 입력은 양자화 단계 스킵 (pixel art 라운드트립 비트 정확)

`result.metadata`에 각 스테이지 실행 로그(소요 시간, 입출력 shape, 방법 등)가 기록된다.

## 파일 구조

```
pixelforge_litemodel/
├── __init__.py              # SDK API
├── models.py                # PipelineConfig, StageResult, InputProfile, ConvertResult
├── pipeline.py              # 4-stage orchestrator
├── presets.py               # 14개 프리셋 레지스트리
├── tuning.py                # max_colors 로그 보간
├── utils.py                 # rgb_to_lab, composite_rgba_on_white, resample_alpha_binary
├── palettes/
│   ├── __init__.py          # PaletteRegistry
│   └── data/*.json          # 7개 내장 팔레트
└── stages/
    ├── input_analyzer.py
    ├── preprocessor.py
    ├── resampler.py
    └── quantizer.py
```

## 버전

현재 버전: `0.1.0` (Beta → Release Candidate 이행 중). `pixelforge_litemodel.__version__` 참조.
