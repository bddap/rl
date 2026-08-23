# rl#404 — full-moon brightness to 30% apparent; ground reflectance 0.8

Full-phase ground vista (`game fp-screenshot --ground-look shipped --seed 7
--settle 90 --moon-elevation-deg 45 --moon-phase 0.5 --cam-pitch=-20
--cam-height 1.5`), lavapipe headless.

Mean linear luminance (ImageMagick, `-colorspace RGB %[fx:mean]`):

| render | mean | vs baseline |
|---|---|---|
| baseline.png (lux 9500, reflectance 1.0) | 0.2881 | 100% |
| moon-only probe (lux 2300, reflectance 1.0) | 0.0866 | **30.05%** — the tuned target |
| after.png (lux 2300, reflectance 0.8) | 0.0730 | 25.3% |
| after-low-elev.png (elev 22°, the rl#372 band) | 0.0491 | — no bright-sheet or shadow artifacts |

Lux was tuned on rendered output, not scaled 0.3x on the constant: the 400-lux
ambient floor makes as-rendered nonproportional in lux (0.3x lux measured 35.4%).
