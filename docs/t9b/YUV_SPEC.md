# The YUV to RGB stage: a functional specification

This is the functional specification of the first stage of the browser engine's picture path (`yuvToRgb` in `web/src/browser/resample.ts`):
the planes of a decoded 4:2:0 frame in, an RGB picture at the decode size out. It says **what** is computed, as mathematics over the real
numbers and the integers, so that a new implementation can be written from it alone. It holds no code. The reference the stage must match is
the output of PyAV 19.0.1 (libswscale 10.1.102) `VideoFrame.reformat(width, height, format="rgb24")`, as recorded in
`tests/golden/resample_v2.json` (made by `tools/gen_golden.py` from synthetic pictures, never from footage) and checked by
`web/test/golden-resample.test.ts`. The constants below were measured against that output. The filter of section 5 was first written to follow the reference's own filter construction;
this document restates it as behaviour only, so that the stage can be rewritten from the behaviour. Provenance: the author of this document worked from that earlier, swscale-following code (since removed), so the specification is not independent of the earlier code and the stage written from it is not a verified clean-room (see web/src/notices.txt).

The stages after this one (the upright turn and Pillow's BILINEAR squash) and the helpers `scaledSize` and `matrixOf` stay in `resample.ts` and are
not part of this specification, except where the golden pipeline cases chain them (section 11).

Notation. `⌊x⌋` is the floor (the greatest integer not above x), `⌈x⌉` the ceiling, `trunc(x)` rounds toward zero (`trunc(-1.5) = -1`),
`clamp(x, a, b) = min(b, max(a, x))`, `|x|` the absolute value. All integer arithmetic is exact (no wrap-around; intermediate values stay below 2^53,
so IEEE-754 doubles hold them exactly). Where a formula has real coefficients (sections 7 and 8), it is evaluated in IEEE-754 double precision:
the product first, then the additions left to right as written, then the floor.

## 1. Inputs

The function takes a picture `P`, an output width `W` and an output height `H` (both even, at least 2), and optionally a buffer to write into.
`P` carries (the exported `YuvPicture` type):

| field | meaning |
|---|---|
| `format` | `NV12` or `I420` |
| `width`, `height` | the visible size `w × h` of the luma plane (either may be odd) |
| `fullRange` | true: samples use 0 to 255; false: limited range (luma nominally 16 to 235, chroma 16 to 240) |
| `kr`, `kb` | the luma coefficients of the colour matrix |
| `y`, `yStride` | the luma plane: sample `(i, r)` (column i, row r) is byte `r · yStride + i` |
| `u`, `uStride` | I420: the U plane, sample `(i, r)` at byte `r · uStride + i`. NV12: the interleaved chroma plane, U of `(i, r)` at byte `r · uStride + 2i`, V at `r · uStride + 2i + 1` |
| `v`, `vStride` | I420 only: the V plane, sample `(i, r)` at byte `r · vStride + i` |
| `chromaLoc` | `left` or `center`: the horizontal siting of the chroma samples (section 4) |

The chroma planes have `cw = ⌈w/2⌉` columns and `ch = ⌈h/2⌉` rows. Bytes past the visible width of a row (stride padding) are never read and never
affect the output. *Pinned by:* the test "a row stride wider than the picture changes nothing" (the first six `scale` cases with 13 bytes of
padding per row must equal the unpadded result exactly).

## 2. Output

An array of `W · H · 3` bytes: the pixels row by row, top to bottom, each row left to right, each pixel the three bytes R, G, B, no padding.
When a buffer of at least that length is passed in, the result is written into it and that buffer is returned; otherwise a new one is.

## 3. The two paths

If `W = w` and `H = h` the picture is **not scaled** and takes the unscaled path (section 8). Every other size takes the scaled path (sections 4 to 7),
including a picture enlarged by one pixel: `scaledSize(639, 359, 640)` is `640 × 360`, so a stored odd size of 640 or less reaches this stage
enlarged, and the scaled path must handle it.

## 4. Plane geometry of the scaled path

Each of the three planes is resampled separately, then the three are combined pixel by pixel (section 7).

| plane | source size | target size | horizontal filter | vertical filter |
|---|---|---|---|---|
| Y | `w × h` | `W × H` | `F(w, W, 2^14, 128)` | `F(h, H, 2^12, 128)` |
| U, V (each) | `cw × ch` | `⌈W/2⌉ × H` | `F(cw, ⌈W/2⌉, 2^14, σ)` | `F(ch, H, 2^12, 128)` |

So the chroma planes are brought to half the output width but to the full output height, and each resampled chroma sample later serves two
horizontally adjacent output pixels. `F(n, m, N, σ)` is the filter of section 5. The siting parameter `σ` (in 1/256 of a source sample) is
`64` when `chromaLoc` is `left` and `128` when it is `center`; it applies to the chroma horizontal filter only. Luma in both directions and chroma
vertically always use `128` (centred).

*Pinned by:* `nv12_bt709_limited_2x_centre` and `i420_bt601_full_3x_centre` (siting `center`) against all the other `scale` cases (siting `left`);
the flat cases (32 × 32 to 16 × 16, left siting).

## 5. The resampling filter `F(n, m, N, σ)`

`F` maps `n` source samples to `m` target samples. For each target index `k` (0 to `m − 1`) it gives a set of non-negative integer weights on
source indices, summing to exactly `N`. Positions are measured in **ticks** of 1/65536 of a source sample; source sample `i` sits at tick `65536 i`.

1. **Step.** `Δ = ⌊(65536 n + ⌊m/2⌋) / m⌋`: the ratio `n/m` in ticks, rounded to the nearest tick (an exact half rounds up).
2. **Shrinking** means `Δ > 65536`. **Tap count:** `T = 1 + ⌈2n/m⌉` when shrinking, otherwise `T = 3`.
3. **Centre** of target `k`: `c_k = ⌊Δ/2⌋ − 256 σ + k Δ` ticks. (For `σ = 128` the offset is half a sample, for `σ = 64` a quarter sample.)
4. **First tap:** `a_k = trunc((c_k − 32768 (T − 2)) / 65536)`, rounding toward zero (this differs from the floor when the numerator is negative,
   at the start of a plane). The taps are the source indices `i = a_k, a_k + 1, …, a_k + T − 1`.
5. **Raw weight** of tap `i`: the distance `δ = 2^14 · |65536 i − c_k|` (in units of 2^-30 of a source sample); when shrinking, replace it by
   `⌊δ m / n⌋` (the triangle widens with the shrink factor, which is what makes the filter antialiased); then `ω_i = max(0, 2^30 − δ)`.
   This is a triangle (tent) of half-width one target sample when shrinking and one source sample otherwise.
6. **Edges.** A tap outside the plane gives its raw weight to the nearest edge sample: tap `i` credits source index `clamp(i, 0, n − 1)`.
   Let `Ω` be the sum of `ω_i` over all `T` taps (all of them, before or after crediting; it is the same).
7. **Quantisation.** Take the source indices `j` from `clamp(a_k, 0, n − 1)` to `clamp(a_k + T − 1, 0, n − 1)` in increasing order, each with the raw
   weight `ρ_j` it was credited (zero if none). Let `C_j` be the running sum of `ρ` up to and including `j`, and
   `Q_j = ⌊(2 N C_j + Ω) / (2 Ω)⌋`, that is `N · C_j / Ω` rounded to the nearest integer with halves up. The integer weight of `j` is
   `Q_j − Q_{j−1}`, with `Q` before the first index taken as 0. The running sum is rounded, not each weight, so the weights add up to `N` exactly;
   they are never negative.
8. **Length of the filter** `L(F)`: for each target `k`, the number of source indices from its first non-zero weight to its last non-zero weight
   inclusive; `L(F)` is the largest of these over all `k` (at least 1). Zero weights inside that span count. `L` is used only by the floor rule (section 6).

Examples (they follow from the rules): the identity map `F(n, n, N, 128)` has every weight `N` on the sample itself (`L = 1`);
`F(4, 8, 4096, 128)` gives the targets 0 to 7 the weights {0: 4096}, {0: 3072, 1: 1024}, {0: 1024, 1: 3072}, {1: 3072, 2: 1024},
{1: 1024, 2: 3072}, {2: 3072, 3: 1024}, {2: 1024, 3: 3072}, {3: 4096} (`L = 2`).

Golden lengths (luma horizontal, luma vertical, chroma horizontal, chroma vertical), as a check on an implementation of these rules:
2x cases (128 × 72 to 64 × 36) 4, 4, 4, 1; `nv12_bt709_limited_3x` 5, 5, 6, 3; `nv12_bt709_full_4x` 8, 8, 8, 4; `nv12_bt709_limited_2_4x` 5, 5, 5, 3;
`nv12_bt601_full_1_6x` 4, 4, 4, 2; `nv12_bt709_limited_odd_size` 5, 5, 5, 2; `i420_bt601_limited_3_1x` 7, 7, 7, 4; both `odd_enlarged` 2, 2, 2, 2;
both `luma_down_chroma_up` 3, 3, 3, 2; `i420_bt601_full_3x_centre` 5, 5, 5, 3.

*Pinned by:* every `scale` case except `i420_bt709_full_not_scaled`, and every `flat` and `pipeline` case. The edge rule (6) and the toward-zero
first tap (4) are pinned by the border pixels of every scaled case; the shrink widening (5) by the 3x, 4x, 2.4x and 3.1x cases.

## 6. Applying the filters to a plane, and the floor rule

A plane is resampled horizontally first, then vertically, with an integer intermediate:

1. **Horizontal pass**, for every source row `r` and target column `x`: `t(x, r) = min(32767, ⌊ (Σ_i w_x(i) · s(i, r)) / 2^7 ⌋)`, where `s` is the
   8-bit source sample and `w_x` the weights of the horizontal filter (sum 2^14). The intermediate thus carries the sample with 7 extra bits
   (15 bits in all; the cap at 32767 is never reached by 8-bit input).
2. **Vertical pass**, for every target row `y` and column `x`: `o(x, y) = min(255, ⌊ (β + Σ_r w'_y(r) · t(x, r)) / 2^19 ⌋)`, where `w'_y` are the
   weights of the vertical filter (sum 2^12), so the division by 2^19 removes both the 7 intermediate bits and the 12 weight bits.

The rounding term `β` is the **floor rule**: `β = 0` (the vertical pass floors) when **both** the luma vertical filter and the chroma vertical
filter have length `L = 2`; otherwise `β = 2^18` (the vertical pass rounds to nearest, halves up). The same `β` applies to all three planes.
In practice `β = 0` happens when the picture is enlarged vertically in both luma and chroma; a picture whose luma shrinks while its chroma grows
(for example 960 × 540 or 854 × 480 at 640 × 360, where only the chroma vertical filter has length 2) rounds in every plane.

*Pinned by:* `nv12_bt709_limited_odd_enlarged` and `i420_bt601_full_odd_enlarged` (639 × 359 to 640 × 360: the only golden cases with `β = 0`);
`nv12_bt709_limited_luma_down_chroma_up` and `i420_bt601_full_luma_down_chroma_up` (192 × 108 to 128 × 72: chroma vertical length 2, luma 3, so
`β = 2^18`); the test "an odd stored size of 640 or less is enlarged ... without a black edge" (the last row and column of a 639 × 359 picture of
luma 200 enlarged to 640 × 360 are filter output, not black: R + G + B above 300 at (639, 359), (639, 0), (0, 359), (320, 359), (639, 180)).

## 7. Colour conversion on the scaled path

For output pixel `(x, y)` let `Y = o_Y(x, y)`, `U = o_U(⌊x/2⌋, y)`, `V = o_V(⌊x/2⌋, y)` from the resampled planes.

**Shift functions.** For each of four chroma terms there is a slope `a` and an offset `b`, and the shift of a chroma value `c` (0 to 255) is the
integer `S(c) = ⌊a (c − 128) + b⌋`: `S_R` of V, `S_GU` of U, `S_GV` of V, `S_B` of U.

**Gray ramp.** `Γ(j)` maps an integer `j` (it lies between −320 and 575 for every input) to a byte:

- full range: `Γ(j) = clamp(j, 0, 255)`;
- limited range: `Γ(j) = clamp(⌊(255/219)(j − 16) − 1.2808⌋, 0, 255)`, with 255/219 the double nearest that ratio.

**The pixel.** `R = Γ(Y + S_R(V))`, `G = Γ(Y + S_GU(U) + S_GV(V))`, `B = Γ(Y + S_B(U))`. The shifts are added as integers before the ramp; the
only clamping is the ramp's.

**Matrix and range choice.** The slopes and offsets depend on the matrix (`kr`, `kb`) and the range. A matrix is recognised as BT.709 when
`kr = 0.2126` and `kb = 0.0722` exactly and as BT.601 when `kr = 0.299` and `kb = 0.114` exactly. For those, the parameters are these measured
values (the slopes are the matrix's own, BT.601 full-range blue differing from `2 (1 − kb)` in the sixth digit; the offsets are what the reference
output requires; with them all 256 values of each shift match):

| matrix, range | `S_R` (a, b) | `S_B` (a, b) | `S_GU` (a, b) | `S_GV` (a, b) | ramp |
|---|---|---|---|---|---|
| BT.709 full | 1.5748, 0.5746 | 1.8556, 0.5162 | −0.1873243, 0.0317 | −0.4681243, 0.081 | identity |
| BT.709 limited | 1.5396482, 0.0761 | 1.8141804, 0.2164 | −0.1831429, 0.5593 | −0.4576751, 0.419 | limited |
| BT.601 full | 1.402, 0.457 | 1.7719291, 0.8158 | −0.3441363, 0.9537 | −0.7141363, 0.6429 | identity |
| BT.601 limited | 1.3707054, 0.453 | 1.7324464, 0.7524 | −0.3364547, 0.9391 | −0.6981957, 0.6331 | limited |

The limited-range slopes are close to the full-range ones times 219/224 (the chroma scale 255/224 divided by the luma scale 255/219, which the ramp
applies), but the measured values in the table are the ones that hold (BT.601 limited blue differs in the fifth digit).
Over every chroma value 0 to 255, no value `a (c − 128) + b` in this table lies within 0.0002 of an integer, and no ramp argument within 0.0068,
so double precision decides nothing here.

Any other matrix (BT.2020 and anything else `matrixOf` returns) uses, with `kg = 1 − kr − kb`, `κ = 1` for full range and `κ = 219/224` for limited:
`a_R = 2 (1 − kr) κ`, `a_B = 2 (1 − kb) κ`, `a_GU = −(kb / kg) a_B`, `a_GV = −(kr / kg) a_R`, every offset `b = 0.5`, and the ramp of its range.
This branch is **not pinned** by any golden case and may be a level off the reference.

*Pinned by:* the four `flat` cases (`bt709`/`bt601` × full/limited, 300 random (Y, U, V) triples each: a flat 32 × 32 NV12 picture scaled to
16 × 16 must give exactly the recorded R, G, B at pixel (8, 8)); every scaled `scale` case; the limited ramp's top by the test "a limited-range
picture swscale does not scale" (a flat BT.709 limited 32 × 32 picture of Y = 235, U = V = 128 scaled to 16 × 16 gives 253, 253, 253, not 255).

## 8. The unscaled path (`W = w`, `H = h`)

No filter is applied: output pixel `(x, y)` uses the luma sample `Y = s_Y(x, y)` and the chroma samples `U = s_U(⌊x/2⌋, ⌊y/2⌋)`,
`V = s_V(⌊x/2⌋, ⌊y/2⌋)`, so each chroma sample covers a 2 × 2 block. Siting and the floor rule play no part. Let `a_R, a_B, a_GU, a_GV` be the slopes
of section 7 for the picture's matrix and range.

- **Full range.** The same form as section 7 with every offset replaced by one half: `R = clamp(Y + ⌊a_R (V − 128) + 1/2⌋, 0, 255)`,
  `G = clamp(Y + ⌊a_GU (U − 128) + 1/2⌋ + ⌊a_GV (V − 128) + 1/2⌋, 0, 255)`, `B = clamp(Y + ⌊a_B (U − 128) + 1/2⌋, 0, 255)`.
- **Limited range.** One rounded sum per channel. With `g = (255/219)(Y − 16)` and the effective slopes `ã = a · (255/219)` (the limited slope of
  the table times 255/219, both doubles, multiplied in double precision): `R = clamp(⌊g + ã_R (V − 128) + 0.2⌋, 0, 255)`,
  `G = clamp(⌊g + ã_GU (U − 128) + ã_GV (V − 128) + 0.2⌋, 0, 255)`, `B = clamp(⌊g + ã_B (U − 128) + 0.2⌋, 0, 255)`.
  Some arguments come within 1e-7 of an integer here, so the effective slopes must be formed exactly as stated (not, say, from 255/224 directly).

The reference itself converts some rows of an unscaled picture against the chroma row above, so this path is held to 2 levels on a picture with
smooth chroma (where that shift is invisible), not bit-exact; the limited-range reference changed between two runs of the same call, so no golden
picture holds it.

*Pinned by:* `i420_bt709_full_not_scaled` (64 × 48, smooth chroma, within 2 levels); the gray-ramp test (a BT.709 limited 4 × 4 NV12 picture with
U = V = 128 and flat Y gives, at pixel (0, 0): Y = 16 → 0, 0, 0; 17 → 1, 1, 1; 100 → 98, 98, 98; 235 → 255, 255, 255).

## 9. Tolerances the golden tests hold

- `flat`: exact, every triple.
- `scale`: each case within its `tol` (1 or 2 levels, the largest absolute difference over all output bytes); and, in a separate test, every
  `scale` case whose name contains none of `not_scaled`, `1_6x`, `odd_size` must be **bit-exact** (largest difference 0). That includes both
  `odd_enlarged` cases and both `luma_down_chroma_up` cases, whatever their `tol`.
- `pipeline`: the squashed picture within 1 level (section 11).

## 10. The golden cases

The synthetic planes (the `planes` field of the JSON, also in the test's `picture` helper): `Y` is the generator's LCG byte stream for `seed`;
`U` and `V` are `64 + ⌊b/2⌋` of the streams for `seed + 1` and `seed + 2` (so 64 to 191); with `smooth`, `U(i, r) = 96 + ⌊(2i + 3r)/4⌋` and
`V(i, r) = 160 − ⌊(3i + 2r)/4⌋` instead.

| case | format, matrix, range | size | rules it pins |
|---|---|---|---|
| `nv12_bt709_full_2x`, `nv12_bt709_limited_2x`, `nv12_bt601_full_2x`, `nv12_bt601_limited_2x` | NV12, all four | 128 × 72 → 64 × 36 | sections 4 to 7, both matrices and ranges |
| `i420_bt709_limited_2x`, `i420_bt709_full_2x` | I420, BT.709 | 128 × 72 → 64 × 36 | the I420 layout |
| `nv12_bt709_limited_3x`, `nv12_bt709_full_4x` | NV12, BT.709 | 192 × 108, 256 × 144 → 64 × 36 | whole-number shrink, wide filters |
| `nv12_bt709_limited_2_4x`, `i420_bt601_limited_3_1x` | | 96 × 64 → 40 × 26, 200 × 120 → 64 × 38 | ratios that are not whole |
| `nv12_bt601_full_1_6x` | | 64 × 48 → 40 × 30 | tol 2, not in the bit-exact test |
| `nv12_bt709_limited_odd_size` | | 130 × 74 → 64 × 36 | odd chroma counts; tol 2, not in the bit-exact test |
| `i420_bt709_full_not_scaled` | smooth chroma | 64 × 48 → 64 × 48 | section 8, tol 2 |
| `nv12_bt709_limited_odd_enlarged`, `i420_bt601_full_odd_enlarged` | | 639 × 359 → 640 × 360 | enlargement, odd planes, `β = 0`; bit-exact |
| `nv12_bt709_limited_luma_down_chroma_up`, `i420_bt601_full_luma_down_chroma_up` | | 192 × 108 → 128 × 72 | `β = 2^18` with chroma growing; bit-exact |
| `nv12_bt709_limited_2x_centre`, `i420_bt601_full_3x_centre` | siting `center` | 128 × 72, 192 × 108 → 64 × 36 | `σ = 128` for chroma |

## 11. Where the stage sits in the pipeline

`pipeline.ts` calls the stage with `(W, H) = scaledSize(w, h, 640)`, then turns the result upright and squashes it. The four `pipeline` golden
cases (`portrait_phone_2x`, `landscape_limited_3x`, `rotated_270_mirrored_i420`, `upside_down_4x`; 192 × 108 to 384 × 216 at a long side of 96, then
a 40 or 48 pixel square) hold the whole chain to 1 level, so they also pin this stage through sections 4 to 7.

## 12. Open questions

Questions the specification does not answer, and what was found out about them while the stage was written from it (by experiment against the golden
outputs only).

- **The sum of the raw weights when it is zero.** Section 5 divides by `Ω`. For every geometry the golden cases and the tests reach, `Ω` is positive
  (the tap nearest the centre is always within one source sample of it). The implementation gives the whole weight `N` to the first retained sample if `Ω`
  were ever zero; no case reaches it. Nothing was measured.
- **Intermediate overflow.** Sections 5 to 6 do not say how large the sums get. With the largest sizes the pipeline uses (a long side of 640 from
  sources of 4096 or less) the horizontal sums stay below 2^22 and the vertical sums below 2^27, so 32-bit integers hold them; the filter arithmetic of
  section 5 needs doubles (up to about 2^49), which hold it exactly. Larger sources were not tried [unverified].
- **Whether the tables are cached.** The spec is silent; the implementation caches the filters and colour tables by their parameters, which cannot change
  the output.
- **Chroma planes of an odd source.** Section 1 gives `cw = ceil(w/2)`; the cases `odd_size` and `odd_enlarged` agree with it, so no extra rule for the
  last chroma column or row was needed.
- **The generic matrix branch (section 7, last paragraph)** is not pinned by any golden case and was not tested against the reference; it follows the
  stated formula and may be a level off [unverified].
- **The unscaled path** was checked only on the smooth-chroma 64 x 48 golden picture (within 2 levels) and the gray ramp; the formulas of section 8 were
  used as stated, with no experiment beyond those two.
