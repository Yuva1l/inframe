// What a stream says about its colours, read from the bytes the demux already holds (the avcC / hvcC SPS and the container's colr box), so the
// detector's picture is converted the way the Mac converts it. Chrome's VideoFrame.colorSpace cannot tell a stream that names no matrix from one
// that names bt709 (it reports bt709 for both, open-GOP and bt709 parity notes), but FFmpeg and swscale can: an unnamed matrix is BT.601, limited range, and the
// chroma of an h264 / hevc stream without siting is tagged LEFT, while VP9 and AV1 frames carry no siting and swscale centres them.

export type MatrixName = 'bt709' | 'bt2020-ncl' | 'smpte170m';

export interface StreamColour {
  /** the matrix the stream names (a VideoColorSpace name); null when this code does not read the stream's own tags, so the frame's is kept */
  matrix: MatrixName | null;
  /** limited range unless the stream says full; null when the stream's tags are not read */
  fullRange: boolean | null;
  /** where the chroma of a row sits horizontally, as swscale places it for the frame FFmpeg's decoder gives */
  chromaLoc: 'left' | 'center';
}

/** The stream's tags before anything is known of it: nothing named, so the frame's own colour space stands, and the chroma is centred. */
export const NO_STREAM_COLOUR: StreamColour = { matrix: null, fullRange: null, chromaLoc: 'center' };

/** ITU-T H.273 matrix_coefficients to the matrix swscale converts with. Unspecified (2), identity (0) and the rarer ones convert as BT.601, swscale's default. */
function matrixOfCode(code: number): MatrixName {
  if (code === 1) return 'bt709';
  if (code === 9 || code === 10) return 'bt2020-ncl';
  return 'smpte170m';
}

// ---------------------------------------------------------------- bits
class Bits {
  private pos = 0;
  constructor(private b: Uint8Array) {}
  get left(): number {
    return this.b.length * 8 - this.pos;
  }
  u(n: number): number {
    if (n > this.left) throw new RangeError('SPS ends early');
    let v = 0;
    for (let i = 0; i < n; i++) {
      v = v * 2 + ((this.b[this.pos >> 3] >> (7 - (this.pos & 7))) & 1);
      this.pos++;
    }
    return v;
  }
  skip(n: number): void {
    this.u(n);
  }
  flag(): boolean {
    return this.u(1) === 1;
  }
  ue(): number {
    let zeros = 0;
    while (this.u(1) === 0) if (++zeros > 32) throw new RangeError('bad exp-golomb');
    return zeros ? 2 ** zeros - 1 + this.u(zeros) : 0;
  }
  se(): number {
    const k = this.ue();
    return k % 2 ? (k + 1) / 2 : -k / 2;
  }
}

/** The payload of a NAL unit with its emulation prevention bytes (00 00 03) taken out. */
function unescape(nal: Uint8Array, skip: number): Uint8Array {
  const out = new Uint8Array(nal.length);
  let n = 0;
  let zeros = 0;
  for (let i = skip; i < nal.length; i++) {
    const c = nal[i];
    if (zeros >= 2 && c === 3) {
      zeros = 0;
      continue;
    }
    zeros = c === 0 ? zeros + 1 : 0;
    out[n++] = c;
  }
  return out.subarray(0, n);
}

interface Vui {
  /** vui_parameters_present_flag: an SPS without a VUI names nothing at all, and FFmpeg then leaves the chroma siting unspecified (swscale centres it) */
  present: boolean;
  /** matrix_coefficients when colour_description_present_flag is set, else null */
  matrix: number | null;
  /** video_full_range_flag when video_signal_type_present_flag is set, else null */
  fullRange: boolean | null;
  /** chroma_sample_loc_type_top_field when chroma_loc_info_present_flag is set, else null */
  chromaLocType: number | null;
}

const NO_VUI: Vui = { present: false, matrix: null, fullRange: null, chromaLocType: null };

/** The fields of vui_parameters() that H.264 and HEVC share up to chroma_loc_info (the HEVC and H.264 syntax agree on them). */
function vuiOf(r: Bits): Vui {
  if (r.flag()) {
    if (r.u(8) === 255) r.skip(32); // aspect_ratio_idc EXTENDED_SAR: sar_width, sar_height
  }
  if (r.flag()) r.skip(1); // overscan_appropriate_flag
  const vui: Vui = { present: true, matrix: null, fullRange: null, chromaLocType: null };
  if (r.flag()) {
    r.skip(3); // video_format
    vui.fullRange = r.flag();
    if (r.flag()) {
      r.skip(16); // colour_primaries, transfer_characteristics
      vui.matrix = r.u(8);
    }
  }
  if (r.flag()) {
    vui.chromaLocType = r.ue();
    r.ue();
  }
  return vui;
}

function skipScalingList(r: Bits, size: number): void {
  let last = 8;
  let next = 8;
  for (let j = 0; j < size; j++) {
    if (next !== 0) {
      next = (last + r.se() + 256) % 256;
    }
    last = next === 0 ? last : next;
  }
}

/** vui_parameters() of the first SPS in an avcC record, or null when the SPS is absent or cannot be read. */
function h264Vui(avcC: Uint8Array): Vui | null {
  if (avcC.length < 8 || (avcC[5] & 0x1f) < 1) return null;
  const len = (avcC[6] << 8) | avcC[7];
  const nal = avcC.subarray(8, 8 + len);
  if (nal.length < 4) return null;
  const r = new Bits(unescape(nal, 1));
  const profile = r.u(8);
  r.skip(16); // constraint flags, level_idc
  r.ue(); // seq_parameter_set_id
  if ([100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135].includes(profile)) {
    const chroma = r.ue();
    if (chroma === 3) r.skip(1);
    r.ue();
    r.ue();
    r.skip(1);
    if (r.flag()) {
      for (let i = 0; i < (chroma !== 3 ? 8 : 12); i++) if (r.flag()) skipScalingList(r, i < 6 ? 16 : 64);
    }
  }
  r.ue(); // log2_max_frame_num_minus4
  const pocType = r.ue();
  if (pocType === 0) r.ue();
  else if (pocType === 1) {
    r.skip(1);
    r.se();
    r.se();
    for (let i = r.ue(); i > 0; i--) r.se();
  }
  r.ue(); // max_num_ref_frames
  r.skip(1);
  r.ue();
  r.ue();
  if (!r.flag()) r.skip(1); // frame_mbs_only_flag = 0: mb_adaptive_frame_field_flag
  r.skip(1); // direct_8x8_inference_flag
  if (r.flag()) for (let i = 0; i < 4; i++) r.ue(); // frame cropping
  if (!r.flag()) return NO_VUI; // vui_parameters_present_flag
  return vuiOf(r);
}

/** profile_tier_level() of an HEVC SPS with profilePresentFlag = 1. */
function skipProfileTierLevel(r: Bits, maxSubLayersMinus1: number): void {
  r.skip(88); // general profile (2+1+5, 32 compat flags, 4 flags, 43 reserved bits, 1)
  r.skip(8); // general_level_idc
  const profilePresent: boolean[] = [];
  const levelPresent: boolean[] = [];
  for (let i = 0; i < maxSubLayersMinus1; i++) {
    profilePresent.push(r.flag());
    levelPresent.push(r.flag());
  }
  if (maxSubLayersMinus1 > 0) for (let i = maxSubLayersMinus1; i < 8; i++) r.skip(2);
  for (let i = 0; i < maxSubLayersMinus1; i++) {
    if (profilePresent[i]) r.skip(88);
    if (levelPresent[i]) r.skip(8);
  }
}

function skipHevcScalingListData(r: Bits): void {
  for (let sizeId = 0; sizeId < 4; sizeId++) {
    for (let matrixId = 0; matrixId < 6; matrixId += sizeId === 3 ? 3 : 1) {
      if (!r.flag()) r.ue(); // scaling_list_pred_matrix_id_delta
      else {
        const coefNum = Math.min(64, 1 << (4 + (sizeId << 1)));
        if (sizeId > 1) r.se();
        for (let i = 0; i < coefNum; i++) r.se();
      }
    }
  }
}

/** vui_parameters() of the first SPS in an hvcC record, or null when the SPS is absent or cannot be read. */
function hevcVui(hvcC: Uint8Array): Vui | null {
  if (hvcC.length < 23) return null;
  let p = 22;
  let nal: Uint8Array | null = null;
  for (let a = hvcC[p++]; a > 0 && !nal; a--) {
    const type = hvcC[p] & 0x3f;
    const count = (hvcC[p + 1] << 8) | hvcC[p + 2];
    p += 3;
    for (let k = 0; k < count; k++) {
      const len = (hvcC[p] << 8) | hvcC[p + 1];
      if (type === 33 && !nal) nal = hvcC.subarray(p + 2, p + 2 + len);
      p += 2 + len;
    }
  }
  if (!nal || nal.length < 6) return null;
  const r = new Bits(unescape(nal, 2));
  r.skip(4); // sps_video_parameter_set_id
  const maxSub = r.u(3);
  r.skip(1);
  skipProfileTierLevel(r, maxSub);
  r.ue(); // sps_seq_parameter_set_id
  if (r.ue() === 3) r.skip(1); // chroma_format_idc, separate_colour_plane_flag
  r.ue();
  r.ue();
  if (r.flag()) for (let i = 0; i < 4; i++) r.ue(); // conformance window
  r.ue();
  r.ue();
  const pocBits = r.ue() + 4;
  for (let i = r.flag() ? 0 : maxSub; i <= maxSub; i++) {
    r.ue();
    r.ue();
    r.ue();
  }
  for (let i = 0; i < 6; i++) r.ue(); // coding / transform block sizes and hierarchy depths
  if (r.flag() && r.flag()) skipHevcScalingListData(r); // scaling_list_enabled_flag, sps_scaling_list_data_present_flag
  r.skip(2); // amp, sao
  if (r.flag()) {
    r.skip(8);
    r.ue();
    r.ue();
    r.skip(1);
  }
  const sets = r.ue();
  const numDelta: number[] = [];
  for (let i = 0; i < sets; i++) {
    if (i !== 0 && r.flag()) {
      // inter_ref_pic_set_prediction_flag: the set is predicted from the one before it
      r.skip(1);
      r.ue();
      let n = 0;
      for (let j = 0; j <= numDelta[i - 1]; j++) {
        const used = r.flag();
        if (used || r.flag()) n++;
      }
      numDelta.push(n);
    } else {
      const neg = r.ue();
      const pos = r.ue();
      for (let j = 0; j < neg + pos; j++) {
        r.ue();
        r.skip(1);
      }
      numDelta.push(neg + pos);
    }
  }
  if (r.flag()) {
    for (let i = r.ue(); i > 0; i--) {
      r.skip(pocBits);
      r.skip(1);
    }
  }
  r.skip(2); // sps_temporal_mvp_enabled_flag, strong_intra_smoothing_enabled_flag
  if (!r.flag()) return NO_VUI; // vui_parameters_present_flag
  return vuiOf(r);
}

/** The vui of a stream's first SPS (null when there is none or it cannot be read: the caller then keeps the frame's tags). */
export function vuiOfDescription(codec: string, description: Uint8Array | null): Vui | null {
  if (!description || (codec !== 'h264' && codec !== 'hevc')) return null;
  try {
    return codec === 'h264' ? h264Vui(description) : hevcVui(description);
  } catch {
    return null;
  }
}

/** The container's colour information (a colr box of type nclx on the sample entry), if it has one. */
export interface Colr {
  matrix_coefficients: number;
  full_range: boolean;
}

/** What swscale will see for this stream. h264 and hevc: the SPS's tags, else the container's, else nothing named (BT.601, limited range). Chroma: a VUI that names no siting is LEFT (how FFmpeg's decoders tag it),
 *  an SPS with no VUI at all is unspecified, which swscale centres; a siting the VUI names is read from it (the horizontal part:
 *  types 0, 2 and 4 are left, 1, 3 and 5 centre). Any other codec: the container's tags if present (else the frame's own), chroma centred. */
export function streamColour(codec: string, description: Uint8Array | null, colr: Colr | null = null): StreamColour {
  const avc = codec === 'h264' || codec === 'hevc';
  if (!avc) return { matrix: colr ? matrixOfCode(colr.matrix_coefficients) : null, fullRange: colr ? colr.full_range : null, chromaLoc: 'center' };
  const vui = vuiOfDescription(codec, description);
  if (!vui) {
    // the SPS could not be read: what the container says, else leave the colour to the frame
    return { matrix: colr ? matrixOfCode(colr.matrix_coefficients) : null, fullRange: colr ? colr.full_range : null, chromaLoc: 'left' };
  }
  return {
    matrix: matrixOfCode(vui.matrix ?? colr?.matrix_coefficients ?? 2),
    fullRange: vui.fullRange ?? colr?.full_range ?? false,
    chromaLoc: !vui.present ? 'center' : vui.chromaLocType === null ? 'left' : vui.chromaLocType % 2 === 0 ? 'left' : 'center',
  };
}

/** The colour a frame is converted with: the stream's tags where it has them, else what the decoder reported for the frame. */
export function pictureColour(frame: { matrix?: string | null; fullRange?: boolean | null } | null | undefined, s: StreamColour = NO_STREAM_COLOUR): { matrix: string | null; fullRange: boolean; chromaLoc: 'left' | 'center' } {
  return { matrix: s.matrix ?? frame?.matrix ?? null, fullRange: s.fullRange ?? frame?.fullRange === true, chromaLoc: s.chromaLoc };
}
