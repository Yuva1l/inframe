// The only place file time meets player time (docs/T9A_DESIGN.md 2.4): fileTime = video.currentTime + clockOffset.
// clockOffset = (file time of the first frame) - (the player's mediaTime of its first frame).

/** firstFrameFileTime is `info.offset` for the original, the proxy's own value for a proxy; m0 is the first rVFC mediaTime after seeking to 0. */
export function clockOffset(firstFrameFileTime: number, m0: number): number {
  return firstFrameFileTime - m0;
}

export function toFileTime(currentTime: number, offset: number): number {
  return currentTime + offset;
}

export function toMediaTime(fileTime: number, offset: number): number {
  return fileTime - offset;
}
