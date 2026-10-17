// About twenty hand-drawn 24x24 icons, inline (no icon package, and the build ships no png/jpg).
const PATHS: Record<string, string> = {
  play: 'M7 4.5v15l12-7.5z',
  pause: 'M8 5v14M16 5v14',
  stepBack: 'M6 5v14M19 5l-9 7 9 7z',
  stepFwd: 'M18 5v14M5 5l9 7-9 7z',
  skipPrev: 'M5 5v14M12 12l7-7v14zM12 12',
  skipNext: 'M19 5v14M12 12l-7-7v14z',
  folder: 'M3 7h6l2 2h10v10H3z',
  open: 'M4 6h6l2 2h8v10H4zM4 12h16',
  export: 'M12 15V4M7 9l5-5 5 5M5 15v5h14v-5',
  undo: 'M9 14L4 9l5-5M4 9h9a6 6 0 010 12h-3',
  redo: 'M15 14l5-5-5-5M20 9h-9a6 6 0 000 12h3',
  split: 'M12 3v18M7 8l-4 4 4 4M17 8l4 4-4 4',
  merge: 'M3 12h6m6 0h6M9 7l5 5-5 5M15 7l-5 5 5 5',
  trash: 'M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6',
  plus: 'M12 5v14M5 12h14',
  magnet: 'M5 4v8a7 7 0 0014 0V4h-4v8a3 3 0 01-6 0V4zM5 8h4M15 8h4',
  zoomIn: 'M10 4a6 6 0 100 12 6 6 0 000-12zM15 15l5 5M10 7v6M7 10h6',
  zoomOut: 'M10 4a6 6 0 100 12 6 6 0 000-12zM15 15l5 5M7 10h6',
  box: 'M5 5h14v14H5zM9 9h6v6H9z',
  scissors: 'M6 6a2.5 2.5 0 100 5 2.5 2.5 0 000-5zM6 13a2.5 2.5 0 100 5 2.5 2.5 0 000-5zM8 9l12 9M8 15L20 6',
  help: 'M12 3a9 9 0 100 18 9 9 0 000-18zM9.5 9.5a2.5 2.5 0 115 0c0 2-2.5 2-2.5 4M12 17v.5',
  close: 'M6 6l12 12M18 6L6 18',
  panelLeft: 'M4 5h16v14H4zM9 5v14',
  panelRight: 'M4 5h16v14H4zM15 5v14',
  volume: 'M4 9h4l5-4v14l-5-4H4zM16 9a4 4 0 010 6',
  up: 'M12 19V5M6 11l6-6 6 6',
  down: 'M12 5v14M6 13l6 6 6-6',
  warn: 'M12 4l9 16H3zM12 10v4M12 17v.5',
};

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  return (
    <svg class="i" width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d={PATHS[name]} />
    </svg>
  );
}
