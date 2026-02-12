export type SimMode = 'life' | 'baseline';

// Simplest toggle point: change this and reload.
export const SIM_MODE: SimMode = 'baseline';

export const SIM_MODE_ID: Record<SimMode, number> = {
  life: 0,
  baseline: 1,
};

