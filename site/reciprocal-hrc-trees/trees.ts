// Column trees. A tree splits the columns [0, root) recursively: a node of length L splits into a first piece of
// length cut and a second of length L - cut. A node holds rays of every rise |d| ≤ L. A ray crosses its node's cut at
// row cut·d / L and is the mix of two chains, through the row below that point (weight 1 - f) and the row above
// (weight f), f = frac(cut·d / L): the linear rule.
//
// The -x and -y frustums run the mirrored tree, which is the same tree in world columns, so the merge stays the
// transpose of the extensions.

export interface Tree {
  name: string;
  /** the frame width; the scene sits at its start and the rest is opaque */
  root: number;
  /** length → [first piece, second piece] */
  pieces: Map<number, [number, number]>;
}

// the family's splits up to its first length ≥ N
function family(name: string, N: number, grow: (pieces: Map<number, [number, number]>) => number) {
  const pieces = new Map<number, [number, number]>();
  let root = 1;
  while (root < N) root = grow(pieces);
  return { name, root, pieces } satisfies Tree;
}

/** lengths 2^k, halved */
export function dyadic(N: number): Tree {
  let L = 1;
  return family('dyadic', N, (p) => (p.set(2 * L, [L, L]), (L *= 2)));
}

/** Fibonacci lengths; F_k splits into F_{k-1}, F_{k-2} */
export function fibonacci(N: number): Tree {
  let a = 1, b = 1;
  return family('Fibonacci', N, (p) => {
    [a, b] = [b, a + b];
    p.set(b, [a, b - a]);
    return b;
  });
}

/** silver-ratio lengths 1, 2, 3, 5, 7, 12, 17, 29, …: after Pell numbers P' < P come H = P + P' (split P, P') and
 * P + H (split H, P) */
export function silver(N: number): Tree {
  let small = 1, large = 2, next: number[] = [2];
  return family('silver', N, (p) => {
    if (next.length === 0) {
      const h = large + small;
      p.set(h, [large, small]);
      p.set(h + large, [h, large]);
      [small, large] = [large, h + large];
      next = [h, large];
    }
    if (p.size === 0) p.set(2, [1, 1]);
    return next.shift()!;
  });
}

export const TREES = { dyadic, fibonacci, silver } as const;

export function mirror(tree: Tree): Tree {
  return { ...tree, name: `${tree.name} (mirrored)`, pieces: new Map([...tree.pieces].map(([L, [a, b]]) => [L, [b, a]])) };
}

/** A node as the GPU reads it. Ids index the frustum's node list; -1 means off the scene, or (as parent) the root. */
export interface Node {
  column: number;
  length: number;
  /** first slot of its rays (-1 for 1-px nodes) */
  base: number;
  first: number;
  second: number;
  parent: number;
  parentLength: number;
  /** the parent's cut */
  cut: number;
  /** 0 first piece, 1 second */
  piece: number;
  sibling: number;
}
export const NODE_FIELDS = 10;

export interface Level {
  length: number;
  cut: number;
  /** its nodes are nodeStart … nodeStart + nodeCount - 1, by column */
  nodeStart: number;
  nodeCount: number;
  /** first slot of its rays */
  base: number;
}

/** One frustum's nodes. Nodes outside the scene's columns [sceneStart, sceneStart + N) are dropped, and so is the root.
 * A node of length ≥ 2 stores (2L + 1) N rays. */
export interface Layout {
  tree: Tree;
  N: number;
  sceneStart: number;
  nodes: Node[];
  /** shortest first */
  levels: Level[];
  /** stored rays */
  slots: number;
}

export function layout(tree: Tree, N: number, sceneStart: number): Layout {
  const { root, pieces } = tree;
  interface Raw { column: number; length: number; parent: Raw | null; piece: number; children: (Raw | null)[]; id: number }
  const overlaps = (column: number, length: number) => column < sceneStart + N && column + length > sceneStart;
  const all: Raw[] = [];
  const visit = (column: number, length: number, parent: Raw | null, piece: number): Raw | null => {
    if (!overlaps(column, length)) return null;
    const node: Raw = { column, length, parent, piece, children: [], id: -1 };
    if (length > 1) {
      const split = pieces.get(length);
      if (!split) throw new Error(`tree ${tree.name}: no split for length ${length}`);
      node.children = [visit(column, split[0], node, 0), visit(column + split[0], split[1], node, 1)];
    }
    all.push(node);
    return node;
  };
  const rootSplit = pieces.get(root)!;
  const top = { column: 0, length: root, parent: null, piece: 0, children: [], id: -1 } as Raw;
  top.children = [visit(0, rootSplit[0], top, 0), visit(rootSplit[0], rootSplit[1], top, 1)];

  all.sort((a, b) => a.length - b.length || a.column - b.column);
  all.forEach((n, i) => (n.id = i));
  const levels: Level[] = [];
  let slots = 0;
  for (const n of all) {
    let level = levels.at(-1);
    if (!level || level.length !== n.length) {
      level = { length: n.length, cut: n.length > 1 ? pieces.get(n.length)![0] : 0, nodeStart: n.id, nodeCount: 0, base: slots };
      levels.push(level);
    }
    level.nodeCount++;
    if (n.length > 1) slots += (2 * n.length + 1) * N;
  }
  const id = (n: Raw | null | undefined) => n?.id ?? -1;
  const nodes = all.map((n): Node => {
    const level = levels.find((l) => l.length === n.length)!;
    const parent = n.parent!;
    return {
      column: n.column,
      length: n.length,
      base: n.length > 1 ? level.base + (n.id - level.nodeStart) * (2 * n.length + 1) * N : -1,
      first: id(n.children[0]),
      second: id(n.children[1]),
      parent: parent === top ? -1 : parent.id,
      parentLength: parent.length,
      cut: pieces.get(parent.length)![0],
      piece: n.piece,
      sibling: id(parent.children[1 - n.piece]),
    };
  });
  return { tree, N, sceneStart, nodes, levels, slots };
}

/** frustums 0 and 2 run the tree with the scene at its start; 1 and 3 the mirrored tree, the scene at its end */
export function frustumLayouts(tree: Tree, N: number): [Layout, Layout] {
  return [layout(tree, N, 0), layout(mirror(tree), N, tree.root - N)];
}

/** the linear rule: the crossing row's floor q0, and f, the weight of row q0 + 1 */
export function split(length: number, cut: number, rise: number) {
  const q0 = Math.floor((cut * rise) / length);
  return { q0, f: (cut * rise - q0 * length) / length };
}

/** One frustum's work, per scene pixel. */
export interface Counts {
  tree: string;
  N: number;
  root: number;
  /** stored lengths */
  levels: number;
  /** dispatches per frustum */
  passes: number;
  /** rays at every length, root included */
  rays: number;
  storedRays: number;
  /** chains joined by the extensions */
  chains: number;
  extensionReads: number;
  /** merge reads with Wu */
  mergeReads: number;
  mergeReadsPlain: number;
}

export function counts(tree: Tree, N: number): Counts {
  const lay = layout(tree, N, 0), pixels = N * N;
  let rays = 0, stored = 0, chains = 0, extensionReads = 0, mergeReads = 0, mergeReadsPlain = 0;
  for (const l of lay.levels) rays += l.nodeCount * (2 * l.length + 1) * N;
  rays += (2 * tree.root + 1) * N;
  for (const node of lay.nodes) {
    const L = node.length;
    if (L > 1) {
      stored += (2 * L + 1) * N;
      const cut = tree.pieces.get(L)![0];
      for (let d = -L; d <= L; d++) {
        const c = split(L, cut, d).f > 0 ? 2 : 1;
        chains += c * N;
        extensionReads += 2 * c * N;
      }
    }
    // the merge into this node's rays, as the shader's gather loops
    const P = node.parentLength, cut = node.cut, piece = node.piece === 0 ? cut : P - cut;
    for (let a = -L; a <= L; a++) {
      const lo = Math.max(-P, Math.floor(((a - 1) * P) / piece) + 1), hi = Math.min(P, Math.ceil(((a + 1) * P) / piece) - 1);
      for (let d = lo; d <= hi; d++) {
        const s = split(P, cut, d), q = node.piece === 0 ? a : d - a;
        const w = (q === s.q0 ? 1 - s.f : 0) + (q === s.q0 + 1 ? s.f : 0);
        if (w === 0) continue;
        const own = node.parent < 0 ? 0 : 1;
        mergeReads += (own * (s.f > 0 ? 3 : 1) + node.piece) * N;
        mergeReadsPlain += (own + node.piece) * N;
      }
    }
  }
  return {
    tree: tree.name, N, root: tree.root,
    levels: lay.levels.length - 1,
    passes: 2 * (lay.levels.length - 1) + 1,
    rays: rays / pixels,
    storedRays: stored / pixels,
    chains: chains / pixels,
    extensionReads: extensionReads / pixels,
    mergeReads: mergeReads / pixels,
    mergeReadsPlain: mergeReadsPlain / pixels,
  };
}
