export interface LineageInput<T> {
  id: string;
  parentId: string | null;
  value: T;
}

export interface LineageNode<T> {
  id: string;
  value: T;
  children: LineageNode<T>[];
}

/**
 * Builds warm-start trees. Runs whose parent is outside the input become roots, so a partial
 * list still renders. Input order is kept among siblings.
 */
export function buildLineage<T>(items: readonly LineageInput<T>[]): LineageNode<T>[] {
  const nodes = new Map(
    items.map((item) => [item.id, { id: item.id, value: item.value, children: [] as LineageNode<T>[] }]),
  );
  const roots: LineageNode<T>[] = [];
  for (const item of items) {
    const node = nodes.get(item.id)!;
    const parent = item.parentId ? nodes.get(item.parentId) : undefined;
    if (parent && parent !== node) parent.children.push(node);
    else roots.push(node);
  }
  return roots;
}
