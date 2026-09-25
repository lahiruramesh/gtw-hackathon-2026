import type { DomainId, Method, RequirementId } from "@/content/methods";

export interface RankedMethod {
  method: Method;
  /** Weighted mean score on the 1 to 5 scale. */
  score: number;
}

/** Weighted mean of a method's scores in one domain; all-zero weights fall back to equal weights. */
export function weightedScore(method: Method, domain: DomainId, weights: Record<RequirementId, number>): number {
  const scores = method.assessments[domain].scores;
  const entries = Object.entries(scores) as [RequirementId, number][];
  const total = entries.reduce((sum, [requirement]) => sum + weights[requirement], 0);
  if (total === 0) return entries.reduce((sum, [, score]) => sum + score, 0) / entries.length;
  return entries.reduce((sum, [requirement, score]) => sum + weights[requirement] * score, 0) / total;
}

export function rankMethods(
  methods: readonly Method[],
  domain: DomainId,
  weights: Record<RequirementId, number>,
): RankedMethod[] {
  return methods
    .map((method) => ({ method, score: weightedScore(method, domain, weights) }))
    .sort((a, b) => b.score - a.score || a.method.name.localeCompare(b.method.name));
}
