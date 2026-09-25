"use client";

import { ChevronRightIcon, RotateCcwIcon } from "lucide-react";
import { Fragment, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { DEFAULT_WEIGHTS, DOMAINS, METHODS, REQUIREMENTS, type DomainId, type RequirementId } from "@/content/methods";
import { formatNumber } from "@/lib/format";
import { rankMethods } from "@/lib/methods-ranking";
import { cn } from "cn";

const SCORE_CLASSES = ["", "bg-muted/30", "bg-muted/60", "bg-chart-1/15", "bg-chart-1/30", "bg-chart-1/45"];

function toggled(set: ReadonlySet<string>, id: string): Set<string> {
  const next = new Set(set);
  if (!next.delete(id)) next.add(id);
  return next;
}

export function MethodsMatrix() {
  const [domain, setDomain] = useState<DomainId>("locomotion");
  const [weights, setWeights] = useState<Record<RequirementId, number>>(DEFAULT_WEIGHTS);
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const ranking = rankMethods(METHODS, domain, weights);
  const best = ranking[0]?.score ?? 5;

  return (
    <div className="space-y-4">
      <Tabs value={domain} onValueChange={(value) => setDomain(value as DomainId)}>
        <TabsList>
          {DOMAINS.map((item) => (
            <TabsTrigger key={item.id} value={item.id}>
              {item.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* Side by side only where the matrix still fits all eight requirement columns. */}
      <div className="grid grid-cols-1 gap-4 2xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <Card className="gap-3">
          <CardHeader>
            <CardTitle className="text-sm">Scores (1 to 5, higher is better for SKF)</CardTitle>
            <CardDescription>
              Hover a requirement to see what a 5 means; open a method for the reasoning.
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="min-w-44">Method</TableHead>
                  {REQUIREMENTS.map((requirement) => (
                    <TableHead key={requirement.id} className="text-center">
                      <Tooltip>
                        <TooltipTrigger className="cursor-help underline decoration-dotted underline-offset-4">
                          {requirement.label}
                        </TooltipTrigger>
                        <TooltipContent>5 = {requirement.five}</TooltipContent>
                      </Tooltip>
                    </TableHead>
                  ))}
                </TableRow>
              </TableHeader>
              <TableBody>
                {METHODS.map((method) => {
                  const assessment = method.assessments[domain];
                  const open = expanded.has(method.id);
                  const rationaleId = `rationale-${method.id}`;
                  return (
                    <Fragment key={method.id}>
                      <TableRow className={cn(open && "border-b-0")}>
                        <TableCell className="align-top whitespace-normal">
                          <button
                            type="button"
                            className="flex items-start gap-1 text-left font-medium"
                            aria-expanded={open}
                            aria-controls={open ? rationaleId : undefined}
                            onClick={() => setExpanded((current) => toggled(current, method.id))}
                          >
                            <ChevronRightIcon
                              className={cn("mt-0.5 size-4 shrink-0 transition-transform", open && "rotate-90")}
                              aria-hidden
                            />
                            {method.name}
                          </button>
                        </TableCell>
                        {REQUIREMENTS.map((requirement) => {
                          const score = assessment.scores[requirement.id];
                          return (
                            <TableCell
                              key={requirement.id}
                              className={cn("text-center font-medium", SCORE_CLASSES[score])}
                            >
                              {score}
                            </TableCell>
                          );
                        })}
                      </TableRow>
                      {open && (
                        <TableRow id={rationaleId} className="hover:bg-transparent">
                          <TableCell
                            colSpan={REQUIREMENTS.length + 1}
                            className="pt-0 pl-7 text-sm whitespace-normal text-muted-foreground"
                          >
                            {assessment.rationale}
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  );
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>

        <Card className="gap-3">
          <CardHeader>
            <CardTitle className="text-sm">Ranking for your priorities</CardTitle>
            <CardDescription>Weighted mean score; updates as you move the weights.</CardDescription>
          </CardHeader>
          <CardContent>
            <ol className="space-y-2.5">
              {ranking.map(({ method, score }, index) => (
                <li key={method.id} className="space-y-1">
                  <div className="flex items-center justify-between gap-2 text-sm">
                    <span className="flex min-w-0 items-center gap-2">
                      <span className="tabular w-4 text-muted-foreground">{index + 1}</span>
                      <span className="truncate">{method.name}</span>
                      {index === 0 && <Badge variant="secondary">Best fit</Badge>}
                    </span>
                    <span className="tabular font-medium">{formatNumber(score, 2)}</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-muted" aria-hidden>
                    <div
                      className="h-full rounded-full bg-chart-1"
                      style={{ width: `${(score / Math.max(best, 1e-9)) * 100}%` }}
                    />
                  </div>
                </li>
              ))}
            </ol>
          </CardContent>
        </Card>
      </div>

      <Card className="gap-3">
        <CardHeader>
          <CardTitle className="text-sm">Weights</CardTitle>
          <CardDescription>How much each requirement matters, from 0 (ignore) to 5 (critical).</CardDescription>
          <CardAction>
            <Button variant="ghost" size="sm" onClick={() => setWeights(DEFAULT_WEIGHTS)}>
              <RotateCcwIcon /> Reset
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="grid grid-cols-1 gap-x-8 gap-y-4 sm:grid-cols-2 lg:grid-cols-4">
          {REQUIREMENTS.map((requirement) => (
            <div key={requirement.id} className="space-y-2">
              <div className="flex items-center justify-between">
                <Label id={`weight-${requirement.id}`}>{requirement.label}</Label>
                <span className="tabular text-sm text-muted-foreground">{weights[requirement.id]}</span>
              </div>
              <Slider
                aria-labelledby={`weight-${requirement.id}`}
                min={0}
                max={5}
                step={1}
                value={[weights[requirement.id]]}
                onValueChange={([value]) => setWeights((current) => ({ ...current, [requirement.id]: value ?? 0 }))}
              />
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
