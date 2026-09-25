import { FlaskConicalIcon } from "lucide-react";
import type { Metadata } from "next";
import Link from "next/link";

import { NoAccess } from "@/components/common/no-access";
import { PageHeader } from "@/components/common/page-header";
import { MethodsMatrix } from "@/components/methods/methods-matrix";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { EVIDENCE, RECOMMENDATION } from "@/content/methods";
import { requireViewer } from "@/lib/session";
import { can } from "@/lib/viewer";

export const metadata: Metadata = { title: "Learning methods" };

export default async function MethodsPage() {
  const viewer = await requireViewer();
  if (!can(viewer, "skill:read")) return <NoAccess what="view the methods analysis" />;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Learning methods"
        description="How each way of teaching a humanoid a skill scores against SKF's industrial requirements."
      >
        <Badge variant="outline" className="mt-1">
          Team assessment
        </Badge>
      </PageHeader>

      <Alert>
        <FlaskConicalIcon />
        <AlertDescription>
          Scores are the team&apos;s judgement. Only the locomotion column is backed by measurements from this project
          (evidence below); manipulation and workflow scores come from published results and should be revisited once
          SKF runs its own pilots.
        </AlertDescription>
      </Alert>

      <MethodsMatrix />

      <section className="space-y-3">
        <div>
          <h2 className="text-base font-semibold">Evidence from this project</h2>
          <p className="text-sm text-muted-foreground">
            Measured on the Unitree G1 in simulation; numbers from the repository.
          </p>
        </div>
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {EVIDENCE.map((item) => (
            <Card key={item.title} className="gap-3">
              <CardHeader>
                <CardTitle className="text-sm">{item.title}</CardTitle>
                <CardDescription>{item.method}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <ul className="list-disc space-y-1 pl-5 text-sm">
                  {item.facts.map((fact) => (
                    <li key={fact}>{fact}</li>
                  ))}
                </ul>
                <p className="text-sm font-medium">{item.takeaway}</p>
              </CardContent>
              <CardFooter className="mt-auto flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                {item.links.map((link) => (
                  <Link key={link.href} href={link.href} className="text-foreground underline-offset-4 hover:underline">
                    {link.label}
                  </Link>
                ))}
                <span className="flex min-w-0 flex-wrap gap-x-2">
                  Source:
                  {item.sources.map((source) => (
                    <code key={source} className="font-mono break-all">
                      {source}
                    </code>
                  ))}
                </span>
              </CardFooter>
            </Card>
          ))}
        </div>
      </section>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Recommendation</CardTitle>
        </CardHeader>
        <CardContent className="text-sm">{RECOMMENDATION}</CardContent>
      </Card>
    </div>
  );
}
