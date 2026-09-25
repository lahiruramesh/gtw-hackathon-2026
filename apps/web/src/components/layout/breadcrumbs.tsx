"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { createContext, Fragment, useCallback, useContext, useEffect, useState, type ReactNode } from "react";

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { SEGMENT_LABELS } from "@/components/layout/nav";

type LabelSetter = (segment: string, label: string) => void;

const LabelsContext = createContext<Record<string, string>>({});
const SetLabelContext = createContext<LabelSetter>(() => {});

export function BreadcrumbProvider({ children }: { children: ReactNode }) {
  const [labels, setLabels] = useState<Record<string, string>>({});
  const setLabel = useCallback<LabelSetter>(
    (segment, label) =>
      setLabels((current) => (current[segment] === label ? current : { ...current, [segment]: label })),
    [],
  );
  return (
    <SetLabelContext.Provider value={setLabel}>
      <LabelsContext.Provider value={labels}>{children}</LabelsContext.Provider>
    </SetLabelContext.Provider>
  );
}

/** Lets a page name its dynamic URL segment (a run id becomes the run name). Renders nothing. */
export function BreadcrumbLabel({ segment, label }: { segment: string; label: string }) {
  const setLabel = useContext(SetLabelContext);
  useEffect(() => setLabel(segment, label), [setLabel, segment, label]);
  return null;
}

export function Breadcrumbs() {
  const pathname = usePathname();
  const labels = useContext(LabelsContext);
  const segments = pathname.split("/").filter(Boolean);

  const crumbs = segments.map((segment, index) => ({
    href: `/${segments.slice(0, index + 1).join("/")}`,
    label: labels[segment] ?? SEGMENT_LABELS[segment] ?? "…",
    linkable: segment !== "admin",
  }));

  return (
    <Breadcrumb className="min-w-0">
      <BreadcrumbList className="flex-nowrap">
        <BreadcrumbItem className={crumbs.length > 0 ? "hidden sm:inline-flex" : undefined}>
          {crumbs.length === 0 ? (
            <BreadcrumbPage>Dashboard</BreadcrumbPage>
          ) : (
            <BreadcrumbLink asChild>
              <Link href="/">Studio</Link>
            </BreadcrumbLink>
          )}
        </BreadcrumbItem>
        {crumbs.map((crumb, index) => {
          const last = index === crumbs.length - 1;
          return (
            <Fragment key={crumb.href}>
              <BreadcrumbSeparator className={index === 0 ? "hidden sm:list-item" : undefined} />
              <BreadcrumbItem className="min-w-0">
                {last || !crumb.linkable ? (
                  <BreadcrumbPage className="truncate">{crumb.label}</BreadcrumbPage>
                ) : (
                  <BreadcrumbLink asChild>
                    <Link href={crumb.href}>{crumb.label}</Link>
                  </BreadcrumbLink>
                )}
              </BreadcrumbItem>
            </Fragment>
          );
        })}
      </BreadcrumbList>
    </Breadcrumb>
  );
}
