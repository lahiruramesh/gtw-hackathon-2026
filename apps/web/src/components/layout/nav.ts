import {
  BookOpenCheckIcon,
  BoxesIcon,
  CheckCheckIcon,
  CpuIcon,
  GitCompareIcon,
  LayoutDashboardIcon,
  PlayIcon,
  ScrollTextIcon,
  UsersIcon,
  type LucideIcon,
} from "lucide-react";

import type { Permission } from "@/lib/permissions";

export interface NavItem {
  href: string;
  label: string;
  icon: LucideIcon;
  /** Shown when the viewer has any of these permissions. */
  anyOf: readonly Permission[];
}

export interface NavSection {
  label: string;
  items: readonly NavItem[];
}

export const NAV_SECTIONS: readonly NavSection[] = [
  {
    label: "Training",
    items: [
      { href: "/", label: "Dashboard", icon: LayoutDashboardIcon, anyOf: ["run:read"] },
      { href: "/skills", label: "Skills", icon: BoxesIcon, anyOf: ["skill:read"] },
      { href: "/runs", label: "Runs", icon: PlayIcon, anyOf: ["run:read"] },
      { href: "/compare", label: "Compare", icon: GitCompareIcon, anyOf: ["run:read"] },
      { href: "/approvals", label: "Approvals", icon: CheckCheckIcon, anyOf: ["release:review", "run:approve_launch"] },
    ],
  },
  {
    label: "Analysis",
    items: [{ href: "/methods", label: "Learning methods", icon: BookOpenCheckIcon, anyOf: ["skill:read"] }],
  },
  {
    label: "Platform",
    items: [
      { href: "/compute", label: "Compute", icon: CpuIcon, anyOf: ["compute:read"] },
      { href: "/admin/users", label: "Users", icon: UsersIcon, anyOf: ["user:manage"] },
      { href: "/admin/audit", label: "Audit log", icon: ScrollTextIcon, anyOf: ["audit:read"] },
    ],
  },
];

/** Static labels for breadcrumb segments; dynamic segments are labelled by their page. */
export const SEGMENT_LABELS: Record<string, string> = {
  skills: "Skills",
  runs: "Runs",
  new: "New run",
  compare: "Compare",
  approvals: "Approvals",
  methods: "Learning methods",
  compute: "Compute",
  admin: "Admin",
  users: "Users",
  audit: "Audit log",
};

export function isActive(pathname: string, href: string): boolean {
  if (href === "/") return pathname === "/";
  return pathname === href || pathname.startsWith(`${href}/`);
}
