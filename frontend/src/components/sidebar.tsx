"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Activity,
  BarChart3,
  Beaker,
  BookOpen,
  Brain,
  Database,
  GaugeCircle,
  LayoutDashboard,
  ListChecks,
  Microscope,
  ScanSearch,
  Settings,
  Sparkles,
  Sliders,
  Target,
  Upload,
  Wand2,
} from "lucide-react";

import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
}

interface NavGroup {
  group: string;
  items: NavItem[];
}

const groups: NavGroup[] = [
  {
    group: "Overview",
    items: [
      { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
    ],
  },
  {
    group: "Data",
    items: [
      { href: "/upload", label: "Upload", icon: Upload },
      { href: "/cleaning", label: "Cleaning", icon: Wand2 },
    ],
  },
  {
    group: "Trust",
    items: [
      { href: "/donor-pool", label: "Donor Pool", icon: Database },
      { href: "/labels", label: "Labels", icon: ListChecks },
      { href: "/features", label: "Features", icon: Beaker },
      { href: "/models", label: "Model Trust", icon: Brain },
    ],
  },
  {
    group: "Decisions",
    items: [
      { href: "/predict", label: "Predict", icon: Target },
      { href: "/portfolio", label: "Portfolio", icon: BarChart3 },
      { href: "/decisions", label: "Decisions", icon: Sparkles },
    ],
  },
  {
    group: "Operations",
    items: [
      { href: "/drift", label: "Drift", icon: ScanSearch },
      { href: "/simulator", label: "Simulator", icon: Sliders },
      { href: "/diagnostics", label: "Diagnostics", icon: Microscope },
    ],
  },
  {
    group: "Help",
    items: [
      { href: "/faq", label: "FAQ & glossary", icon: BookOpen },
      { href: "/settings", label: "Settings", icon: Settings },
    ],
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sticky top-0 hidden h-screen w-60 shrink-0 border-r bg-secondary/30 md:flex md:flex-col">
      <Link
        href="/dashboard"
        className="flex items-center gap-2 border-b px-5 py-4"
      >
        <span className="grid h-8 w-8 place-items-center rounded-md bg-primary text-primary-foreground">
          <GaugeCircle className="h-5 w-5" />
        </span>
        <div className="flex flex-col leading-tight">
          <span className="text-sm font-semibold">PIEmaker</span>
          <span className="text-xs text-muted-foreground">
            Measurement Workbench
          </span>
        </div>
      </Link>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {groups.map((group) => (
          <div key={group.group} className="mb-5">
            <p className="mb-1 px-2 text-[10px] font-medium uppercase tracking-widest text-muted-foreground">
              {group.group}
            </p>
            <ul className="space-y-0.5">
              {group.items.map((item) => {
                const active =
                  pathname === item.href ||
                  pathname?.startsWith(`${item.href}/`);
                const Icon = item.icon;
                return (
                  <li key={item.href}>
                    <Link
                      href={item.href}
                      className={cn(
                        "flex items-center gap-2.5 rounded-md px-2 py-1.5 text-sm transition-colors",
                        active
                          ? "bg-primary text-primary-foreground"
                          : "text-foreground/80 hover:bg-secondary",
                      )}
                    >
                      <Icon className="h-4 w-4 shrink-0" />
                      <span>{item.label}</span>
                    </Link>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      </nav>

      <div className="border-t px-5 py-4 text-xs text-muted-foreground">
        <p className="flex items-center gap-1.5">
          <Activity className="h-3 w-3" />
          Built by Prima Hanura Akbar
        </p>
        <p className="mt-1 text-[10px]">v0.1 · Build plan v3 complete</p>
      </div>
    </aside>
  );
}
