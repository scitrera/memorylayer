"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Database,
  Search,
  GitBranch,
  Route,
  Clock,
  Settings,
} from "lucide-react";
import { cn } from "@/lib/cn";
import { ConnectionStatus } from "./connection-status";

const navItems = [
  { href: "/", label: "Home", icon: LayoutDashboard },
  { href: "/memories", label: "Memories", icon: Database },
  { href: "/search", label: "Search", icon: Search },
  { href: "/graph", label: "Graph", icon: GitBranch },
  { href: "/walk", label: "Walk", icon: Route },
  { href: "/sessions", label: "Sessions", icon: Clock },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-64 flex-col border-r bg-card">
      <div className="flex items-baseline gap-1 px-4 py-5">
        <span className="text-lg font-semibold text-brand-600">MemoryLayer</span>
        <span className="text-sm text-slate-500">Explorer</span>
      </div>

      <nav className="flex-1 space-y-1 px-2">
        {navItems.map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                isActive
                  ? "bg-brand-50 text-brand-600"
                  : "text-muted-foreground hover:bg-accent hover:text-accent-foreground"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="border-t">
        <ConnectionStatus />
      </div>
    </aside>
  );
}
