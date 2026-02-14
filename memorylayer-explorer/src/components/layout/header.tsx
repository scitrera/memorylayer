"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Search, Menu } from "lucide-react";
import { WorkspaceSwitcher } from "@/components/layout/workspace-switcher";

interface HeaderProps {
  onMenuToggle?: () => void;
}

const routeLabels: Record<string, string> = {
  "/": "Dashboard",
  "/memories": "Memories",
  "/search": "Search",
  "/graph": "Graph",
  "/walk": "Relationship Walker",
  "/sessions": "Sessions",
  "/settings": "Settings",
};

function getBreadcrumbs(pathname: string): { label: string; href: string }[] {
  const segments = pathname.split("/").filter(Boolean);
  if (segments.length === 0) return [{ label: "Dashboard", href: "/" }];

  const crumbs: { label: string; href: string }[] = [];
  let currentPath = "";
  for (const segment of segments) {
    currentPath += `/${segment}`;
    const label = routeLabels[currentPath] || decodeURIComponent(segment);
    crumbs.push({ label, href: currentPath });
  }
  return crumbs;
}

export function Header({ onMenuToggle }: HeaderProps) {
  const pathname = usePathname();
  const breadcrumbs = getBreadcrumbs(pathname);

  return (
    <header className="flex h-14 items-center justify-between border-b bg-card px-4">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuToggle}
          className="rounded-md p-1.5 text-muted-foreground hover:bg-accent md:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>

        <nav className="flex items-center gap-1 text-sm">
          {breadcrumbs.map((crumb, i) => (
            <span key={crumb.href} className="flex items-center gap-1">
              {i > 0 && (
                <span className="text-muted-foreground">/</span>
              )}
              <Link
                href={crumb.href}
                className={
                  i === breadcrumbs.length - 1
                    ? "font-medium text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }
              >
                {crumb.label}
              </Link>
            </span>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-3">
        <WorkspaceSwitcher />
        <Link
          href="/search"
          className="rounded-md p-1.5 text-muted-foreground hover:bg-accent"
        >
          <Search className="h-4 w-4" />
        </Link>
      </div>
    </header>
  );
}
