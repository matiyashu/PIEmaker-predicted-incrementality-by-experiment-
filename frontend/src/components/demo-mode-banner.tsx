"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Sparkles } from "lucide-react";

import { isDemoMode, setDemoMode, syncFromUrl } from "@/lib/demo-mode";

export function DemoModeBanner() {
  const router = useRouter();
  const [active, setActive] = useState(false);

  // Pick up ?demo=1 from URL on first mount, persist into localStorage,
  // then hydrate the active state. Re-checks on every pathname change so the
  // banner stays in sync if the user toggles the flag.
  useEffect(() => {
    syncFromUrl();
    setActive(isDemoMode());
    const onStorage = () => setActive(isDemoMode());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  if (!active) return null;

  const exit = () => {
    setDemoMode(false);
    setActive(false);
    router.refresh();
  };

  return (
    <div className="border-b border-amber-400/40 bg-amber-400/10">
      <div className="container flex items-center justify-between gap-3 py-2 text-xs">
        <div className="flex items-center gap-2 text-amber-900">
          <Sparkles className="h-3.5 w-3.5" />
          <span className="font-medium">Demo mode active</span>
          <span className="hidden text-muted-foreground sm:inline">
            · using mock data, no backend calls
          </span>
        </div>
        <button
          type="button"
          onClick={exit}
          className="rounded-md border border-amber-400/40 bg-background px-2 py-1 font-medium text-amber-900 hover:bg-amber-400/20"
        >
          Exit demo mode
        </button>
      </div>
    </div>
  );
}
