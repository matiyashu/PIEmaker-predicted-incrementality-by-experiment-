import { Info, Lightbulb } from "lucide-react";
import { cn } from "@/lib/utils";

interface SummaryCardProps {
  title: string;
  body: string | React.ReactNode;
  recommendations?: (string | React.ReactNode)[];
  tone?: "info" | "warning" | "success";
}

const TONE_STYLES: Record<NonNullable<SummaryCardProps["tone"]>, string> = {
  info: "border-blue-200 bg-blue-50/60 text-blue-900",
  warning: "border-amber-300 bg-amber-50/60 text-amber-900",
  success: "border-emerald-200 bg-emerald-50/60 text-emerald-900",
};

const TONE_ICON_BG: Record<NonNullable<SummaryCardProps["tone"]>, string> = {
  info: "bg-blue-200/60 text-blue-700",
  warning: "bg-amber-200/60 text-amber-800",
  success: "bg-emerald-200/60 text-emerald-700",
};

/**
 * Reusable explainer card. Sits at the top of every workbench page to give
 * non-technical viewers a one-paragraph summary of what they're looking at,
 * plus optional bullet-point recommendations.
 */
export function SummaryCard({
  title,
  body,
  recommendations,
  tone = "info",
}: SummaryCardProps) {
  return (
    <div
      className={cn(
        "mb-6 rounded-lg border p-5 shadow-sm",
        TONE_STYLES[tone],
      )}
    >
      <div className="flex items-start gap-3">
        <span
          className={cn(
            "grid h-8 w-8 shrink-0 place-items-center rounded-md",
            TONE_ICON_BG[tone],
          )}
        >
          <Info className="h-4 w-4" />
        </span>
        <div className="flex-1">
          <p className="font-medium">{title}</p>
          <div className="mt-1.5 text-sm leading-relaxed">
            {typeof body === "string" ? <p>{body}</p> : body}
          </div>
          {recommendations && recommendations.length > 0 && (
            <div className="mt-3 rounded-md border border-current/20 bg-background/50 p-3">
              <p className="mb-1.5 flex items-center gap-1.5 text-xs font-medium uppercase tracking-wider opacity-80">
                <Lightbulb className="h-3 w-3" />
                What to do next
              </p>
              <ul className="space-y-1 text-sm">
                {recommendations.map((rec, i) => (
                  <li key={i} className="flex gap-2">
                    <span className="mt-1 block h-1 w-1 shrink-0 rounded-full bg-current opacity-60" />
                    <span>{rec}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
