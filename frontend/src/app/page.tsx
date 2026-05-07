import Link from "next/link";

const phases = [
  { id: 0, title: "Product Foundation — monorepo + frozen formulas", weeks: "0.1", status: "complete" },
  { id: 1, title: "Upload, Validation, Cleaning, MC Defense", weeks: "1.1, 1.2", status: "complete" },
  { id: 2, title: "Donor Pool, Labels, Features, Model Trust", weeks: "2.1–2.4", status: "complete" },
  { id: 3, title: "Predict, Portfolio Scoring, Decision Recommendations", weeks: "3.1–3.3", status: "complete" },
  { id: 4, title: "Drift Monitoring, Decision Simulator", weeks: "4.1, 4.2", status: "complete" },
];

const quickLinks: { href: "/upload" | "/donor-pool" | "/models" | "/predict" | "/portfolio" | "/decisions" | "/drift" | "/simulator"; label: string; hint: string }[] = [
  { href: "/upload", label: "1. Upload", hint: "drop a CSV → get an upload_id" },
  { href: "/donor-pool", label: "2. Donor Pool", hint: "promote RCTs into the training pool" },
  { href: "/models", label: "3. Model Trust", hint: "train + inspect ablation, hold-out" },
  { href: "/predict", label: "4. Predict", hint: "score a single campaign" },
  { href: "/portfolio", label: "5. Portfolio", hint: "score every non-RCT row in an upload" },
  { href: "/decisions", label: "6. Decisions", hint: "rank with risk gates" },
  { href: "/drift", label: "7. Drift", hint: "PSI vs. training distribution" },
  { href: "/simulator", label: "8. Simulator", hint: "reallocate budget" },
];

export default function Home() {
  return (
    <main className="container py-12">
      <header className="mb-12 border-b pb-8">
        <p className="text-sm uppercase tracking-widest text-muted-foreground">
          PIE Measurement Workbench
        </p>
        <h1 className="mt-2 text-4xl font-semibold tracking-tight">PIEmaker</h1>
        <p className="mt-3 max-w-2xl text-muted-foreground">
          Campaign-level incrementality prediction. Built on Gordon, Moakler &amp;
          Zettelmeyer (NBER w35044, April 2026). Trust before UX — formula
          contracts and the model trust layer ship before the prediction UI.
        </p>
      </header>

      <section className="mb-12">
        <h2 className="mb-4 text-xl font-medium">Build phases</h2>
        <ol className="space-y-3">
          {phases.map((phase) => (
            <li
              key={phase.id}
              className="flex items-center justify-between rounded-md border p-4"
            >
              <div>
                <p className="text-sm text-muted-foreground">
                  Phase {phase.id} · prompts {phase.weeks}
                </p>
                <p className="font-medium">{phase.title}</p>
              </div>
              <span className="rounded-full bg-emerald-500 px-3 py-1 text-xs font-medium text-white">
                {phase.status}
              </span>
            </li>
          ))}
        </ol>
      </section>

      <section className="mb-12">
        <h2 className="mb-4 text-xl font-medium">Click-through tour</h2>
        <p className="mb-4 text-sm text-muted-foreground">
          Suggested order: upload → donor pool → train a model → walk through
          predictions/decisions/drift/simulator.
        </p>
        <div className="grid gap-3 md:grid-cols-2">
          {quickLinks.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className="rounded-md border p-4 hover:bg-secondary/40"
            >
              <p className="font-medium">{link.label}</p>
              <p className="text-sm text-muted-foreground">{link.hint}</p>
            </Link>
          ))}
        </div>
      </section>

      <footer className="mt-16 border-t pt-6 text-sm text-muted-foreground">
        Backend health:{" "}
        <Link
          className="underline hover:text-foreground"
          href="/api/backend/health"
        >
          /api/backend/health
        </Link>
      </footer>
    </main>
  );
}
