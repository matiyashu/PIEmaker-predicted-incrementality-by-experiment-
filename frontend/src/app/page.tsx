import Link from "next/link";

const phases = [
  { id: 0, title: "Product Foundation", weeks: "Week 1", status: "in-progress" },
  { id: 1, title: "Upload, Validation, Cleaning", weeks: "Weeks 2–3", status: "pending" },
  { id: 2, title: "Donor Pool, Labels, Features, Model Lab", weeks: "Weeks 4–7", status: "pending" },
  { id: 3, title: "Prediction Workspace & Decision Simulator", weeks: "Weeks 8–10", status: "pending" },
  { id: 4, title: "Monitoring, Governance, Reporting", weeks: "Weeks 11–13", status: "pending" },
];

const modules = [
  ["A", "Executive Overview", "Leadership / client"],
  ["B", "Upload & Mapping Studio", "Analyst"],
  ["C", "Data Validation & Cleaning Workbench", "Analyst / data engineer"],
  ["D", "Donor Pool Manager", "Measurement lead"],
  ["E", "Feature Engineering Studio", "Data scientist"],
  ["F", "RCT Label Generator", "Data scientist"],
  ["G", "Model Lab", "Data scientist"],
  ["H", "Prediction Workspace", "Consultant / media lead"],
  ["I", "Decision Simulator", "Consultant / leadership"],
  ["J", "Monitoring & Governance", "ML / data team"],
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
                  Phase {phase.id} · {phase.weeks}
                </p>
                <p className="font-medium">{phase.title}</p>
              </div>
              <span
                className={`rounded-full px-3 py-1 text-xs font-medium ${
                  phase.status === "in-progress"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground"
                }`}
              >
                {phase.status}
              </span>
            </li>
          ))}
        </ol>
      </section>

      <section>
        <h2 className="mb-4 text-xl font-medium">Modules</h2>
        <div className="grid gap-3 md:grid-cols-2">
          {modules.map(([id, name, user]) => (
            <div key={id} className="rounded-md border p-4">
              <p className="text-xs uppercase text-muted-foreground">Module {id}</p>
              <p className="mt-1 font-medium">{name}</p>
              <p className="text-sm text-muted-foreground">{user}</p>
            </div>
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
