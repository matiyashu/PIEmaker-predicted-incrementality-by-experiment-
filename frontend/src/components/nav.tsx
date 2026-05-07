import Link from "next/link";

const links = [
  { href: "/", label: "Overview" },
  { href: "/upload", label: "Upload" },
  { href: "/cleaning", label: "Cleaning" },
  { href: "/donor-pool", label: "Donor Pool" },
  { href: "/labels", label: "Labels" },
  { href: "/features", label: "Features" },
  { href: "/models", label: "Model Trust" },
  { href: "/predict", label: "Predict" },
  { href: "/portfolio", label: "Portfolio" },
  { href: "/decisions", label: "Decisions" },
  { href: "/drift", label: "Drift" },
];

export function Nav() {
  return (
    <nav className="border-b">
      <div className="container flex h-14 items-center gap-6">
        <Link href="/" className="font-semibold">
          PIEmaker
        </Link>
        <ul className="flex gap-4 text-sm text-muted-foreground">
          {links.map((l) => (
            <li key={l.href}>
              <Link href={l.href} className="hover:text-foreground">
                {l.label}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
}
