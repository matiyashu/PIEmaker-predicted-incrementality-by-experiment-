import type { Metadata } from "next";
import "./globals.css";

import { Nav } from "@/components/nav";

export const metadata: Metadata = {
  title: "PIEmaker — PIE Measurement Workbench",
  description:
    "Campaign-level incrementality prediction platform built on Gordon, Moakler & Zettelmeyer (NBER w35044, 2026).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        <Nav />
        {children}
      </body>
    </html>
  );
}
