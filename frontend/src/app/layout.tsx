import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "PIEmaker — PIE Measurement Workbench",
  description:
    "Campaign-level incrementality prediction platform built on Gordon, Moakler & Zettelmeyer (NBER w35044, 2026).",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-foreground antialiased">
        {children}
      </body>
    </html>
  );
}
