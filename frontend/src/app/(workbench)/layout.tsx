import { DemoModeBanner } from "@/components/demo-mode-banner";
import { Sidebar } from "@/components/sidebar";

export default function WorkbenchLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 min-w-0 flex flex-col">
        <DemoModeBanner />
        <div className="flex-1">{children}</div>
      </div>
    </div>
  );
}
