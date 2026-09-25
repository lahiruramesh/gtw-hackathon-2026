import { cookies } from "next/headers";

import { AppSidebar } from "@/components/layout/app-sidebar";
import { BreadcrumbProvider } from "@/components/layout/breadcrumbs";
import { QueryProvider } from "@/components/layout/query-provider";
import { TopBar } from "@/components/layout/top-bar";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { requireViewer } from "@/lib/session";

export default async function AppLayout({ children }: LayoutProps<"/">) {
  const viewer = await requireViewer();
  const sidebarOpen = (await cookies()).get("sidebar_state")?.value !== "false";

  return (
    <QueryProvider>
      <SidebarProvider defaultOpen={sidebarOpen}>
        <BreadcrumbProvider>
          <AppSidebar viewer={viewer} />
          <SidebarInset className="min-w-0">
            <TopBar viewer={viewer} />
            <div className="mx-auto w-full max-w-[1400px] flex-1 px-4 py-5 sm:px-6">{children}</div>
          </SidebarInset>
        </BreadcrumbProvider>
      </SidebarProvider>
    </QueryProvider>
  );
}
