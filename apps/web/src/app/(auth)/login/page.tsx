import { BotIcon } from "lucide-react";
import type { Metadata } from "next";
import { redirect } from "next/navigation";

import { LoginForm } from "@/app/(auth)/login/login-form";
import { safeNextPath } from "@/lib/safe-redirect";
import { getViewer } from "@/lib/session";

export const metadata: Metadata = { title: "Sign in" };

export default async function LoginPage({ searchParams }: PageProps<"/login">) {
  const { next } = await searchParams;
  const destination = safeNextPath(typeof next === "string" ? next : null);
  if (await getViewer()) redirect(destination);

  return (
    <main className="flex min-h-svh items-center justify-center bg-muted/40 p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <div className="flex size-10 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <BotIcon className="size-5" aria-hidden />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">SKF Skill Studio</h1>
          <p className="text-sm text-muted-foreground">Sign in to train and release humanoid skills</p>
        </div>
        <LoginForm destination={destination} />
        <p className="text-center text-xs text-muted-foreground">No account? Ask an admin for an account.</p>
      </div>
    </main>
  );
}
