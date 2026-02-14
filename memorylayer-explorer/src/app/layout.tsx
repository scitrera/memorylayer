import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono, Instrument_Serif } from "next/font/google";
import { Toaster } from "sonner";
import { QueryProvider } from "@/providers/query-provider";
import { ConnectionProvider } from "@/providers/connection-provider";
import { AppShell } from "@/components/layout/app-shell";
import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-instrument-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MemoryLayer Explorer",
  description: "Web UI for exploring MemoryLayer memories and graph relationships",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${dmSans.variable} ${jetbrainsMono.variable} ${instrumentSerif.variable}`}
    >
      <body className="font-sans antialiased">
        <QueryProvider>
          <ConnectionProvider>
            <AppShell>{children}</AppShell>
            <Toaster position="bottom-right" />
          </ConnectionProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
