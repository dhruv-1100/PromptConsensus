import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ConsensusPrompt — Human-Centred Prompt Optimisation",
  description:
    "Multi-agent middleware that transparently optimises your prompts while keeping you in control. Built on human-centred AI principles.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
