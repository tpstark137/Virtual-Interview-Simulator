import { Inter } from "next/font/google";
import Image from "next/image";
import "./globals.css";
import { ClerkProvider } from "@clerk/nextjs";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "AI Mock Interview Simulator",
  description: "AI-powered technical interview practice for Machine Learning",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({ children }) {
  return (
    <ClerkProvider>
      <html lang="en" suppressHydrationWarning>
        <body className={inter.className}>{children}</body>
      </html>
    </ClerkProvider>
  );
}
