import type { ReactNode } from "react";

interface CardProps {
  children: ReactNode;
  className?: string;
}

export default function Card({ children, className = "" }: CardProps) {
  return (
    <div
      className={`mb-5 rounded-card border border-line bg-surface px-[22px] py-5 shadow-card ${className}`}
    >
      {children}
    </div>
  );
}
