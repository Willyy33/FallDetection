interface SectionTitleProps {
  title: string;
  tag: string;
}

export default function SectionTitle({ title, tag }: SectionTitleProps) {
  return (
    <div className="mb-4 flex items-center justify-between">
      <h2 className="text-lg">{title}</h2>
      <span className="rounded-full bg-brand-200 px-2.5 py-1 text-xs font-semibold text-brand-700">
        {tag}
      </span>
    </div>
  );
}
