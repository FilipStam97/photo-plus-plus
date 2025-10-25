'use client';

interface FolderProps {
  folder: { id: string; name: string };
}

export default function FolderCard({ folder }: FolderProps) {
  return (
    <div
      className="flex flex-col items-center justify-center h-40 w-full rounded-2xl border border-gray-200 bg-gradient-to-b from-gray-50 to-white shadow-sm hover:shadow-lg hover:-translate-y-1 transition-all cursor-pointer"
    >
      <div className="flex flex-col items-center justify-center px-4 text-center">
        <div className="text-xl font-semibold text-gray-800">{folder.name}</div>
        <div className="mt-2 text-sm text-gray-500">Click to open</div>
      </div>
    </div>
  );
}