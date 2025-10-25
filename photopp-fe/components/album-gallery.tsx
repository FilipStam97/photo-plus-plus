'use client';
import { useEffect, useState } from "react";
import { addToast } from "@heroui/toast";

export default function AlbumGallery({ folderName }: { folderName?: string }) {
  const [files, setFiles] = useState<{ url: string; originalName: string }[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);

    const fetchFiles = async () => {
      try {
        const res = await fetch(folderName 
          ? `/api/s3/presigned/${folderName}` 
          : `/api/s3/presigned` 
        );

        if (!res.ok) throw new Error("Failed to fetch files");

        const data = await res.json();

        if (Array.isArray(data)) {
          setFiles(data);
        } else if (Array.isArray(data.files)) {
          setFiles(data.files);
        } else {
          setFiles([]);
        }

        if (!data || data.length === 0) {
          addToast({
            title: "Info",
            description: "No files found",
            timeout: 3000,
          });
        }
      } catch (err) {
        console.error(err);
        addToast({
          title: "Error",
          description: "Something went wrong!",
          timeout: 3000,
          color: "danger",
        });
        setFiles([]);
      } finally {
        setLoading(false);
      }
    };

    fetchFiles();
  }, [folderName]);

  if (loading) {
    return <p>Loading files...</p>;
  }

  if (files.length === 0) {
    return <p>No files found.</p>;
  }

  return (
    <div className="grid grid-cols-4 gap-2">
      {files.map((file, idx) => (
        <img
          key={idx}
          src={file.url}
          alt={file.originalName || "Uploaded image"}
          className="h-full rounded-md object-cover"
        />
      ))}
    </div>
  );
}