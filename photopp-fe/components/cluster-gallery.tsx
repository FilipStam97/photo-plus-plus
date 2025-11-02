// components/cluster-gallery.tsx
"use client";
import { useEffect, useState } from "react";
import { addToast } from "@heroui/toast";

type ImageFace = {
  point_id: string | number;
  bbox: { x: number; y: number; width: number; height: number };
};

type ImageItem = {
  image_name: string;
  faces: ImageFace[];
};

type ApiResponse = {
  total_images: number;
  images: ImageItem[];
};

export default function ClusterGallery({ clusterId }: { clusterId: number }) {
  const [files, setFiles] = useState<{ url: string; originalName: string }[]>(
    []
  );
  const [loading, setLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);

  useEffect(() => {
    const fetchAll = async () => {
      setLoading(true);
      try {
        const res = await fetch("/api/people", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ cluster_id: clusterId }),
        });
        if (!res.ok) throw new Error(`API error: ${res.status}`);
        const data: ApiResponse = await res.json();

        const fileNames = data.images.map((i) => i.image_name);
        const query = new URLSearchParams();
        fileNames.forEach((name) => query.append("fileNames", name));

        const res2 = await fetch(`/api/s3/presigned?${query.toString()}`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
        if (!res2.ok) throw new Error(`API error: ${res2.status}`);
        const data2: ApiResponse = await res2.json();
        setFiles(Array.isArray(data2) ? data2 : []);
      } catch (err) {
        console.error(err);
        addToast({
          title: "Error",
          description: "Failed to load cluster images",
          color: "danger",
        });
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, []);

  if (loading) return <p>Loading faces...</p>;
  if (!files || files.length === 0) return <p>No faces found.</p>;

  return (
    <>
      <div className="grid grid-cols-4 gap-2">
        {files.map((file, idx) => (
          <img
            key={idx}
            src={file.url}
            alt={file.originalName || "Uploaded image"}
            className="h-full rounded-md object-cover cursor-pointer hover:scale-105 transition-transform duration-150"
            onClick={() => setSelectedImage(file.url)}
          />
        ))}
      </div>
      {selectedImage && (
        <div
          className="fixed inset-0 flex items-center justify-center z-80 bg-black/50 cursor-pointer"
          onClick={() => setSelectedImage(null)}
        >
          <div
            className="p-4 rounded-lg shadow-lg bg-white"
            onClick={(e) => e.stopPropagation()}
          >
            <img
              src={selectedImage}
              alt="Selected"
              className="max-h-[90vh] max-w-[90vw] rounded-md"
            />
          </div>
        </div>
      )}
    </>
  );
}
