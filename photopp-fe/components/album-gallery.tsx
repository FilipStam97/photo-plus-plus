'use client';
import { useEffect, useState } from "react";

export default function AlbumGallery({ folderName }: { folderName: string }) {
    const [files, setFiles] = useState<{ url: string; originalName: string }[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!folderName) return;
        setLoading(true);

        fetch(`/api/s3/presigned/${folderName}`)
            .then(res => {
                if (!res.ok) throw new Error("Failed to fetch files");
                return res.json();
            })
            .then((data) => {
                if (Array.isArray(data)) {
                    setFiles(data);
                } else if (Array.isArray(data.files)) {
                    setFiles(data.files);
                } else {
                    setFiles([]);
                }
            })
            .catch((err) => {
                console.error(err);
                setFiles([]);
            })
            .finally(() => setLoading(false));
    }, [folderName]);

    if (loading) return <p>Loading files...</p>;
    if (files.length === 0) {
        return <p>No files found in this folder.</p>;
    }

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {files.map((file, idx) => (
                <img
                    key={idx}
                    src={file.url}
                    alt={file.originalName}
                    className="h-full w-full object-cover rounded-md"
                />
            ))}
        </div>
    );
}
