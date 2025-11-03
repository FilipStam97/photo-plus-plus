"use client";

import { useParams } from "next/navigation";
import FolderDetail from "@/components/folder-detail";
import { useRouter } from "next/navigation";
import AlbumGallery from "@/components/album-gallery";
import { useState } from "react";

export default function FolderDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [triggerFetch, setTriggerFetch] = useState(true);

  const folderNameParam = params.name;
  const folderName = Array.isArray(folderNameParam)
    ? folderNameParam[0]
    : folderNameParam;

  if (!folderName) {
    return <div>Folder not found</div>;
  }

  return (
    <>
      <div className="p-4">
        <FolderDetail
          folder={{ name: folderName }}
          onBack={() => router.push("/albums")}
          setTriggerFetch={setTriggerFetch}
        />
      </div>
      <section>
        <AlbumGallery folderName={folderName} triggerFetch={triggerFetch} />
      </section>
    </>
  );
}
