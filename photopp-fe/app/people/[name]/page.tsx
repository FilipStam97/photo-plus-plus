'use client';

import { useParams } from "next/navigation";
import FolderDetail from "@/components/folder-detail";
import { useRouter } from "next/navigation";
import AlbumGallery from "@/components/album-gallery";

export default function PersonPage() {
  const params = useParams();
  const router = useRouter();

//   const folderNameParam = params.name;
//   const folderName = Array.isArray(folderNameParam) ? folderNameParam[0] : folderNameParam;

//   if (!folderName) {
//     return <div>Folder not found</div>;
//   }

  return (
    <>
      <section>
        HERE BE Person
        {/* <AlbumGallery folderName={folderName}/> */}
      </section>
    </>
  );
}