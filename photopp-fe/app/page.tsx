"use client";
import AlbumGallery from "@/components/album-gallery";
import UploadForm from "@/components/upload-form";
import UploadGallery from "@/components/upload-gallery";
import { useState } from "react";

export default function Home() {
  const [triggerFetch, setTriggerFetch] = useState(true);
  return (
    <>
      <section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10">
        <UploadForm setTriggerFetch={setTriggerFetch} />
      </section>
      <section>
        <AlbumGallery triggerFetch={triggerFetch} />
        {/* <UploadGallery /> */}
      </section>
    </>
  );
}
