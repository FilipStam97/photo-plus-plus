"use client";

import UploadForm from "@/components/upload-form";
import { Button } from "@heroui/button";
import type { Dispatch, SetStateAction } from "react";

export interface Folder {
  name: string;
  createdAt?: string;
}

interface FolderDetailProps {
  folder: Folder;
  onBack: () => void;
  setTriggerFetch?: Dispatch<SetStateAction<boolean>>;
}

export default function FolderDetail({
  folder,
  onBack,
  setTriggerFetch,
}: FolderDetailProps) {
  return (
    <div className="p-4 w-full ">
      <Button onClick={onBack} className="flex-shrink-0 mb-6">
        Back to Albums
      </Button>
      <div
        className="flex items-center justify-between w-full mb-6 gap-4"
        style={{ alignItems: "flex-start" }}
      >
        <div className="flex-1">
          <UploadForm
            folderName={folder.name}
            setTriggerFetch={setTriggerFetch}
          />
        </div>
      </div>
    </div>
  );
}
