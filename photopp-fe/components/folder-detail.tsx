'use client';

import UploadForm from "@/components/upload-form";
import { Button } from "@heroui/button";

export interface Folder {
    name: string;
    createdAt?: string;
}

interface FolderDetailProps {
    folder: Folder;
    onBack: () => void;
}

export default function FolderDetail({ folder, onBack }: FolderDetailProps) {
    return (
        <div className="p-4 w-full ">
            <div className="flex items-center justify-between w-full mb-6 gap-4" style={{ alignItems: 'flex-start' }}>
                <div className="flex-1">
                    <UploadForm folderName={folder.name} />
                </div>

                <Button
                    onClick={onBack}
                    className="flex-shrink-0"
                >
                    Back to Albums
                </Button>
            </div>
        </div>
    );
}