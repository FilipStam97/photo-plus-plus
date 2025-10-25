'use client';

import { useEffect, useState } from "react";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { addToast } from "@heroui/toast";
import FolderCard from "@/components/folder-card"; // komponenta za folder
import FolderDetail from "@/components/folder-detail";
import { Link } from "@heroui/link";
import NextLink from "next/link";


interface Folder {
  id: string;
  name: string;
}

export default function AlbumsPage() {
  const [folders, setFolders] = useState<Folder[]>([]);
  const [newFolderName, setNewFolderName] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedFolder, setSelectedFolder] = useState<null | typeof folders[0]>(null);

  const fetchFolders = async () => {
    try {
      const res = await fetch("/api/folder");
      if (!res.ok) return;
      const data = await res.json();
      setFolders(data);
    } catch (error) {
      console.error(error);
    }
  };

  useEffect(() => {
    fetchFolders();
  }, []);

  const handleCreateFolder = async () => {
    if (!newFolderName) return;
    setLoading(true);
    try {
      const res = await fetch("/api/folder", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newFolderName }),
      });
      if (!res.ok) {
        const text = await res.text();
        addToast({ title: "Error", description: text, color: "danger", timeout: 3000 });
      } else {
        const folder = await res.json();
        setFolders((prev) => [...prev, folder]);
        setNewFolderName("");
        addToast({ title: "Success", description: "Folder created", color: "success", timeout: 2000 });
      }
    } catch (error) {
      console.error(error);
      addToast({ title: "Error", description: "Unknown error", color: "danger", timeout: 3000 });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 flex flex-col gap-6">
      <div className="flex gap-2 items-center justify-end">
        <Input
          value={newFolderName}
          placeholder="New folder name"
          onChange={(e) => setNewFolderName(e.target.value)}
        />
        <Button onClick={handleCreateFolder} disabled={loading}>
          + New Folder
        </Button>
      </div>

      {/* Folders */}
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {folders.map((folder) => (
          <NextLink href={`/albums/${folder.name}`} key={folder.id}>
            <FolderCard folder={folder} />
          </NextLink>
        ))}
      </div>
    </div>
  );
}
