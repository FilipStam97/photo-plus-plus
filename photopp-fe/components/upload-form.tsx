"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@heroui/button";
import { Input } from "@heroui/input";
import { addToast } from "@heroui/toast";
import type { Dispatch, SetStateAction } from "react";
import {
  getPresignedUrls,
  handleUpload,
  MAX_FILE_SIZE_NEXTJS_ROUTE,
  validateFiles,
} from "./../app/_shared/client/minio";

interface UploadFormProps {
  folderName?: string;
  setTriggerFetch?: Dispatch<SetStateAction<boolean>>;
}

export default function UploadForm({
  folderName,
  setTriggerFetch,
}: UploadFormProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [loading, setLoading] = useState(false);
  const [isDragging, setIsDragging] = useState(false);

  const inputRef = useRef<HTMLInputElement>(null);

  // ---- helpers
  const onPickClick = () => inputRef.current?.click();

  const onFilesSelected = (fileList: FileList | null) => {
    if (!fileList) return;
    const incoming = Array.from(fileList).filter((f) =>
      /image\/(png|jpeg|bmp)/.test(f.type)
    );
    if (incoming.length === 0) return;

    // validacija (koristi tvoju postojeću)
    const shortFileProps = incoming.map((f) => ({
      originalFileName: f.name,
      fileSize: f.size,
    }));
    const error = validateFiles(shortFileProps, MAX_FILE_SIZE_NEXTJS_ROUTE);
    if (error) {
      addToast({
        title: "Nevalidni fajlovi",
        description: error,
        timeout: 3500,
        color: "danger",
      });
      return;
    }

    // spoji nove sa postojećima, izbegni duplikate po imenu+veličini
    setFiles((prev) => {
      const key = (f: File) => `${f.name}-${f.size}`;
      const existing = new Set(prev.map(key));
      const merged = [
        ...prev,
        ...incoming.filter((f) => !existing.has(key(f))),
      ];
      return merged;
    });
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    onFilesSelected(e.target.files);

  // drag & drop
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    onFilesSelected(e.dataTransfer.files);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) setIsDragging(true);
  };

  const onDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  // thumbnails (URL.createObjectURL cleanup)
  const previews = useMemo(
    () => files.map((f) => ({ file: f, url: URL.createObjectURL(f) })),
    [files]
  );
  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.url));
  }, [previews]);

  const removeFile = (name: string, size: number) =>
    setFiles((prev) =>
      prev.filter((f) => !(f.name === name && f.size === size))
    );

  const clearAll = () => {
    setFiles([]);
    if (inputRef.current) inputRef.current.value = "";
  };

  const totalSizeMB = useMemo(
    () => (files.reduce((s, f) => s + f.size, 0) / (1024 * 1024)).toFixed(2),
    [files]
  );

  // ---- submit
  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;

    try {
      setLoading(true);

      const shortFileProps = files.map((file) => ({
        originalFileName: file.name,
        fileSize: file.size,
      }));

      const presignedUrls = await getPresignedUrls(shortFileProps, folderName);

      await handleUpload(files, presignedUrls, () => {
        addToast({
          title: "Uspešno",
          description: "Fajlovi su otpremljeni.",
          timeout: 3000,
          color: "success",
        });
        clearAll();
        setTriggerFetch?.((p) => !p);
      });
    } catch (err) {
      console.error(err);
      addToast({
        title: "Greška",
        description: "Došlo je do nepoznate greške pri uploadu.",
        timeout: 3500,
        color: "danger",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="w-full max-w-xl mx-auto">
      {/* Dropzone */}
      <div
        onClick={onPickClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        aria-label="Izaberi ili prevuci slike"
        className={[
          "flex flex-col items-center justify-center gap-2",
          "rounded-2xl border-2 border-dashed bg-neutral-900/60",
          "px-6 py-8 text-center transition",
          isDragging
            ? "border-blue-500 bg-blue-500/10"
            : "border-neutral-700 hover:border-neutral-500",
        ].join(" ")}
      >
        <div className="text-sm text-neutral-300">
          {isDragging
            ? "Pusti fajlove ovde"
            : "Prevuci slike ili klikni za izbor"}
        </div>
        <div className="text-xs text-neutral-400">
          Podržano: PNG, JPG, JPEG, BMP
          {/* • max{" "}{Math.round(MAX_FILE_SIZE_NEXTJS_ROUTE / (1024 * 1024))}MB po fajlu */}
        </div>

        {/* Skriveni <input type="file" /> */}
        <Input
          ref={inputRef}
          type="file"
          multiple
          accept="image/png,image/jpeg,image/bmp"
          onChange={handleFileInputChange}
          className="hidden"
        />
        <Button size="sm" variant="flat" onPress={onPickClick}>
          Izaberi fajlove
        </Button>
      </div>

      {/* Info bar */}
      <div className="mt-3 flex items-center justify-between text-sm">
        <div className="text-neutral-400">
          {files.length > 0
            ? `${files.length} fajl(a) • ${totalSizeMB} MB ukupno`
            : "Nijedan fajl nije izabran"}
        </div>
        {files.length > 0 && (
          <Button size="sm" variant="light" onPress={clearAll}>
            Očisti
          </Button>
        )}
      </div>

      {/* Previews grid */}
      {files.length > 0 && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {previews.map(({ file, url }) => (
            <div
              key={`${file.name}-${file.size}`}
              className="relative overflow-hidden rounded-xl border border-neutral-800"
            >
              <img
                src={url}
                alt={file.name}
                className="h-32 w-full object-cover"
              />
              <div className="absolute inset-x-0 bottom-0 bg-black/50 px-2 py-1">
                <p className="truncate text-[11px] text-neutral-200">
                  {file.name}
                </p>
              </div>
              <button
                type="button"
                onClick={() => removeFile(file.name, file.size)}
                className="absolute right-2 top-2 rounded-full bg-black/60 px-2 py-1 text-xs text-white hover:bg-black/80"
                aria-label="Ukloni fajl"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Submit */}
      <Button
        type="submit"
        disabled={loading || files.length === 0}
        className="mt-4 w-full"
      >
        {loading ? "Uploading..." : "Upload"}
      </Button>
    </form>
  );
}
