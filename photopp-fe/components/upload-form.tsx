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

  const onPickClick = () => inputRef.current?.click();

  const onFilesSelected = (fileList: FileList | null) => {
    if (!fileList) return;
    const incoming = Array.from(fileList).filter((f) =>
      /image\/(png|jpeg|bmp)/.test(f.type)
    );
    if (incoming.length === 0) return;

    // const shortFileProps = incoming.map((f) => ({
    //   originalFileName: f.name,
    //   fileSize: f.size,
    // }));
    // const error = validateFiles(shortFileProps, MAX_FILE_SIZE_NEXTJS_ROUTE);
    // if (error) {
    //   addToast({
    //     title: "Invalid files",
    //     description: error,
    //     timeout: 3500,
    //     color: "danger",
    //   });
    //   return;
    // }
    setFiles((prev) => {
      const key = (f: File) => `${f.name}-${f.size}`;
      const existing = new Set(prev.map(key));
      return [...prev, ...incoming.filter((f) => !existing.has(key(f)))];
    });
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) =>
    onFilesSelected(e.target.files);

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

  const previews = useMemo(
    () => files.map((f) => ({ file: f, url: URL.createObjectURL(f) })),
    [files]
  );
  useEffect(
    () => () => previews.forEach((p) => URL.revokeObjectURL(p.url)),
    [previews]
  );

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

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;
    try {
      setLoading(true);
      const shortFileProps = files.map((file) => ({
        originalFileName: file.name,
        fileSize: file.size,
      }));
      console.log(shortFileProps);
      const presignedUrls = await getPresignedUrls(shortFileProps, folderName);
      await handleUpload(files, presignedUrls, () => {
        addToast({
          title: "Upload complete",
          description: "Files uploaded successfully.",
          timeout: 3000,
          color: "success",
        });
        clearAll();
        setTriggerFetch?.((p) => !p);
      });
    } catch (err) {
      console.error(err);
      addToast({
        title: "Upload error",
        description: "An unexpected error occurred while uploading.",
        timeout: 3500,
        color: "danger",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={onSubmit} className="w-full max-w-xl mx-auto">
      <div
        onClick={onPickClick}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        aria-label="Select or drag images"
        className={[
          "flex flex-col items-center justify-center gap-2 px-6 py-8 text-center transition rounded-2xl border-2 border-dashed",
          "bg-gray-50 border-gray-300 text-gray-700 shadow-sm",
          "dark:bg-neutral-900 dark:border-neutral-700 dark:text-gray-200 dark:shadow-none",
          isDragging
            ? "border-blue-400 bg-blue-50 dark:bg-blue-950/30 dark:border-blue-500"
            : "hover:border-gray-400 dark:hover:border-neutral-600",
        ].join(" ")}
      >
        <div className="text-sm font-medium">
          {isDragging
            ? "Drop files here"
            : "Drag and drop images or click to select"}
        </div>
        <div className="text-xs text-gray-500 dark:text-gray-500">
          Supported: PNG, JPG, JPEG, BMP
        </div>

        <Input
          ref={inputRef}
          type="file"
          multiple
          accept="image/png,image/jpeg,image/bmp"
          onChange={handleFileInputChange}
          className="hidden"
        />

        <Button
          size="sm"
          onPress={onPickClick}
          className="px-3 py-1 rounded-lg text-sm font-medium
                     bg-gray-200 text-gray-800 hover:bg-gray-300
                     dark:bg-neutral-700 dark:text-gray-100 dark:hover:bg-neutral-600"
        >
          Choose Files
        </Button>
      </div>

      <div className="mt-3 flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
        <div>
          {files.length > 0
            ? `${files.length} file(s) • ${totalSizeMB} MB total`
            : "No files selected"}
        </div>
        {files.length > 0 && (
          <Button
            size="sm"
            variant="light"
            onPress={clearAll}
            className="text-gray-700 dark:text-gray-300"
          >
            Clear
          </Button>
        )}
      </div>

      {files.length > 0 && (
        <div className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {previews.map(({ file, url }) => (
            <div
              key={`${file.name}-${file.size}`}
              className="relative overflow-hidden rounded-xl border bg-white shadow-sm
                         border-gray-200 dark:border-neutral-800 dark:bg-neutral-900 dark:shadow-none"
            >
              <img
                src={url}
                alt={file.name}
                className="h-32 w-full object-cover rounded-t-xl"
              />
              <div
                className="absolute inset-x-0 bottom-0 truncate text-[11px] px-2 py-1
                              bg-white/90 text-gray-700 dark:bg-black/50 dark:text-gray-200"
              >
                {file.name}
              </div>
              <button
                type="button"
                onClick={() => removeFile(file.name, file.size)}
                className="absolute right-2 top-2 rounded-full px-2 py-1 text-xs
                           bg-white/70 text-gray-700 hover:bg-white
                           dark:bg-black/50 dark:text-gray-300 dark:hover:bg-black/70"
                aria-label="Remove file"
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      )}

      <Button
        type="submit"
        disabled={loading || files.length === 0}
        className="mt-4 w-full rounded-2xl font-medium
                   bg-gray-200 text-gray-800 hover:bg-gray-300
                   dark:bg-neutral-800 dark:text-gray-100 dark:hover:bg-neutral-700"
      >
        {loading ? "Uploading..." : "Upload"}
      </Button>
    </form>
  );
}
