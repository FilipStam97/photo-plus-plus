"use client";
import { Button } from "@heroui/button";
import {
  getPresignedUrls,
  handleUpload,
  MAX_FILE_SIZE_NEXTJS_ROUTE,
  validateFiles,
} from "./../app/_shared/client/minio";
import { useRef, useState } from "react";
import { Input } from "@heroui/input";
import { addToast } from "@heroui/toast";
import type { Dispatch, SetStateAction } from "react";
interface UploadFormProps {
  folderName?: string;
  setTriggerFetch?: Dispatch<SetStateAction<boolean>>;
}

export default function UploadForm({
  folderName,
  setTriggerFetch,
}: UploadFormProps) {
  const [loading, setLoading] = useState(false);
  const [files, setFiles] = useState<File[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    try {
      setLoading(true);

      if (files.length) {
        // Create the expected ShortFileProp array for the API
        const shortFileProps = files.map((file) => ({
          originalFileName: file.name,
          fileSize: file.size,
        }));

        // Validate files before uploading
        // const error = validateFiles(shortFileProps, MAX_FILE_SIZE_NEXTJS_ROUTE);
        // if (error) {
        //   addToast({
        //     title: "Error",
        //     description: error,
        //     timeout: 3000,
        //     color: "danger",
        //   });
        //   return;
        // }

        // Get presigned URLs from the API
        const presignedUrls = await getPresignedUrls(
          shortFileProps,
          folderName
        );
        // Upload files using presigned URLs
        const res = await handleUpload(files, presignedUrls, () => {
          addToast({
            title: "Success",
            description: "Files uploaded successfully",
            timeout: 3000,
            color: "success",
          });
          setFiles([]); // Clear files after successful upload
          setTriggerFetch?.((prev) => !prev);
          if (inputRef.current) {
            inputRef.current.value = "";
          }
        });
        // reload
        // window.location.reload();
      }
    } catch (error) {
      addToast({
        title: "Error",
        description: "Unknown error",
        timeout: 3000,
        color: "danger",
      });
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (fileList) {
      setFiles(Array.from(fileList));
    }
  };

  return (
    <form onSubmit={onSubmit}>
      <Input
        ref={inputRef}
        className="mb-2"
        type="file"
        multiple
        disabled={loading}
        onChange={handleFileChange}
        accept=".png, .jpg, .jpeg, .bmp"
      />

      <Button
        type="submit"
        disabled={loading || files.length < 1}
        className="w-full"
      >
        {loading && "Loading..."}
        Upload
      </Button>
    </form>
  );
}
