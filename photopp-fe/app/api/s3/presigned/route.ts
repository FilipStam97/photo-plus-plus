import type { ShortFileProp, PresignedUrlProp } from "../../../_shared/server/minio";
import { createPresignedUrlToDownload, createPresignedUrlToUpload } from "../../../_shared/server/minio";
import { nanoid } from "nanoid";
import { NextResponse } from "next/server";
import prisma from "../../../_shared/server/prisma";

const bucketName = process.env.MINIO_BUCKET_NAME!;
const expiry = 60 * 60; // 1 hour

export async function POST(req: Request) {
  try {
    // get the files from the request body
    const { files, folderName } = await req.json() as { files: ShortFileProp[], folderName?: string };

    console.log({ files });

    if (!files?.length) {
      return new NextResponse("No files to upload", { status: 400 });
    }

    const presignedUrls = [] as PresignedUrlProp[];

    if (files?.length) {
      // use Promise.all to get all the presigned urls in parallel
      await Promise.all(
        // loop through the files
        files.map(async (file) => {
          const fileName = nanoid(12);
          const pathInBucket = folderName ? `${folderName}/${fileName}` : fileName;


          // get presigned url using s3 sdk
          const url = await createPresignedUrlToUpload({
            bucketName,
            fileName: pathInBucket,
            expiry,
          });

          // add presigned url to the list
          presignedUrls.push({
            fileNameInBucket: pathInBucket,
            originalFileName: file.originalFileName,
            fileSize: file.fileSize,
            url,
          });
        })
      );
    }

    console.log({ presignedUrls });

    const publicUrls = presignedUrls.map((presignedUrl) => {
      const publicUrl = `http${process.env.MINIO_SSL === "true" ? "s" : ""}://${process.env.MINIO_ENDPOINT}:${process.env.MINIO_PORT}/${process.env.MINIO_BUCKET_NAME}/${presignedUrl.fileNameInBucket}`;
      return { ...presignedUrl, publicUrl };
    });

    console.log({ publicUrls });

    //Get the file name in bucket from the database

    const saveFilesInfo = await prisma.file.createMany({
      data: publicUrls.map((presignedUrl: any) => ({
        bucket: process.env.MINIO_BUCKET_NAME,
        fileName: presignedUrl.fileNameInBucket,
        originalName: presignedUrl.originalFileName,
        size: presignedUrl.fileSize,
        url: presignedUrl.publicUrl,
      })),
    });

    return NextResponse.json(presignedUrls);
  } catch (error) {
    console.error({ error });
    return new NextResponse("Internal error", { status: 500 });
  }
}