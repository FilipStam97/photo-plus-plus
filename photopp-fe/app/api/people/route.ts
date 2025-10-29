import { createPresignedUrlToDownload } from "@/app/_shared/server/minio";
import { MinioClient } from "@/app/_shared/server/minio/client";
import { NextResponse } from "next/server";
const bucketName = process.env.MINIO_BUCKET_NAME!;

export async function GET(req: Request) {
  try {
    //gets images from faces folder
    const objects: any[] = [];
    await new Promise<void>((resolve, reject) => {
      const stream = MinioClient.listObjectsV2(bucketName, "faces", true);
      stream.on("data", (obj) => objects.push(obj));
      stream.on("end", () => resolve());
      stream.on("error", (err) => reject(err));
    });

    if (objects.length === 0) {
      return NextResponse.json([], { status: 200 });
    }

    const presignedUrls = await Promise.all(
      objects.map(async (obj) => {
        const fileName = obj.name;
        const url = await createPresignedUrlToDownload({
          bucketName,
          fileName,
          expiry: 60 * 60, // 1h
        });
        return {
          originalName: fileName.split("/").pop(),
          url,
        };
      })
    );
    console.log(presignedUrls)
    return NextResponse.json(presignedUrls, { status: 200 });
  } catch (err) {
    console.error("Error fetching presigned URLs:", err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}