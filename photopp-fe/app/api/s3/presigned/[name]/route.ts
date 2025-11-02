// /api/s3/presigned/[name]/route.ts
import { createPresignedUrlToDownload } from "@/app/_shared/server/minio";
import { NextResponse } from "next/server";
const bucketName = process.env.MINIO_BUCKET_NAME!;
import prisma
 from "@/app/_shared/server/prisma";
 
export async function GET(req: Request, context: { params: Promise<{ name: string }> }) {
  const { name } = await context.params;
  const folderName = name;

  if (!folderName) {
    return NextResponse.json({ error: "Folder name is required" }, { status: 400 });
  }

  try {
    //get file names from db
    const files = await prisma.file.findMany({
      where: { fileName: { startsWith: `${folderName}/` } },
      select: { fileName: true },
    });

    if (!files || files.length === 0) {
      return NextResponse.json({ files: [] }, { status: 200 });
    }

    const presignedUrls = await Promise.all(
      files.map(async (f: any) => {
        const fileName = f.fileName;
        const url = await createPresignedUrlToDownload({
          bucketName: bucketName,
          fileName,
          expiry: 60 * 60,
        });
        return { originalName: fileName.split("/").pop(), url };
      })
    );

    return NextResponse.json(presignedUrls, { status: 200 });
  } catch (err) {
    console.error("Error fetching presigned URLs:", err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }
}
