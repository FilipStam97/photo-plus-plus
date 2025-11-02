import { NextResponse } from "next/server";
import prisma from "@/app/_shared/server/prisma";

export async function POST(req: Request) {
  try {
    const { presignedUrls } = await req.json();

    const publicUrls = presignedUrls.map((p: any) => ({
      ...p,
      publicUrl: `http${process.env.MINIO_SSL === "true" ? "s" : ""}://${process.env.MINIO_ENDPOINT}:${process.env.MINIO_PORT}/${process.env.MINIO_BUCKET_NAME}/${p.fileNameInBucket}`,
    }));

    await prisma.file.createMany({
      data: publicUrls.map((p: any) => ({
        bucket: process.env.MINIO_BUCKET_NAME,
        fileName: p.fileNameInBucket,
        originalName: p.originalFileName,
        size: p.fileSize,
        url: p.publicUrl,
      })),
    });

    // Kick off Flask clustering in background
    fetch("http://127.0.0.1:5001/api/cluster-images", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bucket_name: process.env.MINIO_BUCKET_NAME }),
    }).catch(err => console.error("Flask clustering failed:", err));

    return new NextResponse("OK", { status: 200 });
  } catch (err) {
    console.error(err);
    return new NextResponse("Internal error", { status: 500 });
  }
}