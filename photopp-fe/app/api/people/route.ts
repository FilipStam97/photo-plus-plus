import { createPresignedUrlToDownload } from "@/app/_shared/server/minio";
import { MinioClient } from "@/app/_shared/server/minio/client";
import { NextResponse } from "next/server";
const bucketName = process.env.MINIO_BUCKET_NAME!;

export async function GET(req: Request) {
  try {
    //gets images from faces folder
    // const objects: any[] = [];
    // await new Promise<void>((resolve, reject) => {
    //   const stream = MinioClient.listObjectsV2(bucketName, "faces", true);
    //   stream.on("data", (obj) => objects.push(obj));
    //   stream.on("end", () => resolve());
    //   stream.on("error", (err) => reject(err));
    // });

    // if (objects.length === 0) {
    //   return NextResponse.json([], { status: 200 });
    // }

    const flaskRes = await fetch("http://127.0.0.1:5001/api/faces/clusters?bucket_name=" + bucketName , {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });
    if (!flaskRes.ok) {
      throw new Error("Flask API error");
    }
    const data = await flaskRes.json();
    console.log(data)
    // const presignedUrls = await Promise.all(
    //   objects.map(async (obj) => {
    //     const fileName = obj.name;
    //     const url = await createPresignedUrlToDownload({
    //       bucketName,
    //       fileName,
    //       expiry: 60 * 60, // 1h
    //     });
    //     return {
    //       originalName: fileName.split("/").pop(),
    //       url,
    //     };
    //   })
    // );

    return NextResponse.json(data.clusters, { status: 200 });
  } catch (err) {
    console.error("Error fetching presigned URLs:", err);
    return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
  }


}

export async function POST(req: Request) {
    const { cluster_id } = await req.json() as { cluster_id: number };
    try {
    const flaskRes = await fetch("http://127.0.0.1:5001/api/faces/find-similar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bucket_name: process.env.MINIO_BUCKET_NAME, cluster_id }),
    }).catch(err => console.error("Flask clustering failed:", err));

    if (!flaskRes?.ok) {
      throw new Error("Flask API error");
    }
    const data = await flaskRes.json(); 
        return NextResponse.json( data);
    } catch (error) {
        console.error(error);
        return new NextResponse("Internal server error", { status: 500 });
    }
}