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
            url
          });
        })
      );
    }
    // console.log({ presignedUrls });


    //   //TODO: ovo ne mora da se zavrsi da bi se zavrsio request, moze da se doda da se to odradi u pozadini dok ostalo radi
    //   const flaskRes = await fetch("http://127.0.0.1:5000/api/cluster-images" , {
    //     method: "POST",
    //     headers: { "Content-Type": "application/json" },
    //     body: JSON.stringify({
    //       bucket_name: process.env.MINIO_BUCKET_NAME
    //     })
    //   });

    // if (!flaskRes.ok) {
    //   throw new Error("Flask API error");
    // }
    // const flaskData = await flaskRes.json();

    //save face embeddings in database
    // const newEmbedding = await prisma.faceEmbedding.createMany(normalizeEmbeddings);

    return NextResponse.json(presignedUrls);
  } catch (error) {
    console.error({ error });
    return new NextResponse("Internal error", { status: 500 });
  }
}

export async function GET(req: Request) {
  try {
    const url = new URL(req.url);
    const fileNames = url.searchParams.getAll("fileNames");

    const files = await prisma.file.findMany({
      where: fileNames && fileNames.length > 0
    ? {
        fileName: {
          in: fileNames,
        },
      }
    : undefined, 
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