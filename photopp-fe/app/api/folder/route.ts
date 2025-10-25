import { NextResponse } from "next/server";
import { createBucketIfNotExists } from "@/app/_shared/server/minio";

const bucketName = process.env.MINIO_BUCKET_NAME!;

export async function POST(req: Request) {
    try {
        const { name } = await req.json();
        if (!name) return new NextResponse("Folder name required", { status: 400 });

        await createBucketIfNotExists(bucketName);
        // check if folder already existis
        const existing = await prisma.folder.findUnique({
            where: { name }
        });

        if (existing)
            return new NextResponse("Folder already exists", { status: 400 });

        const folder = await prisma.folder.create({
            data: { name }
        });

        return NextResponse.json(folder);
    } catch (error) {
        console.error(error);
        return new NextResponse("Internal server error", { status: 500 });
    }
}

export async function GET() {
    console.log("get")
    const folders = await prisma.folder.findMany({ orderBy: { createdAt: "desc" } });
    return NextResponse.json(folders);
}