import prisma from "@/app/_shared/server/prisma";
import { NextResponse } from "next/server";

export async function GET() {
    try {
        //gets all names currently present in db
        // const people = await prisma.faceEmbedding.findMany({
        //     select: { imageName: true, personName: true },
        //     where: { isRepresentative: true }
        // });
        const flaskRes = await fetch("http://127.0.0.1:5000/api/faces/clusters?bucket_name=test"  , {
            method: "GET",
            });

            if (!flaskRes.ok) {
            throw new Error("Flask API error");
            }
        return NextResponse.json(flaskRes.body, { status: 200 });
    } catch (err) {
        console.error("Error fetching people:", err);
        return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
    }
}

// export async function POST(req: Request) {
//     try {
//         //updates name 
//         const { fileName, name } = await req.json();

//         if (!fileName || !name) {
//             return NextResponse.json({ error: "fileName and name required" }, { status: 400 });
//         }
        
//         // const face = await prisma.faceEmbedding.findFirst({
//         //     where: { imageName: fileName }
//         // });

//         // if (!face) {
//         //     return NextResponse.json({ error: "Face not found" }, { status: 404 });
//         // }

//         // const person = await prisma.faceEmbedding.update({
//         //     where: { id: face.id },
//         //     data: { personName: name },
//         // });

//         return NextResponse.json(person, { status: 200 });
//     } catch (err) {
//         console.error("Error saving person:", err);
//         return NextResponse.json({ error: "Internal Server Error" }, { status: 500 });
//     }
// }