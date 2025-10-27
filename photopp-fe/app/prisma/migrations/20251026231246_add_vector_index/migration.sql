/*
  Warnings:

  - You are about to drop the `FaceEmbedding` table. If the table is not empty, all the data it contains will be lost.
  - You are about to drop the `File` table. If the table is not empty, all the data it contains will be lost.

*/
-- DropTable
DROP TABLE "public"."FaceEmbedding";

-- DropTable
DROP TABLE "public"."File";

-- CreateTable
CREATE TABLE "images" (
    "id" SERIAL NOT NULL,
    "bucket" TEXT,
    "fileName" TEXT,
    "originalName" TEXT,
    "size" DOUBLE PRECISION,
    "url" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "images_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "face_embeddings" (
    "id" SERIAL NOT NULL,
    "image_id" INTEGER NOT NULL,
    "embedding" TEXT,
    "bbox" JSONB,
    "cluster_id" INTEGER,
    "is_representative" BOOLEAN NOT NULL DEFAULT false,
    "created_at" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "face_embeddings_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "face_embeddings_cluster_id_idx" ON "face_embeddings"("cluster_id");

-- AddForeignKey
ALTER TABLE "face_embeddings" ADD CONSTRAINT "face_embeddings_image_id_fkey" FOREIGN KEY ("image_id") REFERENCES "images"("id") ON DELETE CASCADE ON UPDATE CASCADE;
