/*
  Warnings:

  - You are about to drop the column `image_id` on the `face_embeddings` table. All the data in the column will be lost.
  - Added the required column `bucket_name` to the `face_embeddings` table without a default value. This is not possible if the table is not empty.
  - Added the required column `image_name` to the `face_embeddings` table without a default value. This is not possible if the table is not empty.
  - Added the required column `embedding` to the `face_embeddings` table without a default value. This is not possible if the table is not empty.
  - Made the column `bbox` on table `face_embeddings` required. This step will fail if there are existing NULL values in that column.

*/
-- DropForeignKey
ALTER TABLE "public"."face_embeddings" DROP CONSTRAINT "face_embeddings_image_id_fkey";

-- AlterTable
ALTER TABLE "face_embeddings" DROP COLUMN "image_id",
ADD COLUMN     "bucket_name" TEXT NOT NULL,
ADD COLUMN     "face_image_path" TEXT,
ADD COLUMN     "image_name" TEXT NOT NULL,
DROP COLUMN "embedding",
ADD COLUMN     "embedding" JSONB NOT NULL,
ALTER COLUMN "bbox" SET NOT NULL;

-- CreateIndex
CREATE INDEX "face_embeddings_bucket_name_idx" ON "face_embeddings"("bucket_name");

-- CreateIndex
CREATE INDEX "face_embeddings_is_representative_idx" ON "face_embeddings"("is_representative");

-- CreateIndex
CREATE INDEX "face_embeddings_image_name_idx" ON "face_embeddings"("image_name");
