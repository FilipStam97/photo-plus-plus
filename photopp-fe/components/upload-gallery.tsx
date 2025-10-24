import prisma from "@/app/_shared/server/prisma";
import { addToast } from "@heroui/toast";

export default async function UploadGallery() {
  const files = await prisma.file.findMany({
    orderBy: {
      createdAt: "desc",
    },
  });

  console.log(files)
  if (!files) {
    addToast({
        title: "Error",
        description: "Something went wrong!",
        timeout: 3000,
        color: 'danger',
    });
    return <div>Something went wrong</div>;
  }

  return (
    <div className="grid grid-cols-4 gap-2">
      {files.map((file) => (
        <img
          className="h-full rounded-md object-cover"
          key={file.id}
          alt="Image uploaded by user"
          src={file.url!}
        />
      ))}
    </div>
  );
}