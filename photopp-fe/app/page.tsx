
import UploadForm from "@/components/upload-form";
import UploadGallery from "@/components/upload-gallery";


export default function Home() {


  return (
    <>
    <section className="flex flex-col items-center justify-center gap-4 py-8 md:py-10">
      <UploadForm />
    </section>
    <section>
       <UploadGallery />
    </section>
    </>
  );
}
