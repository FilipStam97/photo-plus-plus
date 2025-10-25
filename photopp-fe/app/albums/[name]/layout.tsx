import UploadGallery from "@/components/upload-gallery";

export default function PricingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <section >
      <div className="inline-block max-w-lg text-center justify-center">
        {children}
      </div>
    </section>
  );
}
