'use client';

import { useParams } from "next/navigation";
import { useRouter } from "next/navigation";
import FacesGallery from "@/components/faces";

export default function PeoplePage() {
  const params = useParams();
  const router = useRouter();

  return (
      <section>
        <FacesGallery />
      </section>
  );
}