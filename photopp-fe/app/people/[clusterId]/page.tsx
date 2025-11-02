"use client";

import { useParams } from "next/navigation";
import { useRouter } from "next/navigation";
import ClusterGallery from "@/components/cluster-gallery";

export default function PersonPage() {
  const params = useParams();
  const router = useRouter();

  const clusterIdParam = params.clusterId;
  const clusterId = Array.isArray(clusterIdParam)
    ? Number(clusterIdParam[0])
    : Number(clusterIdParam);

  if (clusterId == null) {
    return <div>Images not found</div>;
  }

  return (
    <>
      <section>
        <ClusterGallery clusterId={clusterId} />
      </section>
    </>
  );
}
