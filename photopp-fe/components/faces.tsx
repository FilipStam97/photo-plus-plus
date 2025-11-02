"use client";
import { useEffect, useState } from "react";
import { addToast } from "@heroui/toast";
import FaceComponent from "./face-component";

export default function FacesGallery() {
  const [files, setFiles] = useState<
    {
      cropped_face: string;
      cluster_id: number;
      face_count: number;
      sample_image: string;
    }[]
  >([]);
  const [people, setPeople] = useState<
    { imageName: string; personName: string }[]
  >([]);

  const [loading, setLoading] = useState(false);
  useEffect(() => {
    setLoading(true);

    const fetchAll = async () => {
      try {
        const [filesRes] = await Promise.all([
          fetch("/api/people"),
          // fetch("/api/person"),
        ]);

        const [filesData] = await Promise.all([
          filesRes.json(),
          // peopleRes.json(),
        ]);

        setFiles(Array.isArray(filesData) ? filesData : []);
        // setPeople(Array.isArray(peopleData) ? peopleData : []);
      } catch (err) {
        console.error(err);
        addToast({
          title: "Error",
          description: "Failed to load data",
          color: "danger",
        });
      } finally {
        setLoading(false);
      }
    };

    fetchAll();
  }, []);

  if (loading) {
    return <p>Loading faces...</p>;
  }

  if (files.length === 0) {
    return <p>No faces found.</p>;
  }

  return (
    <div className="grid grid-cols-8 gap-2">
      {files.map((cluster, idx) => (
        <div key={idx} className="flex flex-col items-center">
          <FaceComponent
            cropped_image={cluster.cropped_face}
            originalName={cluster.sample_image}
            personName=""
            cluster_id={cluster.cluster_id}
          ></FaceComponent>
        </div>
      ))}
    </div>
  );
}
