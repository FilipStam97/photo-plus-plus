'use client';
import { addToast } from "@heroui/toast";
import { useState } from "react";

interface FaceComponentProps {
    url: string;
    originalName?: string;
    personName?: string;
}

export default function FaceComponent({ url, originalName, personName = "" }: FaceComponentProps) {
    const [name, setName] = useState(personName);
    const [showTick, setShowTick] = useState(false);

    const handleSave = async () => {
        if (!originalName || !name) return;

        try {
            const res = await fetch("/api/person", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    fileName: originalName,
                    name,
                }),
            });

            if (!res.ok) throw new Error("Failed to save name");

            addToast({
                title: "Success",
                description: `Name "${name}" saved successfully!`,
                timeout: 3000,
                color: "success",
            });

            setShowTick(false);
        } catch (err) {
            console.error(err);
            addToast({
                title: "Error",
                description: "Could not save name",
                timeout: 3000,
                color: "danger",
            });
        }
    };
    return (
        <div className="relative w-32 h-32 overflow-hidden border border-gray-300 shadow-md">
            <img
                src={url}
                alt={originalName || "Face"}
                className="object-cover w-full h-full"
            />

            <div className="absolute bottom-2 left-1/2 transform -translate-x-1/2 flex items-center gap-1">
                <input
                    type="text"
                    placeholder="Enter name"
                    value={name}
                    onChange={(e) => {
                        setName(e.target.value);
                        setShowTick(e.target.value.length > 0);
                    }}
                    className="px-2 py-1 bg-white/80 rounded text-sm w-24 text-center focus:outline-none"
                />
                {showTick && (
                    <button
                        onClick={handleSave}
                        className="font-bold hover:text-green-800"
                    >
                        ✔
                    </button>
                )}
            </div>
        </div>
    );
}