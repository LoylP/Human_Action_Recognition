"use client";
import { useState } from "react";
import { House, UserPlus, Webcam } from 'lucide-react';
import Link from "next/link";

const path_yolo_api = "http://127.0.0.1:8080";

export default function Home() {
  const [inputUrl, setInputUrl] = useState<string>("");  // URL input by the user
  const [videoSet, setVideoSet] = useState<boolean>(false); // State to track if video URL is set

  const handleUrlChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputUrl(e.target.value);
  };

  const handleSubmit = async () => {
    if (!inputUrl) {
      alert("Please enter a valid video URL!");
      return;
    }

    // Send the input URL to the backend API
    try {
      const response = await fetch(`${path_yolo_api}/set_input_url/?url=${encodeURIComponent(inputUrl)}`, {
        method: "POST",
      });

      const data = await response.json();

      if (data.status === "success") {
        alert(data.message); 
        setVideoSet(true); 
      } else {
        alert("Failed to set video URL.");
      }
    } catch (error) {
      console.error("Error setting video URL:", error);
      alert("An error occurred while setting the video URL.");
    }
  };

  return (
    <div className="flex flex-col justify-center items-center h-screen bg-gray-200">
      <div className="w-full flex justify-center text-gray-900">
        <h1 className="text-3xl font-bold">Action Recognition Web Real-Time</h1>
      </div>

      <div className="w-full flex justify-center items-center px-20 py-2 text-white">
        <div className="flex space-x-4">
          <Link href={"/"}>
            <button className="bg-gray-600 py-2 px-4 rounded-md hover:bg-gray-900"><House /></button>
          </Link>
          <Link href={"/multi"}>
            <button className="bg-gray-600 py-2 px-4 rounded-md hover:bg-gray-900"><UserPlus /></button>
          </Link>
          <Link href={"/realtime"}>
            <button className="bg-gray-800 py-2 px-4 rounded-md hover:bg-gray-900"><Webcam /></button>
          </Link>
        </div>
      </div>

      <div className="flex flex-col w-3/4 bg-white p-4 rounded-xl shadow-lg">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">Enter Video URL</label>
          <div className="flex rounded-xl border-2 border-gray-300 p-1 hover:shadow-lg">
            <input
              type="text"
              className="w-full text-md size-auto rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring focus:ring-blue-500 focus:ring-opacity-50"
              value={inputUrl}
              onChange={handleUrlChange}
              placeholder="path to video/camera"
            />
            <button
            onClick={handleSubmit}
            className="bg-green-500 py-2 px-4 rounded-md text-white hover:bg-green-700"
          >
            Submit
          </button>
          </div>
        </div>

        {videoSet && (
          <div className="flex-1 bg-black rounded-xl">
            <h2 className="text-lg font-bold text-center text-white">Real-Time:</h2>
            <div className="flex justify-center">
              <img
                src={`${path_yolo_api}/pose_video_url/`}
                alt="Real-Time"
                className="w-full max-h-96 object-contain"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
