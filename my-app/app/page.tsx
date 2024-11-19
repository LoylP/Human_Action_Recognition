"use client";
import { useState } from "react";
import { BsRobot } from "react-icons/bs";
import { SiRobotframework } from "react-icons/si";
import { House, UserPlus, Webcam } from 'lucide-react';
import Link from "next/link";

const path_api = "http://127.0.0.1:8000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadSuccess, setUploadSuccess] = useState<boolean>(false);
  const [cameraUrl, setCameraUrl] = useState<string>("");

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setFile(file);
    setUploadSuccess(false);
    setCameraUrl(""); // Reset the video URL when a new file is selected
  };

  const handleFileUpload = async () => {
    if (!file) {
      alert("Please choose a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${path_api}/upload_video/`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        setUploadSuccess(true);
        alert(data.message); // Show success message
      } else {
        const errorData = await response.json();
        console.error("Failed to upload video:", errorData.message);
        alert(errorData.message);
      }
    } catch (error) {
      console.error("Error uploading video:", error);
      alert("An error occurred while uploading the video.");
    }
  };

  const handleShowVideo = () => {
    if (!file) {
      alert("No video file uploaded!");
      return;
    }
    // Generate video URL for uploaded file
    setCameraUrl(`${path_api}/predict/${file.name}`);
  };

  return (
    <div className="flex flex-col justify-center items-center h-screen bg-gray-200">
      {/* Title and Navigation Buttons */}
      <div className="w-full flex justify-center text-gray-900">
        <h1 className="text-3xl font-bold">Action Recognition Web in Nextjs</h1>
      </div>

      <div className="w-full flex justify-center items-center px-20 py-6 text-white">
        <div className="flex space-x-4">
          <Link href={"/"}>
            <button className="bg-gray-800 py-2 px-4 rounded-md hover:bg-gray-900"><House /></button>
          </Link>
          <Link href={"/multi"}>
            <button className="bg-gray-600 py-2 px-4 rounded-md hover:bg-gray-900"><UserPlus /></button>
          </Link>
          <Link href={"/realtime"}>
            <button className="bg-gray-600 py-2 px-4 rounded-md hover:bg-gray-900"><Webcam /></button>
          </Link>
        </div>
      </div>

      <div className="flex w-3/4 bg-white p-4 rounded-xl shadow-lg">
        <div className="flex-1">
          <div className="mb-4">
            {file ? (
              <SiRobotframework className="text-7xl text-sky-600 mx-auto mb-4" />
            ) : (
              <BsRobot className="text-7xl text-gray-600 mx-auto mb-4" />
            )}
            {uploadSuccess ? (
              <h1 className="text-xl font-bold text-center text-green-600">
                Video Uploaded Successfully
              </h1>
            ) : (
              <h1 className="text-xl font-bold text-center">
                Upload Your Video File
              </h1>
            )}
          </div>
          <div className="flex w-2/4 mx-auto flex-col bg-sky-500 py-2 rounded-md hover:cursor-pointer hover:bg-sky-700 hover:shadow-lg">
            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".mp4,.avi,.mov"
              onChange={handleFileChange}
            />
            <label htmlFor="file-upload" className="text-white mx-auto">
              Choose File
            </label>
          </div>
          {file ? (
            <div className="text-gray-500 text-sm text-center mt-2">
              {file.name}
            </div>
          ) : (
            <div className="text-gray-500 text-sm text-center mt-2">
              No file chosen (Only accept '.mp4', '.avi', '.mov' files)
            </div>
          )}
          <div className="flex flex-col mt-4">
            <div
              onClick={handleFileUpload}
              className="mt-4 flex w-2/4 mx-auto flex-col bg-sky-500 py-2 rounded-md hover:cursor-pointer hover:bg-sky-700 hover:shadow-lg"
            >
              <button className="text-white mx-auto">Upload</button>
            </div>
          </div>
          <div className="flex flex-col mt-4">
            <div
              onClick={handleShowVideo}
              className="mt-4 flex w-2/4 mx-auto flex-col bg-green-500 py-2 rounded-md hover:cursor-pointer hover:bg-green-700 hover:shadow-lg"
            >
              <button className="text-white mx-auto">Submit</button>
            </div>
            <div className="text-gray-500 text-sm text-center mt-2 mx-24">
              Action prediction in video: falling_down, jump_up, kicking, punch, running, sit_down, stand_up
            </div>
          </div>
        </div>

        {/* Right side for showing the uploaded video */}
        <div className="flex-1 bg-black rounded-xl">
          <div>
            {cameraUrl && (
              <div className="mt-6">
                <h2 className="text-lg font-bold text-center text-white">
                  Uploaded Video:
                </h2>
                <div className="flex justify-center">
                  <img
                    src={cameraUrl}
                    alt="Uploaded Video"
                    className="w-full max-h-96 object-contain"
                  />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
