"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  DrawingUtils,
  FaceLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import "./style.css";

const FaceDetectionApp = () => {
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [webcamRunning, setWebcamRunning] = useState(false);
  const [distance, setDistance] = useState("");
  const [orientation, setOrientation] = useState("");
  const [runningMode, setRunningMode] = useState("IMAGE");

  const canvasImageRef = useRef(null);
  const canvasVideoRef = useRef(null);
  const videoRef = useRef(null);
  const enableWebcamButtonRef = useRef(null);

  const videoWidth = 480;

  useEffect(() => {
    const loadFaceLandmarker = async () => {
      const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );

      const landmarker = await FaceLandmarker.createFromOptions(
        filesetResolver,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          outputFaceBlendshapes: true,
          runningMode: runningMode,
          numFaces: 1,
        }
      );

      setFaceLandmarker(landmarker);
    };

    loadFaceLandmarker();
  }, [runningMode]);

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file || !faceLandmarker) return;

    const img = await loadImageToCanvas(file, canvasImageRef.current);
    const faceLandmarkerResult = await faceLandmarker.detect(img);

    const canvasCtx = canvasImageRef.current.getContext("2d");
    const drawingUtils = new DrawingUtils(canvasCtx);

    drawLandmarksToCanvas(faceLandmarkerResult.faceLandmarks, drawingUtils);

    const { zDistance } = calculateHeadDepth(
      faceLandmarkerResult.faceLandmarks
    );
    setDistance(`Estimated nose distance: ${zDistance.toFixed(2)} in z`);

    const landmarks = faceLandmarkerResult.faceLandmarks[0];
    const { yaw, pitch, roll } = await calculateHeadOrientation(landmarks, {
      width: img.width,
      height: img.height,
    });

    const { msgRoll, msgPitch, msgYaw } = displayOrientationResultMessage(
      yaw,
      pitch,
      roll
    );
    setOrientation(`Face orientation: ${msgRoll} | ${msgPitch} | ${msgYaw}`);
  };

  const loadImageToCanvas = (file, canvasElement) => {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = URL.createObjectURL(file);

      img.onload = () => {
        canvasElement.width = img.width;
        canvasElement.height = img.height;
        const ctx = canvasElement.getContext("2d");
        ctx.drawImage(img, 0, 0);
        resolve(img);
      };
    });
  };

  const enableWebcam = () => {
    if (!faceLandmarker) {
      console.log("Wait! faceLandmarker not loaded yet.");
      return;
    }

    if (webcamRunning) {
      setWebcamRunning(false);
    } else {
      setWebcamRunning(true);

      // Enable webcam stream
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
          videoRef.current.addEventListener("loadeddata", predictWebcam);
        })
        .catch((err) => console.error("Webcam not accessible", err));
    }
  };

  const predictWebcam = async () => {
    if (!webcamRunning || !faceLandmarker || !videoRef.current) return;

    if (runningMode === "IMAGE") {
      setRunningMode("VIDEO");
      await faceLandmarker.setOptions({ runningMode: "VIDEO" });
    }

    const canvasCtx = canvasVideoRef.current.getContext("2d");
    canvasVideoRef.current.width = videoRef.current.videoWidth;
    canvasVideoRef.current.height = videoRef.current.videoHeight;

    const startTimeMs = performance.now();
    const results = faceLandmarker.detectForVideo(
      videoRef.current,
      startTimeMs
    );

    if (results.faceLandmarks) {
      const drawingUtils = new DrawingUtils(canvasCtx);
      results.faceLandmarks.forEach((landmarks) => {
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_TESSELATION,
          { color: "#C0C0C070", lineWidth: 1 }
        );
        drawingUtils.drawConnectors(
          landmarks,
          FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
          { color: "#E0E0E0" }
        );
      });
    }

    // Keep predicting if the webcam is running
    if (webcamRunning) {
      requestAnimationFrame(predictWebcam);
    }
  };

  const drawLandmarksToCanvas = (faceLandmarks, drawingUtils) => {
    faceLandmarks.forEach((landmarks) => {
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_TESSELATION,
        { color: "#C0C0C070", lineWidth: 1 }
      );
      drawingUtils.drawConnectors(
        landmarks,
        FaceLandmarker.FACE_LANDMARKS_FACE_OVAL,
        { color: "#E0E0E0" }
      );
    });
  };

  const calculateHeadDepth = (faceLandmarks) => {
    const nose = faceLandmarks[0][1];
    const zDistance = Math.abs(nose.z * 100);
    return { zDistance };
  };

  const calculateHeadOrientation = async (landmarks, { width, height }) => {
    const yaw = 0,
      pitch = 0,
      roll = 0; // Placeholder
    return { yaw, pitch, roll };
  };

  const displayOrientationResultMessage = (yaw, pitch, roll) => {
    const msgRoll = `roll: ${(180.0 * (roll / Math.PI)).toFixed(2)}`;
    const msgPitch = `pitch: ${(180.0 * (pitch / Math.PI)).toFixed(2)}`;
    const msgYaw = `yaw: ${(180.0 * (yaw / Math.PI)).toFixed(2)}`;
    return { msgRoll, msgPitch, msgYaw };
  };

  return (
    <div>
      <h1>Face Detection App</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <canvas ref={canvasImageRef}></canvas>
      <button ref={enableWebcamButtonRef} onClick={enableWebcam}>
        {webcamRunning ? "Stop Webcam" : "Start Webcam"}
      </button>
      <video ref={videoRef} autoPlay muted hidden={!webcamRunning}></video>
      <canvas ref={canvasVideoRef}></canvas>
      <p>{distance}</p>
      <p>{orientation}</p>
    </div>
  );
};

export default FaceDetectionApp;
