import React, { useState } from "react";
import type { SegmentationSlice } from "../types/segmentation";

interface SegmentationViewerProps {
  slices: SegmentationSlice[];
}

const SegmentationViewer: React.FC<SegmentationViewerProps> = ({
  slices,
}) => {
  const [currentSlice, setCurrentSlice] = useState(0);

  if (slices.length === 0) {
    return (
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
        <p className="text-slate-400">
          No segmentation data available.
        </p>
      </div>
    );
  }

  const slice = slices[currentSlice];

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
      <h3 className="text-lg font-semibold text-white mb-4">
        2D Segmentation Viewer
      </h3>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        {/* MRI */}
        <div>
          <p className="text-sm text-slate-400 mb-2">
            MRI Slice
          </p>

          <div className="aspect-square bg-black rounded-lg overflow-hidden">
            <img
              src={slice.mriSlice}
              alt={`MRI Slice ${slice.sliceIndex}`}
              className="w-full h-full object-contain"
            />
          </div>
        </div>

        {/* Ground Truth */}
        <div>
          <p className="text-sm text-slate-400 mb-2">
            Ground Truth
          </p>

          <div className="aspect-square bg-black rounded-lg overflow-hidden">
            {slice.groundTruthMask ? (
              <img
                src={slice.groundTruthMask}
                alt="Ground Truth Mask"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="h-full flex items-center justify-center">
                <p className="text-sm text-slate-500">
                  Ground Truth unavailable
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Prediction */}
        <div>
          <p className="text-sm text-slate-400 mb-2">
            Prediction
          </p>

          <div className="aspect-square bg-black rounded-lg overflow-hidden">
            {slice.predictedMask ? (
              <img
                src={slice.predictedMask}
                alt="Predicted Mask"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="h-full flex items-center justify-center">
                <p className="text-sm text-slate-500">
                  Prediction unavailable
                </p>
              </div>
            )}
          </div>
        </div>

      </div>

      {/* Navigation */}
      <div className="mt-6">
        <div className="flex justify-between mb-2">
          <button
            onClick={() =>
              setCurrentSlice((prev) => Math.max(prev - 1, 0))
            }
            disabled={currentSlice === 0}
            className="px-4 py-2 bg-slate-800 rounded-lg text-white disabled:opacity-40"
          >
            Previous
          </button>

          <span className="text-sm text-slate-400">
            Slice {currentSlice + 1} / {slices.length}
          </span>

          <button
            onClick={() =>
              setCurrentSlice((prev) =>
                Math.min(prev + 1, slices.length - 1)
              )
            }
            disabled={currentSlice === slices.length - 1}
            className="px-4 py-2 bg-slate-800 rounded-lg text-white disabled:opacity-40"
          >
            Next
          </button>
        </div>

        <input
          type="range"
          min="0"
          max={slices.length - 1}
          value={currentSlice}
          onChange={(e) =>
            setCurrentSlice(Number(e.target.value))
          }
          className="w-full"
        />
      </div>
    </div>
  );
};

export default SegmentationViewer;