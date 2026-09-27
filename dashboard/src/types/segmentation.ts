export interface SegmentationSlice {
    sliceIndex: number;
    mriSlice: string;
    groundTruthMask: string;
    predictedMask: string;
  }