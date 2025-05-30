import type { TrainProgress } from '@/service/train';
import type { CloudProgressData } from '@/service/cloudService';
import LocalTrainingProgress from './LocalTrainingProgress';
import CloudTrainingProgress from './CloudTrainingProgress';

interface TrainingProgressProps {
  trainingProgress: TrainProgress; // For local training
  status: string;
  trainingType?: 'local' | 'cloud';
  cloudProgressData?: CloudProgressData | null; // Add new prop for cloud progress
  cloudJobId?: string | null; // Add prop for cloud job ID
}

const TrainingProgress = (props: TrainingProgressProps): JSX.Element => {
  const { 
    trainingProgress, 
    status, 
    trainingType = 'local', 
    cloudProgressData, // Use new prop
    cloudJobId // Use new prop
  } = props;

  if (trainingType === 'cloud') {
    // Pass cloudProgressData and cloudJobId to CloudTrainingProgress
    return <CloudTrainingProgress cloudProgressData={cloudProgressData || null} cloudJobId={cloudJobId || null} />;
  }

  return <LocalTrainingProgress status={status} trainingProgress={trainingProgress} />;
};

export default TrainingProgress;
