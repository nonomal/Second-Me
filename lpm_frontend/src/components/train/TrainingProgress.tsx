import type { TrainProgress } from '@/service/train';
import type { TrainingJobInfo } from '@/service/cloudService';
import LocalTrainingProgress from './LocalTrainingProgress';
import CloudTrainingProgress from './CloudTrainingProgress';

interface TrainingProgressProps {
  trainingProgress: TrainProgress;
  status: string;
  trainingType?: 'local' | 'cloud';
  cloudJobInfo?: TrainingJobInfo | null;
}

const TrainingProgress = (props: TrainingProgressProps): JSX.Element => {
  const { trainingProgress, status, trainingType = 'local', cloudJobInfo } = props;

  // 根据训练类型条件渲染对应的组件
  if (trainingType === 'cloud') {
    return <CloudTrainingProgress cloudJobInfo={cloudJobInfo || null} status={status} />;
  }

  // 默认显示本地训练进度
  return <LocalTrainingProgress status={status} trainingProgress={trainingProgress} />;
};

export default TrainingProgress;
