import { Request } from '../utils/request';
import type { CommonResponse, EmptyResponse } from '../types/responseModal';

interface ProcessInfo {
  cmdline: string[];
  cpu_percent: string;
  create_time: string;
  memory_percent: string;
  pid: string;
}

interface ServiceStatusRes {
  process_info?: ProcessInfo;
  is_running?: boolean;
}

interface StartTrainResponse {
  progress_id: string;
}

export type StepStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'suspended';
export type StageStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'suspended';

interface TrainStep {
  completed: boolean;
  name: string;
  status: StepStatus;
}

interface TrainStage {
  name: string;
  progress: number;
  status: StageStatus;
  steps: Record<string, TrainStep>;
  current_step: string | null;
}

export type SynthesisMode = 'low' | 'medium' | 'high';

export interface TrainingParams {
  number_of_epochs?: number;
  learning_rate?: number;
  concurrency_threads?: number;
  data_synthesis_mode?: SynthesisMode;
}

export enum StageName {
  Stage1 = 'downloading_the_base_model',
  Stage2 = 'activating_the_memory_matrix',
  Stage3 = 'synthesize_your_life_narrative',
  Stage4 = 'prepare_training_data_for_deep_comprehension',
  Stage5 = 'training_to_create_second_me'
}

export enum StageDisplayName {
  Stage1 = 'Downloading the Base Model',
  Stage2 = 'Activating the Memory Matrix',
  Stage3 = 'Synthesize Your Life Narrative',
  Stage4 = 'Prepare Training Data for Deep Comprehension',
  Stage5 = 'Training to create Second Me'
}

interface TrainProgressResponse {
  stages: {
    downloading_the_base_model: TrainStage;
    activating_the_memory_matrix: TrainStage;
    synthesize_your_life_narrative: TrainStage;
    prepare_training_data_for_deep_comprehension: TrainStage;
    training_to_create_second_me: TrainStage;
  };
  overall_progress: number;
  current_stage: StageName;
  status: StageStatus;
}

export interface TrainingConfig extends TrainingParams {
  model_name: string;
}

export const startTrain = (config: TrainingConfig) => {
  return Request<CommonResponse<StartTrainResponse>>({
    method: 'post',
    url: '/api/trainprocess/start',
    data: config
  });
};

export const getTrainProgress = (config: TrainingConfig) => {
  return Request<CommonResponse<TrainProgressResponse>>({
    method: 'get',
    url: `/api/trainprocess/progress/${config.model_name}`
  });
};

export const stopTrain = () => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/trainprocess/stop`
  });
};

export const retrain = (config: TrainingConfig) => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/trainprocess/retrain`,
    data: config
  });
};

export const startService = (config: TrainingConfig) => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/kernel2/llama/start`,
    data: config
  });
};

export const getServiceStatus = () => {
  return Request<CommonResponse<ServiceStatusRes>>({
    method: 'get',
    url: `/api/kernel2/llama/status`
  });
};

export const stopService = () => {
  return Request<CommonResponse<EmptyResponse>>({
    method: 'post',
    url: `/api/kernel2/llama/stop`
  });
};

export const getModelName = () => {
  return Request<CommonResponse<TrainingConfig>>({
    method: 'get',
    url: `/api/trainprocess/model_name`
  });
};

export const getTrainingParams = () => {
  return Request<CommonResponse<TrainingParams>>({
    method: 'get',
    url: `/api/trainprocess/training_params`
  });
};
