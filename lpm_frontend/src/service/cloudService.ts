import { Request } from '../utils/request';
import type { CommonResponse } from '../types/responseModal';

export interface CloudModel {
  model_id: string;
  model_name: string;
  description?: string;
  created_at?: string;
}

export const listAvailableModels = () => {
  return Request<CommonResponse<CloudModel[]>>({
    method: 'get',
    url: '/api/cloud_service/list_available_models'
  });
};

export interface TrainingJobInfo {
  job_id: string;
  timestamp: string;
  status: string;
  model_id?: string;
  message?: string;
}

export interface CloudTrainingStatus {
  status: string; // e.g., "PROCESSING", "SUCCEEDED", "FAILED"
  model_id?: string; // Available when SUCCEEDED
  message?: string;
  details?: Record<string, unknown>; // For more detailed status information
}

export const startCloudTraining = (params: {
  base_model: string; // Corresponds to model_name in TrainingConfig
  training_type?: string; // e.g., "efficient_sft" or other types your backend supports
  hyper_parameters?: Record<string, unknown>;
  // Add any other parameters your /api/cloud_service/train/start endpoint expects
}) => {
  return Request<CommonResponse<unknown>>({
    method: 'post',
    url: '/api/cloud_service/train/start',
    data: params
  });
};

export const searchJobInfo = () => {
  return Request<CommonResponse<TrainingJobInfo>>({
    method: 'get',
    url: '/api/cloud_service/train/search_job_info'
  });
};

export const getCloudTrainingStatus = (jobId: string) => {
  return Request<CommonResponse<CloudTrainingStatus>>({
    method: 'get',
    url: `/api/cloud_service/train/status/training/${jobId}`
  });
};

export interface CloudDeployment {
  base_model: string;
  deployed_model: string;
  name: string;
  status: string; // e.g., "RUNNING", "PENDING", "STOPPED"
}

export interface DeploymentListResponse {
  count: number;
  deployments: CloudDeployment[];
}

export const listDeployments = () => {
  return Request<CommonResponse<DeploymentListResponse>>({
    method: 'get',
    url: '/api/cloud_service/train/list_deployments'
  });
};
