/**
 * Utilities for managing cloud model state
 */

import { getServiceStatus } from '../service/train';
import { getCloudServiceStatus } from '../service/cloudService';

export interface ActiveCloudModel {
  name: string;
  deployed_model: string;
  base_model: string;
  status: string;
}

/**
 * Check if a cloud model is currently active
 */
export const isCloudModelActive = (): boolean => {
  try {
    const cloudModelInfo = localStorage.getItem('activeCloudModel');

    return !!cloudModelInfo;
  } catch (error) {
    console.error('Failed to check cloud model status:', error);

    return false;
  }
};

/**
 * Get the active cloud model information
 */
export const getActiveCloudModel = (): ActiveCloudModel | null => {
  try {
    const cloudModelInfo = localStorage.getItem('activeCloudModel');

    if (!cloudModelInfo) return null;

    return JSON.parse(cloudModelInfo) as ActiveCloudModel;
  } catch (error) {
    console.error('Failed to parse cloud model info:', error);

    return null;
  }
};

/**
 * Set the active cloud model
 */
export const setActiveCloudModel = (model: ActiveCloudModel): void => {
  try {
    localStorage.setItem('activeCloudModel', JSON.stringify(model));
  } catch (error) {
    console.error('Failed to save cloud model info:', error);
  }
};

/**
 * Clear the active cloud model
 */
export const clearActiveCloudModel = (): void => {
  try {
    localStorage.removeItem('activeCloudModel');
  } catch (error) {
    console.error('Failed to clear cloud model info:', error);
  }
};

/**
 * Extract clean model name from full model path or name
 */
const extractModelName = (modelName: string): string => {
  if (!modelName) return 'Unknown Model';
  
  // Remove file extensions
  let cleanName = modelName.replace(/\.(gguf|bin|safetensors)$/i, '');
  
  // If it contains a path separator, take the directory name (base model)
  if (cleanName.includes('/')) {
    cleanName = cleanName.split('/')[0];
  }
  
  return cleanName;
};

/**
 * Get display name for current model (cloud or local)
 * This function checks localStorage for model information as a fallback
 * For real-time status, use getCurrentModelDisplayNameAsync instead
 */
export const getCurrentModelDisplayName = (): string => {
  // Check cloud model first
  const cloudModel = getActiveCloudModel();

  if (cloudModel && cloudModel.status === 'active') {
    return `Cloud: ${extractModelName(cloudModel.name)}`;
  }

  // Check local model from training params
  try {
    const storedParams = localStorage.getItem('trainingParams');

    if (storedParams) {
      const params = JSON.parse(storedParams);

      if (params.model_name) {
        return `Local: ${extractModelName(params.model_name)}`;
      }
    }
  } catch {
    // Ignore parsing error
  }

  // Default fallback
  return 'Model Status Unknown';
};

/**
 * Get current model display name with real-time service status check
 * This function fetches the actual service status from backend
 */
export const getCurrentModelDisplayNameAsync = async (): Promise<string> => {
  try {
    // Check both local and cloud service status
    const [localRes, cloudRes] = await Promise.allSettled([
      getServiceStatus(),
      getCloudServiceStatus()
    ]);

    // Check cloud service first
    if (
      cloudRes.status === 'fulfilled' &&
      cloudRes.value.data.code === 0 &&
      cloudRes.value.data.data.status === 'active'
    ) {
      const modelData = cloudRes.value.data.data.model_data;

      if (modelData) {
        return `Cloud: ${extractModelName(modelData.model_name || 'Unknown Cloud Model')}`;
      }

      return 'Cloud Model';
    }

    // Check local service
    if (
      localRes.status === 'fulfilled' &&
      localRes.value.data.code === 0 &&
      localRes.value.data.data.is_running
    ) {
      // Get local model name from stored data if available
      const storedParams = localStorage.getItem('trainingParams');

      if (storedParams) {
        try {
          const params = JSON.parse(storedParams);

          return `Local: ${extractModelName(params.model_name || 'Local Model')}`;
        } catch {
          // Ignore parsing error
        }
      }

      return 'Local Model';
    }

    // If no service is running, return default
    return 'No Model Running';
  } catch (error) {
    console.error('Failed to get current service status:', error);
    // Fallback to localStorage method

    return getCurrentModelDisplayName();
  }
};
