/**
 * Utilities for managing cloud model state
 */

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
 * Get display name for current model (cloud or local)
 */
export const getCurrentModelDisplayName = (): string => {
  const cloudModel = getActiveCloudModel();

  if (cloudModel) {
    return `Cloud: ${cloudModel.name}`;
  }

  return 'Local Model';
};
